// server/recorder.go
// 使用 ffmpeg 进行屏幕录制
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// ScreenRecorder 屏幕录制器
type ScreenRecorder struct {
	recordsDir  string
	cmd         *exec.Cmd
	recording   bool
	mutex       sync.Mutex
	stopChan    chan bool
	eventBuffer chan *LogEntry // 使用 LogEntry 类型而不是 MonitorEvent
}

// NewScreenRecorder 创建录屏器
func NewScreenRecorder(recordsDir string) *ScreenRecorder {
	return &ScreenRecorder{
		recordsDir:  recordsDir,
		stopChan:    make(chan bool),
		eventBuffer: make(chan *LogEntry, 1000), // 使用 LogEntry 类型
	}
}

// AddEvent 添加文件操作事件到缓冲区
func (r *ScreenRecorder) AddEvent(event *LogEntry) { // 使用 LogEntry 类型
	r.mutex.Lock()
	isRecording := r.recording
	r.mutex.Unlock()

	if !isRecording {
		return
	}

	// 非阻塞地将事件添加到缓冲区
	select {
	case r.eventBuffer <- event:
		// 事件添加成功
	default:
		// 缓冲区满，记录警告
		log.Println("Event buffer is full, dropping event")
	}
}

// processEvents 处理事件缓冲区中的事件
func (r *ScreenRecorder) processEvents() {
	for {
		select {
		case event, ok := <-r.eventBuffer:
			if !ok {
				// 通道已关闭
				return
			}
			// 处理事件 - 目前我们只打印日志，后续可以扩展处理逻辑
			log.Printf("File operation event: %s %s", event.AppName, event.FilePath)

		case <-r.stopChan:
			// 收到停止信号，退出处理循环
			return
		}
	}
}

// IsRecording 检查是否正在录制
func (r *ScreenRecorder) IsRecording() bool {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	return r.recording
}

// Start 开始录屏
func (r *ScreenRecorder) Start(outputPath string, fps int) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if r.recording {
		return fmt.Errorf("已经在录制中")
	}

	// 检查 ffmpeg 是否可用
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("ffmpeg 未安装，请运行: brew install ffmpeg")
	}

	if fps <= 0 {
		fps = 10 // Default to 10 to match Windows monitor default
	}

	// 构建 ffmpeg 命令
	// 注意：macOS 屏幕捕获支持的像素格式：uyvy422, yuyv422, nv12, 0rgb, bgr0
	// 我们使用 uyvy422 捕获，然后通过 scale 过滤器转换为 yuv420p 以获得更好的压缩
	args := []string{
		"-f", "avfoundation",
		"-capture_cursor", "1", // 捕获鼠标光标
		"-capture_mouse_clicks", "1", // 显示鼠标点击
		"-framerate", fmt.Sprintf("%d", fps), // 动态帧率
		"-i", "2:none", // 屏幕捕获设备(Capture screen 0):无音频 - Index 2 based on `ffmpeg -list_devices`
		"-vf", "scale=in_range=full:out_range=full", // 颜色空间转换
		"-c:v", "h264_videotoolbox", // 使用硬件加速编码
		"-pix_fmt", "nv12", // 使用 nv12 格式（硬件编码器支持）
		"-b:v", "5000k", // 比特率
		"-movflags", "+faststart", // 优化网页播放
		"-y", // 覆盖输出文件
		outputPath,
	}

	r.cmd = exec.Command("ffmpeg", args...)

	// 创建错误日志文件
	errorLogPath := outputPath + ".error.log"
	errorLog, err := os.Create(errorLogPath)
	if err != nil {
		log.Printf("⚠️ 无法创建错误日志文件: %v", err)
		r.cmd.Stderr = os.Stderr
	} else {
		r.cmd.Stderr = errorLog
		// 不要在这里关闭，等录制结束后关闭
	}
	r.cmd.Stdout = os.Stdout

	// 启动录制
	if err := r.cmd.Start(); err != nil {
		if errorLog != nil {
			errorLog.Close()
		}
		return fmt.Errorf("启动 ffmpeg 失败: %v", err)
	}

	// 等待一小段时间，确保 ffmpeg 真正启动
	time.Sleep(100 * time.Millisecond)

	// 检查进程是否还在运行
	if r.cmd.Process == nil {
		if errorLog != nil {
			errorLog.Close()
		}
		return fmt.Errorf("ffmpeg 进程启动后立即退出，请检查错误日志: %s", errorLogPath)
	}

	r.recording = true
	log.Printf("🎬 开始录屏: %s (错误日志: %s)", outputPath, errorLogPath)

	// 在后台等待进程结束
	go func() {
		r.cmd.Wait()
		if errorLog != nil {
			errorLog.Close()
		}
		r.mutex.Lock()
		r.recording = false
		r.mutex.Unlock()
	}()

	// 启动事件处理协程
	go r.processEvents()

	return nil
}

// Stop 停止录屏
func (r *ScreenRecorder) Stop() error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.recording || r.cmd == nil {
		return fmt.Errorf("没有正在进行的录制")
	}

	// 发送停止信号
	r.stopChan <- true

	// 关闭事件缓冲区
	close(r.eventBuffer)

	// 发送 SIGINT 信号让 ffmpeg 优雅退出（保存文件）
	if r.cmd.Process != nil {
		// 发送 SIGINT 信号让 ffmpeg 正常结束并保存文件
		if err := r.cmd.Process.Signal(os.Interrupt); err != nil {
			log.Printf("发送中断信号失败: %v, 尝试 SIGTERM", err)
			if err := r.cmd.Process.Signal(syscall.SIGTERM); err != nil {
				// 如果信号发送失败，强制终止
				r.cmd.Process.Kill()
			}
		}

		// 创建一个通道来等待进程结束
		done := make(chan error, 1)
		go func() {
			done <- r.cmd.Wait()
		}()

		// 等待进程结束或超时
		select {
		case err := <-done:
			if err != nil {
				log.Printf("ffmpeg 进程退出时出错: %v", err)
			} else {
				log.Println("✅ ffmpeg 进程已正常退出")
			}
		case <-time.After(5 * time.Second):
			log.Println("⚠️ ffmpeg 进程超时，强制终止")
			r.cmd.Process.Kill()
			<-done // 仍然等待 Wait() 完成
		}
	}

	r.recording = false
	log.Println("🎬 录屏已停止")

	return nil
}
