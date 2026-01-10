// fsevents_client/main.go
// 独立的 FSEvents 监控程序 - 通过 Unix Socket 发送事件
package main

/*
#cgo LDFLAGS: -framework CoreServices -framework CoreFoundation
#include <CoreServices/CoreServices.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>
#include <stdio.h>

// 全局 Go 回调声明
extern void goFSEventCallback(int numEvents, void *eventPaths, void *eventFlags);

// C 回调函数
static void fsEventCallback(
    ConstFSEventStreamRef streamRef,
    void *clientCallBackInfo,
    size_t numEvents,
    void *eventPaths,
    const FSEventStreamEventFlags eventFlags[],
    const FSEventStreamEventId eventIds[])
{
    goFSEventCallback((int)numEvents, eventPaths, (void*)eventFlags);
}

// 创建 FSEventStream
static inline FSEventStreamRef createEventStream(CFArrayRef pathsToWatch) {
    FSEventStreamContext context = {0, NULL, NULL, NULL, NULL};

    FSEventStreamRef stream = FSEventStreamCreate(
        kCFAllocatorDefault,
        (FSEventStreamCallback)&fsEventCallback,
        &context,
        pathsToWatch,
        kFSEventStreamEventIdSinceNow,
        0.1,  // 100ms 延迟
        kFSEventStreamCreateFlagFileEvents |
        kFSEventStreamCreateFlagNoDefer |
        kFSEventStreamCreateFlagWatchRoot
    );

    return stream;
}

// 辅助函数
static inline CFStringRef createCFString(const char *str) {
    return CFStringCreateWithCString(kCFAllocatorDefault, str, kCFStringEncodingUTF8);
}

static inline CFArrayRef createCFArray(CFStringRef *strings, int count) {
    return CFArrayCreate(
        kCFAllocatorDefault,
        (const void **)strings,
        count,
        &kCFTypeArrayCallBacks
    );
}
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"os/user"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// FSEvent 事件结构
type FSEvent struct {
	Timestamp string `json:"timestamp"`
	EventType string `json:"event_type"`
	Path      string `json:"path"`
}

// FSEvents 事件类型标志
const (
	kFSEventFlagItemCreated  = 0x00000100
	kFSEventFlagItemRemoved  = 0x00000200
	kFSEventFlagItemRenamed  = 0x00000800
	kFSEventFlagItemModified = 0x00001000
	kFSEventFlagItemIsFile   = 0x00010000
	kFSEventFlagItemIsDir    = 0x00020000
)

// 默认 socket 路径
const DefaultSocketPath = "/tmp/fsevents_monitor.sock"

var (
	socketConn   net.Conn
	connMutex    sync.Mutex
	running      = true
	homeDir      string
	recentEvents = make(map[string]time.Time)
	eventMutex   sync.Mutex
)

func main() {
	// 获取真实用户的 home 目录（考虑 sudo 环境）
	homeDir = getRealUserHomeDir()

	// 解析参数
	socketPath := DefaultSocketPath
	if len(os.Args) > 1 {
		socketPath = os.Args[1]
	}

	fmt.Println("🎯 FSEvents 独立监控进程")
	fmt.Printf("📡 Socket 路径: %s\n", socketPath)
	fmt.Printf("📁 监控目录: %s/{Desktop,Documents,Downloads}\n", homeDir)

	// 连接到服务器
	var err error
	socketConn, err = net.Dial("unix", socketPath)
	if err != nil {
		log.Fatalf("❌ 连接 socket 失败: %v", err)
	}
	defer socketConn.Close()

	// 禁用写缓冲，确保数据立即发送
	if unixConn, ok := socketConn.(*net.UnixConn); ok {
		unixConn.SetWriteBuffer(0)
	}

	fmt.Println("✅ 已连接到主服务器")

	// 监控的目录
	watchPaths := []string{
		filepath.Join(homeDir, "Desktop"),
		filepath.Join(homeDir, "Documents"),
		filepath.Join(homeDir, "Downloads"),
	}

	// 创建 CFArray
	pathsArray := createCFArrayFromPaths(watchPaths)
	defer C.CFRelease(C.CFTypeRef(pathsArray))

	// 创建事件流
	stream := C.createEventStream(pathsArray)
	if stream == nil {
		log.Fatal("❌ 无法创建 FSEvent 流")
	}

	// 使用 dispatch queue 启动（在独立进程中可以正常工作）
	go func() {
		queue := C.dispatch_get_global_queue(C.long(C.DISPATCH_QUEUE_PRIORITY_HIGH), 0)
		C.FSEventStreamSetDispatchQueue(stream, queue)
		C.FSEventStreamStart(stream)
	}()

	fmt.Println("🚀 FSEvents 监控已启动")

	// 等待退出信号
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	fmt.Println("\n🛑 正在停止...")
	running = false

	C.FSEventStreamStop(stream)
	C.FSEventStreamInvalidate(stream)
	C.FSEventStreamRelease(stream)

	fmt.Println("👋 再见!")
}

// 创建 CFArray
func createCFArrayFromPaths(paths []string) C.CFArrayRef {
	cfStrings := make([]C.CFStringRef, len(paths))

	for i, path := range paths {
		cStr := C.CString(path)
		cfStrings[i] = C.createCFString(cStr)
		C.free(unsafe.Pointer(cStr))
	}

	return C.createCFArray(&cfStrings[0], C.int(len(cfStrings)))
}

// Go 回调函数 (从 C 调用)
//
//export goFSEventCallback
func goFSEventCallback(numEvents C.int, eventPaths unsafe.Pointer, eventFlags unsafe.Pointer) {
	if !running {
		return
	}

	// 转换 C 数组为 Go 切片
	paths := (*[1 << 20]*C.char)(eventPaths)[:numEvents:numEvents]
	flags := (*[1 << 20]C.uint)(eventFlags)[:numEvents:numEvents]

	// 处理每个事件
	for i := 0; i < int(numEvents); i++ {
		path := C.GoString(paths[i])
		flag := uint32(flags[i])
		handleEvent(path, flag)
	}
}

func handleEvent(path string, flags uint32) {
	// 过滤系统路径和隐藏文件
	if shouldIgnore(path) {
		return
	}

	// 确定事件类型（按优先级判断）
	// FSEvents 可能同时设置多个标志，需要按优先级处理
	eventType := ""
	isDir := (flags & kFSEventFlagItemIsDir) != 0

	// 删除事件优先级最高
	if (flags & kFSEventFlagItemRemoved) != 0 {
		eventType = "deleted"
	} else if (flags & kFSEventFlagItemCreated) != 0 {
		eventType = "created"
	} else if (flags & kFSEventFlagItemRenamed) != 0 {
		eventType = "renamed"
	} else if (flags & kFSEventFlagItemModified) != 0 {
		// 目录的 modified 事件通常不重要（内容变化）
		if isDir {
			return
		}
		eventType = "modified"
	} else {
		return
	}

	// 去重检查
	key := eventType + ":" + path
	eventMutex.Lock()
	if lastTime, exists := recentEvents[key]; exists {
		if time.Since(lastTime) < 500*time.Millisecond {
			eventMutex.Unlock()
			return
		}
	}
	recentEvents[key] = time.Now()
	eventMutex.Unlock()

	// 创建事件（关键：立即记录时间戳）
	event := FSEvent{
		Timestamp: time.Now().Format("2006-01-02T15:04:05.000"),
		EventType: eventType,
		Path:      path,
	}

	// 发送到服务器
	sendEvent(event)

	fmt.Printf("📄 [%s] %s\n", eventType, filepath.Base(path))
}

func sendEvent(event FSEvent) {
	connMutex.Lock()
	defer connMutex.Unlock()

	if socketConn == nil {
		return
	}

	data, err := json.Marshal(event)
	if err != nil {
		return
	}

	// 添加换行符作为消息分隔
	data = append(data, '\n')

	// 立即写入并检查错误
	_, err = socketConn.Write(data)
	if err != nil {
		log.Printf("⚠️ Socket 写入失败: %v", err)
	}
}

func shouldIgnore(path string) bool {
	// 系统路径前缀
	ignorePrefixes := []string{
		"/private/var",
		"/private/tmp",
		"/System/",
		"/Applications/",
		"/Volumes/",
		"/dev/",
		"/usr/",
	}
	for _, prefix := range ignorePrefixes {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}

	// Library 目录
	if strings.Contains(path, "/Library/") {
		return true
	}

	// 隐藏文件
	baseName := filepath.Base(path)
	if strings.HasPrefix(baseName, ".") {
		return true
	}
	if strings.Contains(path, "/.") {
		return true
	}

	// 临时文件
	if strings.Contains(baseName, "~tmp") || strings.HasPrefix(baseName, "~$") {
		return true
	}

	// 临时扩展名
	ext := strings.ToLower(filepath.Ext(path))
	tempExts := []string{".tmp", ".lock", ".dat", ".plist", ".db", ".log", ".crdownload", ".download", ".part"}
	for _, te := range tempExts {
		if ext == te {
			return true
		}
	}

	return false
}

// getRealUserHomeDir 获取真实用户的 home 目录
// 在 sudo 环境下，使用 SUDO_USER 获取原始用户
func getRealUserHomeDir() string {
	// 优先检查 SUDO_USER（表示通过 sudo 运行）
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		// 通过 lookup 获取用户信息
		if u, err := user.Lookup(sudoUser); err == nil && u.HomeDir != "" {
			return u.HomeDir
		}
		// 备选：直接构造路径
		return "/Users/" + sudoUser
	}

	// 非 sudo 环境，尝试获取当前用户
	if currentUser, err := user.Current(); err == nil && currentUser.HomeDir != "" {
		return currentUser.HomeDir
	}

	// 最后尝试 HOME 环境变量
	return os.Getenv("HOME")
}

// 定期清理事件缓存
func init() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		for range ticker.C {
			eventMutex.Lock()
			now := time.Now()
			for k, v := range recentEvents {
				if now.Sub(v) > 30*time.Second {
					delete(recentEvents, k)
				}
			}
			eventMutex.Unlock()
		}
	}()
}
