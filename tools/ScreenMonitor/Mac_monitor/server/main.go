// server/main.go
// macOS 录屏和 Log 抓取系统 - 后端服务
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// 全局状态
var (
	recorder          *ScreenRecorder
	fileMonitor       *FileMonitor
	unifiedLogMonitor *UnifiedLogMonitor
	// windowMonitor    *WindowMonitor
	// clipboardMonitor *ClipboardMonitor
	sessionManager *SessionManager
	mutex          sync.Mutex
	recordsDir     = "./recordings"
)

// Session 录制会话 - 增强版本，包含多种日志文件
type Session struct {
	ID            string    `json:"id"`
	StartTime     time.Time `json:"start_time"`
	EndTime       time.Time `json:"end_time,omitempty"`
	Duration      float64   `json:"duration"`    // 录制时长(秒)
	SessionDir    string    `json:"session_dir"` // 会话文件夹路径
	VideoFile     string    `json:"video_file"`
	FullVideoPath string    `json:"full_video_path"` // 完整视频路径用于前端播放
	LogFile       string    `json:"log_file"`        // 文件操作日志
	KeyEventsFile string    `json:"key_events_file"` // 关键事件
	SummaryFile   string    `json:"summary_file"`    // 事件摘要
	// WindowLogFile string    `json:"window_log_file"` // 窗口切换日志
	// ClipboardFile string    `json:"clipboard_file"`  // 剪贴板日志
	Status     string `json:"status"`      // recording, completed, error
	RiskEvents int    `json:"risk_events"` // 风险事件数量
}

var currentSession *Session
var sessions []Session

// API 响应结构
type Response struct {
	Success bool        `json:"success"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

func main() {
	// 创建录制目录
	if err := os.MkdirAll(recordsDir, 0755); err != nil {
		log.Fatalf("无法创建录制目录: %v", err)
	}

	// 初始化控制器
	recorder = NewScreenRecorder(recordsDir)
	// windowMonitor = NewWindowMonitor()
	fileMonitor = NewFileMonitor(recordsDir, nil) // windowMonitor passed as nil
	unifiedLogMonitor = NewUnifiedLogMonitor("./macos-UnifiedLogs/examples/monitor/target/release/unified_log_monitor")
	// clipboardMonitor = NewClipboardMonitor()
	sessionManager = NewSessionManager(recordsDir)

	// 加载历史会话
	loadSessions()

	// 设置路由
	// 设置路由 - Align with Windows Monitor API
	http.HandleFunc("/api/recording/status", corsMiddleware(handleStatus))
	http.HandleFunc("/api/recording/start", corsMiddleware(handleStart))
	http.HandleFunc("/api/recording/stop", corsMiddleware(handleStop))
	http.HandleFunc("/api/sessions", corsMiddleware(handleSessions))
	http.HandleFunc("/api/sessions/", corsMiddleware(handleSessionDetail)) // Note: Windows uses /api/sessions/<id>
	http.HandleFunc("/api/logs/", corsMiddleware(handleLogs))
	// http.HandleFunc("/api/windows/", corsMiddleware(handleWindowLogs))
	// http.HandleFunc("/api/clipboard/", corsMiddleware(handleClipboardLogs))
	http.HandleFunc("/api/key-events/", corsMiddleware(handleKeyEvents))
	http.HandleFunc("/api/summary/", corsMiddleware(handleSummary))
	http.HandleFunc("/api/all-events/", corsMiddleware(handleAllEvents))

	// 静态文件服务 - 录制文件（支持视频、日志等完整结构访问）
	http.Handle("/recordings/", http.StripPrefix("/recordings/",
		http.FileServer(http.Dir(recordsDir))))

	port := ":8081"
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("🚨 macOS 数据泄露行为监控系统")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("🚀 服务器运行在 http://localhost%s\n", port)
	fmt.Println("📁 录制文件保存在:", recordsDir)
	fmt.Println("📊 监控功能: 录屏 | 文件操作 | 窗口切换 | 剪贴板")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	log.Fatal(http.ListenAndServe(port, nil))
}

// CORS 中间件
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// 获取当前状态
func handleStatus(w http.ResponseWriter, r *http.Request) {
	mutex.Lock()
	defer mutex.Unlock()

	status := map[string]interface{}{
		"recording": recorder.IsRecording(),
		"session":   currentSession,
	}

	sendJSON(w, Response{Success: true, Data: status})
}

// 开始录制
func handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		sendError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	if recorder.IsRecording() {
		sendError(w, "已经在录制中", http.StatusBadRequest)
		return
	}

	// 使用会话管理器创建新会话
	sessionInfo, err := sessionManager.CreateSession()
	if err != nil {
		sendError(w, fmt.Sprintf("创建会话失败: %v", err), http.StatusInternalServerError)
		return
	}

	// 创建当前会话对象
	currentSession = &Session{
		ID:            sessionInfo.ID,
		StartTime:     sessionInfo.StartTime,
		SessionDir:    sessionInfo.SessionDir,
		VideoFile:     sessionInfo.VideoFile,                                                                          // 使用会话管理器提供的包含 video/ 子目录的路径
		FullVideoPath: fmt.Sprintf("/recordings/%s/%s", filepath.Base(sessionInfo.SessionDir), sessionInfo.VideoFile), // 正确构建路径
		LogFile:       sessionInfo.LogFile,
		KeyEventsFile: sessionInfo.KeyEventsFile,
		SummaryFile:   sessionInfo.SummaryFile,
		// WindowLogFile: sessionInfo.WindowLogFile,
		// ClipboardFile: sessionInfo.ClipboardFile,
		Status: "recording",
	}

	// 解析请求参数
	var req struct {
		FPS int `json:"fps"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Ignore error, use defaults
	}

	// 启动录屏
	videoPath := filepath.Join(sessionInfo.SessionDir, sessionInfo.VideoFile)
	if err := recorder.Start(videoPath, req.FPS); err != nil {
		currentSession = nil
		sendError(w, fmt.Sprintf("启动录屏失败: %v", err), http.StatusInternalServerError)
		return
	}

	// 启动窗口监控 (先启动，因为文件监控需要它)
	// windowPath := filepath.Join(sessionInfo.SessionDir, sessionInfo.WindowLogFile)
	// if err := windowMonitor.Start(windowPath); err != nil {
	// 	log.Printf("⚠️ 启动窗口监控失败: %v", err)
	// }

	// 启动文件监控 (增强版)
	logPath := filepath.Join(sessionInfo.SessionDir, sessionInfo.LogFile)
	if err := fileMonitor.Start(logPath); err != nil {
		log.Printf("⚠️ 启动文件监控失败: %v", err)
	}

	// 启动 Unified Log Monitor (辅助)
	if err := unifiedLogMonitor.Start(); err != nil {
		log.Printf("⚠️ 启动 Unified Log Monitor 失败: %v", err)
	}

	// 启动剪贴板监控
	// clipboardPath := filepath.Join(sessionInfo.SessionDir, sessionInfo.ClipboardFile)
	// if err := clipboardMonitor.Start(clipboardPath); err != nil {
	// 	log.Printf("⚠️ 启动剪贴板监控失败: %v", err)
	// }

	log.Printf("✅ 录制开始: %s (会话目录: %s)", sessionInfo.ID, sessionInfo.SessionDir)
	sendJSON(w, Response{Success: true, Message: "录制已开始", Data: currentSession})
}

// 停止录制
func handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		sendError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	mutex.Lock()
	defer mutex.Unlock()

	if !recorder.IsRecording() {
		sendError(w, "没有正在进行的录制", http.StatusBadRequest)
		return
	}

	// 停止录屏
	if err := recorder.Stop(); err != nil {
		log.Printf("停止录屏出错: %v", err)
	}

	// 停止文件监控
	var keyEvents []KeyEvent
	if fileMonitor.IsRunning() {
		keyEvents = fileMonitor.GetKeyEvents()
		if err := fileMonitor.Stop(); err != nil {
			log.Printf("停止文件监控出错: %v", err)
		}
	}

	// 停止 Unified Log Monitor
	if err := unifiedLogMonitor.Stop(); err != nil {
		log.Printf("停止 Unified Log Monitor 出错: %v", err)
	}

	// 停止窗口监控
	// if err := windowMonitor.Stop(); err != nil {
	// 	log.Printf("停止窗口监控出错: %v", err)
	// }

	// 停止剪贴板监控
	// if err := clipboardMonitor.Stop(); err != nil {
	// 	log.Printf("停止剪贴板监控出错: %v", err)
	// }

	// 更新会话状态
	if currentSession != nil {
		currentSession.EndTime = time.Now()
		currentSession.Duration = currentSession.EndTime.Sub(currentSession.StartTime).Seconds()
		currentSession.Status = "completed"

		// 使用会话管理器完成会话（生成关键事件、摘要和INDEX）
		if err := sessionManager.CompleteSession(keyEvents); err != nil {
			log.Printf("完成会话出错: %v", err)
		}

		// 统计风险事件数量
		currentSession.RiskEvents = len(keyEvents)

		sessions = append(sessions, *currentSession)
		saveSessions()

		log.Printf("✅ 录制完成: %s (风险事件: %d)", currentSession.ID, currentSession.RiskEvents)
	}

	result := currentSession
	currentSession = nil

	sendJSON(w, Response{Success: true, Message: "录制已停止", Data: result})
}

// 获取所有会话
func handleSessions(w http.ResponseWriter, r *http.Request) {
	mutex.Lock()
	defer mutex.Unlock()

	sendJSON(w, Response{Success: true, Data: sessions})
}

// 获取会话详情
func handleSessionDetail(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	mutex.Lock()
	defer mutex.Unlock()

	for _, s := range sessions {
		if s.ID == sessionID {
			sendJSON(w, Response{Success: true, Data: s})
			return
		}
	}

	sendError(w, "会话不存在", http.StatusNotFound)
}

// 获取会话 logs
func handleLogs(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	// 尝试新的会话目录结构
	logFile := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID), "logs", fmt.Sprintf("monitor_%s.json", sessionID))

	// 如果新结构不存在，尝试旧结构
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		logFile = filepath.Join(recordsDir, fmt.Sprintf("logs_%s.json", sessionID))
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		sendError(w, "Log 文件不存在", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// 辅助函数
func sendJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func sendError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(Response{Success: false, Message: message})
}

func saveSessions() {
	data, _ := json.MarshalIndent(sessions, "", "  ")
	os.WriteFile(filepath.Join(recordsDir, "sessions.json"), data, 0644)
}

func loadSessions() {
	data, err := os.ReadFile(filepath.Join(recordsDir, "sessions.json"))
	if err != nil {
		sessions = []Session{}
		return
	}
	json.Unmarshal(data, &sessions)

	// 为所有会话补全 FullVideoPath 字段
	for i, s := range sessions {
		// 总是补全 FullVideoPath 字段，确保路径正确
		if s.SessionDir != "" && s.VideoFile != "" {
			sessions[i].FullVideoPath = fmt.Sprintf("/recordings/%s/%s", filepath.Base(s.SessionDir), s.VideoFile)
		}
	}
}

// 获取窗口切换日志
func handleWindowLogs(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	// 尝试新的会话目录结构
	logFile := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID), "logs", fmt.Sprintf("windows_%s.json", sessionID))

	// 如果新结构不存在，尝试旧结构
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		logFile = filepath.Join(recordsDir, fmt.Sprintf("windows_%s.json", sessionID))
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		sendError(w, "窗口日志文件不存在", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// 获取剪贴板日志
func handleClipboardLogs(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	// 尝试新的会话目录结构
	logFile := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID), "logs", fmt.Sprintf("clipboard_%s.json", sessionID))

	// 如果新结构不存在，尝试旧结构
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		logFile = filepath.Join(recordsDir, fmt.Sprintf("clipboard_%s.json", sessionID))
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		sendError(w, "剪贴板日志文件不存在", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// 获取关键事件
func handleKeyEvents(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	// 尝试新的会话目录结构
	logFile := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID), "key_events", fmt.Sprintf("key_events_%s.json", sessionID))

	data, err := os.ReadFile(logFile)
	if err != nil {
		sendError(w, "关键事件文件不存在", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// 获取事件摘要
func handleSummary(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	// 尝试新的会话目录结构
	logFile := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID), "key_events", fmt.Sprintf("summary_%s.json", sessionID))

	data, err := os.ReadFile(logFile)
	if err != nil {
		sendError(w, "事件摘要文件不存在", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// AllEventsResponse 所有事件的综合响应
type AllEventsResponse struct {
	SessionID     string           `json:"session_id"`
	FileLogs      []LogEntry       `json:"file_logs"`
	WindowEvents  []WindowEvent    `json:"window_events"`
	ClipboardLogs []ClipboardEvent `json:"clipboard_logs"`
	RiskSummary   RiskSummary      `json:"risk_summary"`
}

// RiskSummary 风险摘要
type RiskSummary struct {
	TotalEvents      int            `json:"total_events"`
	HighRiskEvents   int            `json:"high_risk_events"`
	MediumRiskEvents int            `json:"medium_risk_events"`
	AppUsage         map[string]int `json:"app_usage"`
	SensitiveFiles   []string       `json:"sensitive_files"`
}

// 获取所有事件的综合视图
func handleAllEvents(w http.ResponseWriter, r *http.Request) {
	sessionID := filepath.Base(r.URL.Path)

	response := AllEventsResponse{
		SessionID: sessionID,
		RiskSummary: RiskSummary{
			AppUsage:       make(map[string]int),
			SensitiveFiles: []string{},
		},
	}

	// 尝试新结构路径
	sessionDir := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID))
	useNewStructure := false
	if _, err := os.Stat(sessionDir); err == nil {
		useNewStructure = true
	}

	// 读取文件操作日志
	var logFilePath string
	if useNewStructure {
		logFilePath = filepath.Join(sessionDir, "logs", fmt.Sprintf("monitor_%s.json", sessionID))
	} else {
		logFilePath = filepath.Join(recordsDir, fmt.Sprintf("logs_%s.json", sessionID))
	}

	if data, err := os.ReadFile(logFilePath); err == nil {
		var logs []LogEntry
		scanner := bufio.NewScanner(bytes.NewReader(data))
		// Increase buffer size for long lines
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)

		for scanner.Scan() {
			var logEntry LogEntry
			if err := json.Unmarshal(scanner.Bytes(), &logEntry); err == nil {
				logs = append(logs, logEntry)

				// 聚合统计
				if logEntry.UploadInfo != nil && logEntry.UploadInfo.IsUpload {
					if logEntry.UploadInfo.UploadType == "Sensitive Access" {
						response.RiskSummary.HighRiskEvents++
					} else {
						response.RiskSummary.MediumRiskEvents++
					}
					response.RiskSummary.SensitiveFiles = append(response.RiskSummary.SensitiveFiles, logEntry.FilePath)
				}
				if logEntry.AppName != "" {
					response.RiskSummary.AppUsage[logEntry.AppName]++
				}
			}
		}
		response.FileLogs = logs
	}

	// 读取窗口事件
	var windowFilePath string
	if useNewStructure {
		windowFilePath = filepath.Join(sessionDir, "logs", fmt.Sprintf("windows_%s.json", sessionID))
	} else {
		windowFilePath = filepath.Join(recordsDir, fmt.Sprintf("windows_%s.json", sessionID))
	}
	if data, err := os.ReadFile(windowFilePath); err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		for scanner.Scan() {
			var event WindowEvent
			if err := json.Unmarshal(scanner.Bytes(), &event); err == nil {
				response.WindowEvents = append(response.WindowEvents, event)
				if event.RiskLevel == "高" {
					response.RiskSummary.HighRiskEvents++
				}
				response.RiskSummary.AppUsage[event.AppName]++
			}
		}
	}

	// 读取剪贴板日志
	var clipboardFilePath string
	if useNewStructure {
		clipboardFilePath = filepath.Join(sessionDir, "logs", fmt.Sprintf("clipboard_%s.json", sessionID))
	} else {
		clipboardFilePath = filepath.Join(recordsDir, fmt.Sprintf("clipboard_%s.json", sessionID))
	}
	if data, err := os.ReadFile(clipboardFilePath); err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		for scanner.Scan() {
			var logEntry ClipboardEvent
			if err := json.Unmarshal(scanner.Bytes(), &logEntry); err == nil {
				response.ClipboardLogs = append(response.ClipboardLogs, logEntry)
				if logEntry.IsSensitive {
					response.RiskSummary.MediumRiskEvents++
				}
			}
		}
	}

	response.RiskSummary.TotalEvents = len(response.FileLogs) + len(response.WindowEvents) + len(response.ClipboardLogs)

	sendJSON(w, Response{Success: true, Data: response})
}

// 统计风险事件数量
func countRiskEvents(sessionID string) int {
	count := 0

	// 尝试新结构路径
	sessionDir := filepath.Join(recordsDir, fmt.Sprintf("session_%s", sessionID))
	useNewStructure := false
	if _, err := os.Stat(sessionDir); err == nil {
		useNewStructure = true
	}

	// 统计文件操作日志中的风险事件
	var logFilePath string
	if useNewStructure {
		logFilePath = filepath.Join(sessionDir, "logs", fmt.Sprintf("monitor_%s.json", sessionID))
	} else {
		logFilePath = filepath.Join(recordsDir, fmt.Sprintf("logs_%s.json", sessionID))
	}
	if data, err := os.ReadFile(logFilePath); err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		for scanner.Scan() {
			var logEntry LogEntry
			if err := json.Unmarshal(scanner.Bytes(), &logEntry); err == nil {
				if logEntry.UploadInfo != nil && logEntry.UploadInfo.IsUpload {
					count++
				}
			}
		}
	}

	// 统计窗口事件中的风险事件
	var windowFilePath string
	if useNewStructure {
		windowFilePath = filepath.Join(sessionDir, "logs", fmt.Sprintf("windows_%s.json", sessionID))
	} else {
		windowFilePath = filepath.Join(recordsDir, fmt.Sprintf("windows_%s.json", sessionID))
	}
	if data, err := os.ReadFile(windowFilePath); err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		for scanner.Scan() {
			var event WindowEvent
			if err := json.Unmarshal(scanner.Bytes(), &event); err == nil {
				if event.RiskLevel != "" {
					count++
				}
			}
		}
	}

	return count
}
