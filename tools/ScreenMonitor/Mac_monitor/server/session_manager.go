// server/session_manager.go
// 会话管理器 - 管理录制会话的文件结构和数据格式
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// SessionManager 会话管理器
type SessionManager struct {
	recordsDir     string
	currentSession *SessionInfo
	mutex          sync.Mutex
}

// SessionInfo 会话信息 - 增强版，包含数据格式版本信息
type SessionInfo struct {
	ID            string    `json:"id"`
	StartTime     time.Time `json:"start_time"`
	EndTime       time.Time `json:"end_time,omitempty"`
	Duration      float64   `json:"duration"`
	SessionDir    string    `json:"session_dir"`     // 会话文件夹路径
	VideoFile     string    `json:"video_file"`      // 相对于会话目录
	FullVideoPath string    `json:"full_video_path"` // 完整视频路径
	LogFile       string    `json:"log_file"`        // 监控日志
	KeyEventsFile string    `json:"key_events_file"` // 关键事件
	SummaryFile   string    `json:"summary_file"`    // 事件摘要
	// WindowLogFile string    `json:"window_log_file"`
	// ClipboardFile string    `json:"clipboard_file"`
	Status     string `json:"status"`
	RiskEvents int    `json:"risk_events"`
	Version    string `json:"version"` // 数据格式版本
}

// LogEntry 日志条目 - 与 Windows Monitor 保持一致
type LogEntry struct {
	Timestamp     string                 `json:"timestamp"`
	EventType     string                 `json:"event_type"`
	FilePath      string                 `json:"file_path"`
	FileName      string                 `json:"file_name"`
	FileSize      int64                  `json:"file_size"`
	FileExtension string                 `json:"file_extension"`
	ProcessInfo   ProcessInfo            `json:"process_info"`
	WindowInfo    WindowInfo             `json:"window_info"`
	UserInfo      UserInfo               `json:"user_info"`
	DiskInfo      DiskInfo               `json:"disk_info"`
	AppName       string                 `json:"app_name,omitempty"`
	UploadInfo    *UploadDetection       `json:"upload_detection,omitempty"`
	Extra         map[string]interface{} `json:"extra,omitempty"`
}

// FileMetadata 文件元数据
type FileMetadata struct {
	Size         int64     `json:"size"`
	ModifiedTime time.Time `json:"modified_time"`
	CreatedTime  time.Time `json:"created_time"`
	Extension    string    `json:"extension"`
	IsSensitive  bool      `json:"is_sensitive"`
}

// ProcessInfo 进程信息
type ProcessInfo struct {
	PID         string `json:"pid"`
	ProcessName string `json:"process_name"`
	ProcessPath string `json:"process_path"`
	CmdLine     string `json:"cmdline"`
}

// WindowInfo 窗口信息
type WindowInfo struct {
	WindowHandle string `json:"window_handle"`
	WindowTitle  string `json:"window_title"`
	WindowClass  string `json:"window_class"`
}

// UserInfo 用户信息
type UserInfo struct {
	Username string `json:"username"`
	Hostname string `json:"hostname"`
}

// DiskInfo 磁盘信息
type DiskInfo struct {
	DriveLetter string `json:"drive_letter"`
	DiskType    string `json:"disk_type"`
}

type UploadDetection struct {
	IsUpload      bool   `json:"is_upload"`
	AppName       string `json:"app_name"`
	UploadType    string `json:"upload_type"`
	OriginalFile  string `json:"original_file"`
	TempDirectory string `json:"temp_directory"`
}

// KeyEvent 关键事件 - 使用 LogEntry 结构
type KeyEvent LogEntry

// EventSummary 事件摘要
type EventSummary struct {
	TotalEvents    int            `json:"total_events"`
	EventTypes     map[string]int `json:"event_types"`
	FileExtensions map[string]int `json:"file_extensions"`
	Apps           map[string]int `json:"apps"`
	UploadCount    int            `json:"upload_count"`
	TimeRange      TimeRange      `json:"time_range"`
}

// TimeRange 时间范围
type TimeRange struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

// NewSessionManager 创建会话管理器
func NewSessionManager(recordsDir string) *SessionManager {
	return &SessionManager{
		recordsDir: recordsDir,
	}
}

// CreateSession 创建新会话
// 自动生成独立文件夹并严格遵循命名规范
func (sm *SessionManager) CreateSession() (*SessionInfo, error) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// 生成会话 ID 和目录路径，严格遵循格式
	sessionID := time.Now().Format("20060102_150405")
	sessionDir := filepath.Join(sm.recordsDir, fmt.Sprintf("session_%s", sessionID))

	// 创建会话目录结构
	dirs := []string{
		sessionDir,
		filepath.Join(sessionDir, "video"),
		filepath.Join(sessionDir, "logs"),
		filepath.Join(sessionDir, "key_events"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("创建会话目录 '%s' 失败: %v", dir, err)
		}
		log.Printf("📂 创建目录: %s", dir)
	}

	// 初始化会话信息，包含数据格式版本
	session := &SessionInfo{
		ID:            sessionID,
		StartTime:     time.Now(),
		SessionDir:    sessionDir,
		VideoFile:     fmt.Sprintf("video/recording_%s.mp4", sessionID),
		LogFile:       fmt.Sprintf("logs/monitor_%s.json", sessionID),
		KeyEventsFile: fmt.Sprintf("key_events/key_events_%s.json", sessionID),
		SummaryFile:   fmt.Sprintf("key_events/summary_%s.json", sessionID),
		// WindowLogFile: fmt.Sprintf("logs/windows_%s.json", sessionID),
		// ClipboardFile: fmt.Sprintf("logs/clipboard_%s.json", sessionID),
		Status:  "recording",
		Version: "1.0.0", // 当前数据格式版本
	}

	sm.currentSession = session
	log.Printf("🆕 会话创建成功: %s, ID: %s", sessionDir, sessionID)
	return session, nil
}

// GetCurrentSession 获取当前会话
func (sm *SessionManager) GetCurrentSession() *SessionInfo {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	return sm.currentSession
}

// CompleteSession 完成会话
func (sm *SessionManager) CompleteSession(keyEvents []KeyEvent) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if sm.currentSession == nil {
		return fmt.Errorf("没有活动的会话")
	}

	sm.currentSession.EndTime = time.Now()
	sm.currentSession.Duration = sm.currentSession.EndTime.Sub(sm.currentSession.StartTime).Seconds()
	sm.currentSession.Status = "completed"
	// 更新完整视频路径
	if sm.currentSession.VideoFile != "" && sm.currentSession.SessionDir != "" {
		sm.currentSession.FullVideoPath = fmt.Sprintf("/recordings/%s/%s", filepath.Base(sm.currentSession.SessionDir), sm.currentSession.VideoFile)
	}

	// 保存关键事件
	if err := sm.saveKeyEvents(keyEvents); err != nil {
		return err
	}

	// 生成摘要
	if err := sm.generateSummary(keyEvents); err != nil {
		return err
	}

	// 生成INDEX.md
	if err := sm.generateIndex(); err != nil {
		return err
	}

	sm.currentSession = nil
	return nil
}

// saveKeyEvents 保存关键事件
func (sm *SessionManager) saveKeyEvents(events []KeyEvent) error {
	if sm.currentSession == nil {
		return nil
	}

	filePath := filepath.Join(sm.currentSession.SessionDir, sm.currentSession.KeyEventsFile)
	data, err := json.MarshalIndent(events, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

// generateSummary 生成事件摘要
func (sm *SessionManager) generateSummary(events []KeyEvent) error {
	if sm.currentSession == nil {
		return nil
	}

	summary := EventSummary{
		TotalEvents:    len(events),
		EventTypes:     make(map[string]int),
		FileExtensions: make(map[string]int),
		Apps:           make(map[string]int),
		UploadCount:    0,
	}

	var startTime, endTime string
	for i, event := range events {
		// 统计事件类型
		summary.EventTypes[event.EventType]++

		// 统计文件扩展名
		if event.FileExtension != "" {
			summary.FileExtensions[event.FileExtension]++
		}

		// 统计应用
		if event.AppName != "" {
			summary.Apps[event.AppName]++
		}

		// 时间范围
		ts := fmt.Sprintf("%v", event.Timestamp)
		if i == 0 {
			startTime = ts
		}
		endTime = ts
	}

	summary.TimeRange = TimeRange{Start: startTime, End: endTime}

	filePath := filepath.Join(sm.currentSession.SessionDir, sm.currentSession.SummaryFile)
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

// generateIndex 生成INDEX.md索引文件，保持格式一致性
func (sm *SessionManager) generateIndex() error {
	if sm.currentSession == nil {
		return fmt.Errorf("没有活动的会话")
	}

	content := fmt.Sprintf(`# Recording Session Index

**Session ID**: %s  
**Recording Time**: %s  
**Recording Duration**: %.2f 秒  
**Data Format Version**: %s
**Risk Events**: %d

## File List

### Video Files
- `+"`%s`"+` - Recorded screen video

### Original Logs
- `+"`%s`"+` - Complete monitoring log

### Key Events
- `+"`%s`"+` - Extracted key events
- `+"`%s`"+` - Event statistics summary

---
*Auto-generated by Mac_monitor*`,
		sm.currentSession.ID,
		sm.currentSession.StartTime.Format("2006-01-02 15:04:05"),
		sm.currentSession.Duration,
		sm.currentSession.Version,
		sm.currentSession.RiskEvents,
		sm.currentSession.VideoFile,
		sm.currentSession.LogFile,
		sm.currentSession.KeyEventsFile,
		sm.currentSession.SummaryFile,
	)

	filePath := filepath.Join(sm.currentSession.SessionDir, "INDEX.md")
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return fmt.Errorf("写入INDEX.md失败: %v", err)
	}

	log.Printf("📄 生成索引文件: %s", filePath)
	return nil
}

// validateDataFormat 验证数据格式是否符合标准
func (sm *SessionManager) validateDataFormat() error {
	if sm.currentSession == nil {
		return fmt.Errorf("没有活动的会话")
	}

	// 验证会话目录存在性
	if _, err := os.Stat(sm.currentSession.SessionDir); os.IsNotExist(err) {
		return fmt.Errorf("会话目录不存在: %s", sm.currentSession.SessionDir)
	}

	// 可以在此添加更多的数据格式验证
	log.Println("✅ 数据格式验证通过")
	return nil
}

// GetVideoPath 获取视频文件完整路径
func (sm *SessionManager) GetVideoPath() string {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if sm.currentSession == nil {
		return ""
	}
	return filepath.Join(sm.currentSession.SessionDir, sm.currentSession.VideoFile)
}

// GetLogPath 获取日志文件完整路径
func (sm *SessionManager) GetLogPath() string {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if sm.currentSession == nil {
		return ""
	}
	return filepath.Join(sm.currentSession.SessionDir, sm.currentSession.LogFile)
}

// GetWindowLogPath 获取窗口日志文件完整路径
// func (sm *SessionManager) GetWindowLogPath() string {
// 	sm.mutex.Lock()
// 	defer sm.mutex.Unlock()
//
// 	if sm.currentSession == nil {
// 		return ""
// 	}
// 	// return filepath.Join(sm.currentSession.SessionDir, sm.currentSession.WindowLogFile)
//     return ""
// }

// GetClipboardPath 获取剪贴板日志文件完整路径
// func (sm *SessionManager) GetClipboardPath() string {
// 	sm.mutex.Lock()
// 	defer sm.mutex.Unlock()
//
// 	if sm.currentSession == nil {
// 		return ""
// 	}
// 	// return filepath.Join(sm.currentSession.SessionDir, sm.currentSession.ClipboardFile)
//     return ""
// }
