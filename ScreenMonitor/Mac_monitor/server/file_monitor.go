// server/file_monitor.go
// 增强版文件监控器 - 整合应用-文件关联
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// LogEntry and other structs are defined in session_manager.go to avoid duplication

// FileMonitor 增强版文件监控器
type FileMonitor struct {
	recordsDir    string
	cmd           *exec.Cmd
	running       bool
	mutex         sync.Mutex
	events        []LogEntry
	keyEvents     []KeyEvent
	logFile       *os.File
	myPid         string
	homeDir       string
	startTime     time.Time
	currentUser   string
	hostname      string
	windowMonitor *WindowMonitor // 关联窗口监控器以获取当前窗口信息
}

// 关注的操作
var fileOperationMap = map[string]string{
	"open":     "opened",
	"openat":   "opened",
	"create":   "created",
	"mkdir":    "created",
	"mkdirat":  "created",
	"rename":   "renamed",
	"renameat": "renamed",
	"unlink":   "deleted",
	"unlinkat": "deleted",
	"rmdir":    "deleted",
	"write":    "modified",
	"modify":   "modified",
	"pwrite":   "modified",
}

// 忽略的系统路径前缀
var fileIgnorePrefixes = []string{
	"/dev/",
	"/sys",
	"/private/var/",
	"/private/tmp/",
	"/bin/",
	"/usr/",
	"/sbin/",
	"/System/",
	"/System/Library/",
	"/tmp/",
	"/opt/",
	"/Library/",
	"/Applications/",
	"/.DocumentRevisions-V100/",
	"/.Spotlight-V100/",
	"/.fseventsd/",
	"/Volumes/",
}

// NewFileMonitor 创建文件监控器
func NewFileMonitor(recordsDir string, windowMonitor *WindowMonitor) *FileMonitor {
	homeDir := os.Getenv("HOME")
	username := "unknown"
	hostname := "unknown"

	if currentUser, err := user.Current(); err == nil {
		if currentUser.HomeDir != "" {
			homeDir = currentUser.HomeDir
		}
		username = currentUser.Username
	}

	if h, err := os.Hostname(); err == nil {
		hostname = h
	}

	return &FileMonitor{
		recordsDir:    recordsDir,
		myPid:         strconv.Itoa(os.Getpid()),
		homeDir:       homeDir,
		currentUser:   username,
		hostname:      hostname,
		windowMonitor: windowMonitor,
	}
}

// IsRunning 检查是否正在运行
func (fm *FileMonitor) IsRunning() bool {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()
	return fm.running
}

// Start 开始监控
func (fm *FileMonitor) Start(outputPath string) error {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	if fm.running {
		return fmt.Errorf("已经在运行中")
	}

	// 检查是否有 sudo 权限
	if os.Geteuid() != 0 {
		log.Println("⚠️ 警告: 文件监控需要 sudo 权限才能运行 fs_usage")
		log.Println("💡 请使用 sudo 运行服务器以启用完整的文件监控功能")
	}

	// 创建输出文件
	var err error
	fm.logFile, err = os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("创建日志文件失败: %v", err)
	}

	fm.events = []LogEntry{}
	fm.keyEvents = []KeyEvent{}
	fm.startTime = time.Now()

	// 启动 fs_usage
	fm.cmd = exec.Command("fs_usage", "-f", "filesys", "-w")
	stdout, err := fm.cmd.StdoutPipe()
	if err != nil {
		fm.logFile.Close()
		return fmt.Errorf("创建 stdout pipe 失败: %v", err)
	}

	if err := fm.cmd.Start(); err != nil {
		fm.logFile.Close()
		return fmt.Errorf("启动 fs_usage 失败: %v (需要 sudo 权限)", err)
	}

	fm.running = true
	log.Printf("📂 开始文件监控: %s", outputPath)

	// 在后台读取并解析输出
	go func() {
		scanner := bufio.NewScanner(stdout)
		buf := make([]byte, 0, 128*1024)
		scanner.Buffer(buf, 1024*1024)

		// 跳过头部
		for i := 0; i < 5; i++ {
			if !scanner.Scan() {
				break
			}
		}

		for scanner.Scan() {
			fm.mutex.Lock()
			if !fm.running {
				fm.mutex.Unlock()
				break
			}
			fm.mutex.Unlock()

			line := scanner.Text()
			event := fm.parseAndFilter(line)
			if event != nil {
				fm.mutex.Lock()
				fm.events = append(fm.events, *event)

				// 写入文件 (JSON Lines 格式)
				if fm.logFile != nil {
					data, _ := json.Marshal(event)
					fm.logFile.Write(data)
					fm.logFile.WriteString("\n")
				}

				// 检查是否是关键事件
				if fm.isKeyEvent(event) {
					fm.keyEvents = append(fm.keyEvents, fm.toKeyEvent(event))
				}

				fm.mutex.Unlock()

				// 打印日志
				log.Printf("📄 [%s] %s %s -> %s",
					event.EventType, event.AppName, event.FileName, event.FilePath)
			}
		}

		fm.cmd.Wait()
	}()

	return nil
}

// Stop 停止监控
func (fm *FileMonitor) Stop() error {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	if !fm.running || fm.cmd == nil {
		return fmt.Errorf("没有正在进行的监控")
	}

	// 停止 fs_usage
	if fm.cmd.Process != nil {
		fm.cmd.Process.Signal(syscall.SIGTERM)
		time.Sleep(100 * time.Millisecond)
		fm.cmd.Process.Kill()
	}

	// 关闭日志文件
	if fm.logFile != nil {
		fm.logFile.Close()
		fm.logFile = nil
	}

	fm.running = false
	log.Printf("📂 文件监控已停止, 共 %d 条记录, %d 个关键事件", len(fm.events), len(fm.keyEvents))

	return nil
}

// GetKeyEvents 获取关键事件
func (fm *FileMonitor) GetKeyEvents() []KeyEvent {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	result := make([]KeyEvent, len(fm.keyEvents))
	copy(result, fm.keyEvents)
	return result
}

// GetEvents 获取所有事件
func (fm *FileMonitor) GetEvents() []LogEntry {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	result := make([]LogEntry, len(fm.events))
	copy(result, fm.events)
	return result
}

// parseAndFilter 解析并过滤 fs_usage 输出
func (fm *FileMonitor) parseAndFilter(line string) *LogEntry {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}

	fields := strings.Fields(line)
	if len(fields) < 4 {
		return nil
	}

	rawProcess := fields[len(fields)-1]

	// 过滤自身
	if strings.Contains(line, "."+fm.myPid) {
		return nil
	}

	processName := rawProcess
	processID := ""
	if idx := strings.LastIndex(processName, "."); idx != -1 {
		processID = processName[idx+1:]
		processName = processName[:idx]
	}

	// 过滤系统噪音进程（纯后台服务）
	// 注意：保留 filecoordinationd 和 quicklookd 用于后续关联分析
	systemProcesses := []string{
		// 原有核心系统进程
		"mdworker", "mds", "git", "notifyd", "bird", "com.apple",
		"kernel_task", "ffmpeg", "fs_usage", "monitor_server",
		"spotlight", "mds_stores", "coreaudiod", "bluetoothd", "WindowServer", "touchbar",
		// 新增：macOS 系统守护进程（高噪音）
		"cfprefsd",              // 配置服务
		"BiomeAgent",            // 生物特征代理
		"analyticsd",            // 分析服务
		"cloudd",                // iCloud 同步服务
		"nsurlsessiond",         // 网络会话服务（系统级）
		"trustd",                // 证书信任服务
		"symptomsd",             // 系统症状服务
		"logd",                  // 日志服务
		"syslogd",               // 系统日志
		"configd",               // 配置守护进程
		"diskarbitrationd",      // 磁盘仲裁
		"coreduetd",             // 核心预测
		"contextstored",         // 上下文存储
		"powerd",                // 电源管理
		"timed",                 // 时间服务
		"locationd",             // 位置服务
		"tccd",                  // 透明度同意控制
		"sharingd",              // 系统级共享（非用户共享）
		"rapportd",              // 设备连接
		"suggestd",              // Siri 建议
		"remindd",               // 提醒事项后台
		"CalendarAgent",         // 日历代理
		"AddressBookSourceSync", // 通讯录同步
	}
	for _, sp := range systemProcesses {
		if processName == sp || strings.Contains(processName, sp) {
			return nil
		}
	}

	rawOp := fields[1]
	cleanOp := strings.Split(strings.Split(strings.ToLower(rawOp), "(")[0], "[")[0]

	eventType, exists := fileOperationMap[cleanOp]
	if !exists {
		return nil
	}

	// 路径解析
	pathFields := []string{}
	for i := 2; i < len(fields)-1; i++ {
		field := fields[i]
		if strings.Contains(field, ".") && len(field) < 10 {
			if _, err := strconv.ParseFloat(field, 64); err == nil {
				continue
			}
		}
		if strings.HasPrefix(field, "F=") || strings.HasPrefix(field, "fd=") ||
			strings.HasPrefix(field, "B=") || field == "|" {
			continue
		}
		pathFields = append(pathFields, field)
	}

	if len(pathFields) == 0 {
		return nil
	}

	fullPathStr := strings.Join(pathFields, " ")

	if !strings.HasPrefix(fullPathStr, "/") {
		slashIdx := strings.Index(fullPathStr, "/")
		if slashIdx == -1 {
			return nil
		}
		fullPathStr = fullPathStr[slashIdx:]
	}

	// 过滤系统路径
	for _, prefix := range fileIgnorePrefixes {
		if strings.HasPrefix(fullPathStr, prefix) {
			return nil
		}
	}

	if strings.Contains(fullPathStr, "/Library/") {
		return nil
	}

	// 忽略隐藏文件
	baseName := filepath.Base(fullPathStr)
	if strings.HasPrefix(baseName, ".") {
		return nil
	}
	if strings.Contains(fullPathStr, "/.") {
		return nil
	}

	// 忽略临时文件
	ext := strings.ToLower(filepath.Ext(fullPathStr))
	tempExts := []string{".tmp", ".lock", ".dat", ".plist", ".db", ".log", ".crdownload", ".download"}
	for _, te := range tempExts {
		if ext == te {
			return nil
		}
	}

	if strings.Contains(baseName, "~tmp") || strings.HasPrefix(baseName, "~$") {
		return nil
	}

	// 获取完整的文件信息
	var fileSize int64 = 0
	if info, err := os.Stat(fullPathStr); err == nil {
		fileSize = info.Size()
	}

	// 获取当前活动窗口信息
	var windowTitle string
	var activeApp string
	if fm.windowMonitor != nil {
		activeApp, windowTitle = fm.windowMonitor.getActiveWindow()
	}

	// 规范化应用名称和获取风险信息
	baseProcessName := processName
	if activeApp != "" {
		baseProcessName = activeApp
	}
	appName, category, _ := fm.normalizeProcessName(baseProcessName, windowTitle)

	// Check for sensitive/upload
	var isSensitive bool
	// Re-implement or reuse isKeyEvent logic check
	// But actually we have checkSensitiveFile in log_capturer!
	// I should probably have checkSensitiveFile as helper in file_monitor too or util.
	// For now, inline check:
	fileNameLower := strings.ToLower(baseName)
	for _, keyword := range SensitiveFileKeywords {
		if strings.Contains(fileNameLower, strings.ToLower(keyword)) {
			isSensitive = true
			break
		}
	}

	var uploadInfo *UploadDetection
	if isSensitive {
		uploadInfo = &UploadDetection{
			IsUpload:     true,
			AppName:      appName,
			UploadType:   "Sensitive Access",
			OriginalFile: fullPathStr,
		}
	}

	// 构建事件
	event := &LogEntry{
		Timestamp:     time.Now().Format("2006-01-02T15:04:05.000"),
		EventType:     eventType,
		FilePath:      fullPathStr,
		FileName:      baseName,
		FileSize:      fileSize,
		FileExtension: ext,
		ProcessInfo: ProcessInfo{
			PID:         processID,
			ProcessName: processName,
		},
		WindowInfo: WindowInfo{
			WindowTitle: windowTitle,
		},
		UserInfo: UserInfo{
			Username: fm.currentUser,
			Hostname: fm.hostname,
		},
		DiskInfo: DiskInfo{
			DriveLetter: "/",
			DiskType:    "SSD/HDD",
		},
		AppName:    appName,
		UploadInfo: uploadInfo,
		Extra: map[string]interface{}{
			"raw_operation": cleanOp,
			"category":      category,
		},
	}

	return event
}

// normalizeProcessName 规范化进程名称并识别应用
func (fm *FileMonitor) normalizeProcessName(processName, windowTitle string) (string, string, string) {
	// 首先检查黑名单应用
	if risk, exists := BlacklistApps[processName]; exists {
		return risk.Name, risk.Category, "高"
	}

	// 检查窗口标题中是否包含黑名单应用信息
	windowTitleLower := strings.ToLower(windowTitle)
	for name, risk := range BlacklistApps {
		if strings.Contains(windowTitleLower, strings.ToLower(name)) ||
			strings.Contains(windowTitleLower, strings.ToLower(risk.Name)) {
			return risk.Name, risk.Category, "高"
		}
	}

	// 检查窗口标题中是否包含黑名单网站
	for domain, risk := range BlacklistWebsites {
		if strings.Contains(windowTitleLower, domain) ||
			strings.Contains(windowTitleLower, strings.ToLower(risk.Name)) {
			return risk.Name, risk.Category, "高"
		}
	}

	// 常见应用名称映射
	appNameMap := map[string]string{
		"Google Chrome": "Chrome",
		"Safari":        "Safari",
		"Firefox":       "Firefox",
		"Finder":        "Finder",
		"Preview":       "预览",
		"TextEdit":      "文本编辑",
		"Terminal":      "终端",
		"Code":          "VS Code",
		"Arc":           "Arc",
	}

	if appName, exists := appNameMap[processName]; exists {
		return appName, "", ""
	}

	return processName, "", ""
}

// isKeyEvent 判断是否是关键事件
func (fm *FileMonitor) isKeyEvent(event *LogEntry) bool {
	// 1. 敏感文件操作
	fileName := strings.ToLower(event.FileName)
	for _, keyword := range SensitiveFileKeywords {
		if strings.Contains(fileName, strings.ToLower(keyword)) {
			return true
		}
	}

	// 2. 敏感文件扩展名
	if event.FileExtension != "" {
		ext := strings.ToLower(event.FileExtension)
		for _, sensitiveExt := range SensitiveFileExtensions {
			if ext == sensitiveExt {
				return true
			}
		}
	}

	// 3. 风险级别 (Extra check if needed)
	if event.UploadInfo != nil && event.UploadInfo.IsUpload {
		return true
	}

	// 4. 特定目录下的操作
	if strings.Contains(event.FilePath, "/Documents") ||
		strings.Contains(event.FilePath, "/Desktop") {
		// 仅保留非临时文件的关键操作
		if event.EventType == "created" || event.EventType == "deleted" {
			return true
		}
	}

	return false
}

// toKeyEvent 转换为关键事件
func (fm *FileMonitor) toKeyEvent(event *LogEntry) KeyEvent {
	return KeyEvent(*event)
}

// AddFileSelectedEvent 添加文件选择事件 (用于模拟文件对话框检测)
func (fm *FileMonitor) AddFileSelectedEvent(filePath, appName, windowTitle string) {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	baseName := filepath.Base(filePath)

	event := LogEntry{
		Timestamp:     time.Now().Format("2006-01-02T15:04:05.000"),
		EventType:     "file_selected",
		FilePath:      filePath,
		FileName:      baseName,
		FileExtension: filepath.Ext(filePath),
		AppName:       appName,
		WindowInfo: WindowInfo{
			WindowTitle: windowTitle,
		},
		UploadInfo: &UploadDetection{
			IsUpload:     true,
			AppName:      appName,
			UploadType:   "File Dialog Selection",
			OriginalFile: filePath,
		},
		Extra: map[string]interface{}{
			"detection_method": "file_dialog",
		},
		UserInfo: UserInfo{
			Username: fm.currentUser,
			Hostname: fm.hostname,
		},
	}

	fm.events = append(fm.events, event)

	// 文件选择事件始终是关键事件
	fm.keyEvents = append(fm.keyEvents, fm.toKeyEvent(&event))

	// 写入日志文件
	if fm.logFile != nil {
		data, _ := json.Marshal(event)
		fm.logFile.Write(data)
		fm.logFile.WriteString("\n")
	}

	log.Printf("📎 文件选择: %s 选择了 %s", appName, baseName)
}

// GetRiskEvents 获取风险事件数量
func (fm *FileMonitor) GetRiskEvents() int {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()
	return len(fm.keyEvents)
}
