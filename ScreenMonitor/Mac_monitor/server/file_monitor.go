// server/file_monitor.go
// 增强版文件监控器 - 整合应用-文件关联
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net"
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

// FileMonitor 增强版文件监控器 - FSEvents IPC 版本
// 使用独立的 fsevents_client 进程获取精准时间戳，使用 fs_usage 获取进程信息
type FileMonitor struct {
	recordsDir    string
	cmd           *exec.Cmd // fs_usage 进程
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
	windowMonitor *WindowMonitor

	// FSEvents IPC 架构字段
	fseventsClient     *exec.Cmd                    // fsevents_client 子进程
	socketListener     net.Listener                 // Unix socket 监听器
	socketPath         string                       // socket 路径
	pendingEvents      map[string]*PendingEvent     // 待合并事件缓存
	fsUsageProcessInfo map[string]*ProcessInfoCache // fs_usage 进程信息缓存
	stopChan           chan struct{}                // 停止信号
}

// FSEventIPC 从 fsevents_client 接收的事件
type FSEventIPC struct {
	Timestamp string `json:"timestamp"`
	EventType string `json:"event_type"`
	Path      string `json:"path"`
}

// PendingEvent 待合并的事件
type PendingEvent struct {
	FSEvent     FSEventIPC   // 来自 FSEvents 的原始事件（含精准时间戳）
	ProcessInfo *ProcessInfo // 来自 fs_usage 的进程信息（可能为空）
	CreateTime  time.Time    // 事件进入缓存的时间
	Emitted     bool         // 是否已发送
}

// ProcessInfoCache fs_usage 进程信息缓存
type ProcessInfoCache struct {
	ProcessName string
	ProcessID   string
	EventType   string
	CaptureTime time.Time
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
		recordsDir:         recordsDir,
		myPid:              strconv.Itoa(os.Getpid()),
		homeDir:            homeDir,
		currentUser:        username,
		hostname:           hostname,
		windowMonitor:      windowMonitor,
		socketPath:         "/tmp/fsevents_monitor.sock",
		pendingEvents:      make(map[string]*PendingEvent),
		fsUsageProcessInfo: make(map[string]*ProcessInfoCache),
		stopChan:           make(chan struct{}),
	}
}

// IsRunning 检查是否正在运行
func (fm *FileMonitor) IsRunning() bool {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()
	return fm.running
}

// Start 开始监控（FSEvents IPC 版本）
func (fm *FileMonitor) Start(outputPath string) error {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	if fm.running {
		return fmt.Errorf("已经在运行中")
	}

	// 检查是否有 sudo 权限
	if os.Geteuid() != 0 {
		log.Println("⚠️ 警告: fs_usage 需要 sudo 权限，进程关联功能将受限")
		log.Println("💡 FSEvents 时间戳仍然准确")
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
	fm.stopChan = make(chan struct{})

	// 1. 删除旧的 socket 文件
	os.Remove(fm.socketPath)

	// 2. 创建 Unix Socket 监听器
	fm.socketListener, err = net.Listen("unix", fm.socketPath)
	if err != nil {
		fm.logFile.Close()
		return fmt.Errorf("创建 socket 监听器失败: %v", err)
	}

	// 3. 启动 fsevents_client 子进程
	clientPath := filepath.Join(filepath.Dir(os.Args[0]), "..", "fsevents_client", "fsevents_client")
	if _, err := os.Stat(clientPath); os.IsNotExist(err) {
		// 尝试当前目录
		clientPath = "./fsevents_client/fsevents_client"
	}

	fm.fseventsClient = exec.Command(clientPath, fm.socketPath)
	fm.fseventsClient.Stdout = os.Stdout
	fm.fseventsClient.Stderr = os.Stderr

	if err := fm.fseventsClient.Start(); err != nil {
		log.Printf("⚠️ 启动 fsevents_client 失败: %v", err)
		// 继续运行，使用纯 fs_usage 模式
	} else {
		log.Printf("🎯 FSEvents IPC 已启动 (socket: %s)", fm.socketPath)
		// 启动 socket 接收协程
		go fm.handleSocketConnections()
	}

	// 4. 启动 fs_usage（提供进程信息）
	fm.cmd = exec.Command("fs_usage", "-f", "filesys", "-w")
	stdout, err := fm.cmd.StdoutPipe()
	if err != nil {
		log.Printf("⚠️ 创建 fs_usage stdout pipe 失败: %v", err)
	}

	if fm.cmd != nil && stdout != nil {
		if err := fm.cmd.Start(); err != nil {
			log.Printf("⚠️ 启动 fs_usage 失败: %v (需要 sudo 权限)", err)
			fm.cmd = nil
		} else {
			// 启动 fs_usage 读取协程
			go fm.fsUsageReaderLoop(stdout)
		}
	}

	// 5. 启动事件协调器
	go fm.eventCoordinatorLoop()

	fm.running = true
	log.Printf("📂 开始混合文件监控: %s", outputPath)

	return nil
}

// handleSocketConnections 处理 socket 连接
func (fm *FileMonitor) handleSocketConnections() {
	for {
		select {
		case <-fm.stopChan:
			return
		default:
		}

		conn, err := fm.socketListener.Accept()
		if err != nil {
			select {
			case <-fm.stopChan:
				return
			default:
				log.Printf("⚠️ Socket accept 错误: %v", err)
				continue
			}
		}

		go fm.handleClientConnection(conn)
	}
}

// handleClientConnection 处理单个客户端连接
func (fm *FileMonitor) handleClientConnection(conn net.Conn) {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var event FSEventIPC
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			log.Printf("⚠️ 解析 FSEvent JSON 失败: %v", err)
			continue
		}

		// 处理事件
		fm.handleFSEventIPC(event)
	}
}

// handleFSEventIPC 处理来自 fsevents_client 的事件
// 立即发送事件，确保时间戳精确
func (fm *FileMonitor) handleFSEventIPC(event FSEventIPC) {
	key := event.EventType + ":" + event.Path

	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	// 尝试获取进程信息（但不等待）
	var processInfo *ProcessInfo
	if cached, exists := fm.fsUsageProcessInfo[key]; exists {
		if time.Since(cached.CaptureTime) < 3*time.Second {
			processInfo = &ProcessInfo{
				PID:         cached.ProcessID,
				ProcessName: cached.ProcessName,
			}
			delete(fm.fsUsageProcessInfo, key)
		}
	}

	// 立即发送事件，确保时间戳精确（不等待进程信息）
	fm.emitEventIPC(event, processInfo)
}

// eventCoordinatorLoop 事件协调器 - 定期刷新待处理事件
func (fm *FileMonitor) eventCoordinatorLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-fm.stopChan:
			fm.flushAllPendingEventsIPC()
			return
		case <-ticker.C:
			fm.flushExpiredPendingEventsIPC()
		}
	}
}

// fsUsageReaderLoop 读取 fs_usage 输出的循环
func (fm *FileMonitor) fsUsageReaderLoop(stdout interface{}) {
	reader, ok := stdout.(interface{ Read([]byte) (int, error) })
	if !ok {
		return
	}

	scanner := bufio.NewScanner(reader)
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
		fm.processFsUsageLine(line)
	}

	if fm.cmd != nil {
		fm.cmd.Wait()
	}
}

// processFsUsageLine 处理 fs_usage 的一行输出
func (fm *FileMonitor) processFsUsageLine(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}

	fields := strings.Fields(line)
	if len(fields) < 4 {
		return
	}

	rawProcess := fields[len(fields)-1]

	// 过滤自身
	if strings.Contains(line, "."+fm.myPid) {
		return
	}

	processName := rawProcess
	processID := ""
	if idx := strings.LastIndex(processName, "."); idx != -1 {
		processID = processName[idx+1:]
		processName = processName[:idx]
	}

	// 过滤系统进程
	if fm.isSystemProcess(processName) {
		return
	}

	rawOp := fields[1]
	cleanOp := strings.Split(strings.Split(strings.ToLower(rawOp), "(")[0], "[")[0]

	eventType, exists := fileOperationMap[cleanOp]
	if !exists {
		return
	}

	// 解析路径
	fullPathStr := fm.extractPathFromFields(fields)
	if fullPathStr == "" {
		return
	}

	// 过滤路径
	if fm.shouldIgnorePath(fullPathStr) {
		return
	}

	key := eventType + ":" + fullPathStr

	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	// 检查是否有对应的待处理 FSEvent
	if pending, exists := fm.pendingEvents[key]; exists && !pending.Emitted {
		// 找到匹配，补充进程信息并输出
		processInfo := &ProcessInfo{
			PID:         processID,
			ProcessName: processName,
		}
		fm.emitEventIPC(pending.FSEvent, processInfo)
		pending.Emitted = true
		delete(fm.pendingEvents, key)
	} else {
		// 缓存进程信息，等待 FSEvent
		fm.fsUsageProcessInfo[key] = &ProcessInfoCache{
			ProcessName: processName,
			ProcessID:   processID,
			EventType:   eventType,
			CaptureTime: time.Now(),
		}
	}
}

// isSystemProcess 检查是否是系统进程
func (fm *FileMonitor) isSystemProcess(processName string) bool {
	systemProcesses := []string{
		"mdworker", "mds", "git", "notifyd", "bird", "com.apple",
		"kernel_task", "ffmpeg", "fs_usage", "monitor_server",
		"spotlight", "mds_stores", "coreaudiod", "bluetoothd", "WindowServer", "touchbar",
		"cfprefsd", "BiomeAgent", "analyticsd", "cloudd", "nsurlsessiond",
		"trustd", "symptomsd", "logd", "syslogd", "configd",
		"diskarbitrationd", "coreduetd", "contextstored", "powerd", "timed",
		"locationd", "tccd", "sharingd", "rapportd", "suggestd",
		"remindd", "CalendarAgent", "AddressBookSourceSync",
	}
	for _, sp := range systemProcesses {
		if processName == sp || strings.Contains(processName, sp) {
			return true
		}
	}
	return false
}

// extractPathFromFields 从 fs_usage 字段中提取路径
func (fm *FileMonitor) extractPathFromFields(fields []string) string {
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
		return ""
	}

	fullPathStr := strings.Join(pathFields, " ")

	if !strings.HasPrefix(fullPathStr, "/") {
		slashIdx := strings.Index(fullPathStr, "/")
		if slashIdx == -1 {
			return ""
		}
		fullPathStr = fullPathStr[slashIdx:]
	}

	return fullPathStr
}

// shouldIgnorePath 检查是否应该忽略路径
func (fm *FileMonitor) shouldIgnorePath(fullPathStr string) bool {
	for _, prefix := range fileIgnorePrefixes {
		if strings.HasPrefix(fullPathStr, prefix) {
			return true
		}
	}

	if strings.Contains(fullPathStr, "/Library/") {
		return true
	}

	baseName := filepath.Base(fullPathStr)
	if strings.HasPrefix(baseName, ".") {
		return true
	}
	if strings.Contains(fullPathStr, "/.") {
		return true
	}

	ext := strings.ToLower(filepath.Ext(fullPathStr))
	tempExts := []string{".tmp", ".lock", ".dat", ".plist", ".db", ".log", ".crdownload", ".download"}
	for _, te := range tempExts {
		if ext == te {
			return true
		}
	}

	if strings.Contains(baseName, "~tmp") || strings.HasPrefix(baseName, "~$") {
		return true
	}

	return false
}

// Stop 停止监控（FSEvents IPC 版本）
func (fm *FileMonitor) Stop() error {
	fm.mutex.Lock()

	if !fm.running {
		fm.mutex.Unlock()
		return fmt.Errorf("没有正在进行的监控")
	}

	fm.running = false
	fm.mutex.Unlock()

	// 1. 发送停止信号
	if fm.stopChan != nil {
		close(fm.stopChan)
	}

	// 2. 关闭 socket 监听器
	if fm.socketListener != nil {
		fm.socketListener.Close()
	}

	// 3. 停止 fsevents_client 子进程
	if fm.fseventsClient != nil && fm.fseventsClient.Process != nil {
		fm.fseventsClient.Process.Signal(syscall.SIGTERM)
		time.Sleep(100 * time.Millisecond)
		fm.fseventsClient.Process.Kill()
	}

	// 4. 删除 socket 文件
	if fm.socketPath != "" {
		os.Remove(fm.socketPath)
	}

	// 5. 停止 fs_usage
	if fm.cmd != nil && fm.cmd.Process != nil {
		fm.cmd.Process.Signal(syscall.SIGTERM)
		time.Sleep(100 * time.Millisecond)
		fm.cmd.Process.Kill()
	}

	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	// 6. 关闭日志文件
	if fm.logFile != nil {
		fm.logFile.Close()
		fm.logFile = nil
	}

	log.Printf("📂 混合文件监控已停止, 共 %d 条记录, %d 个关键事件", len(fm.events), len(fm.keyEvents))

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

// AddUploadDetectedEvent 添加上传检测事件
// 当通过启发式分析检测到文件正在被上传时调用
func (fm *FileMonitor) AddUploadDetectedEvent(filePath, appName, uploadType, windowTitle string) {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	baseName := filepath.Base(filePath)
	var fileSize int64 = 0
	if info, err := os.Stat(filePath); err == nil {
		fileSize = info.Size()
	}

	event := LogEntry{
		Timestamp:     time.Now().Format("2006-01-02T15:04:05.000"),
		EventType:     "upload_detected",
		FilePath:      filePath,
		FileName:      baseName,
		FileSize:      fileSize,
		FileExtension: filepath.Ext(filePath),
		AppName:       appName,
		WindowInfo: WindowInfo{
			WindowTitle: windowTitle,
		},
		UploadInfo: &UploadDetection{
			IsUpload:     true,
			AppName:      appName,
			UploadType:   uploadType,
			OriginalFile: filePath,
		},
		Extra: map[string]interface{}{
			"detection_method": "upload_detection",
		},
		UserInfo: UserInfo{
			Username: fm.currentUser,
			Hostname: fm.hostname,
		},
	}

	fm.events = append(fm.events, event)

	// 上传检测事件始终是关键事件
	fm.keyEvents = append(fm.keyEvents, fm.toKeyEvent(&event))

	// 写入日志文件
	if fm.logFile != nil {
		data, _ := json.Marshal(event)
		fm.logFile.Write(data)
		fm.logFile.WriteString("\n")
	}

	log.Printf("⬆️ 上传检测: %s 正在上传 %s (%s)", appName, baseName, uploadType)
}

// emitEventIPC 输出来自 FSEvents IPC 的事件
func (fm *FileMonitor) emitEventIPC(fsEvent FSEventIPC, processInfo *ProcessInfo) {
	baseName := filepath.Base(fsEvent.Path)
	ext := strings.ToLower(filepath.Ext(fsEvent.Path))

	var fileSize int64 = 0
	if info, err := os.Stat(fsEvent.Path); err == nil {
		fileSize = info.Size()
	}

	// 获取当前活动窗口信息
	var windowTitle string
	var activeApp string
	if fm.windowMonitor != nil {
		activeApp, windowTitle = fm.windowMonitor.getActiveWindow()
	}

	// 确定应用名称
	appName := ""
	if processInfo != nil && processInfo.ProcessName != "" {
		appName = processInfo.ProcessName
	} else if activeApp != "" {
		appName = activeApp
	}

	// 规范化应用名称
	appName, category, _ := fm.normalizeProcessName(appName, windowTitle)

	// 检查敏感文件
	var isSensitive bool
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
			OriginalFile: fsEvent.Path,
		}
	}

	// 使用 FSEvents 的精准时间戳
	event := &LogEntry{
		Timestamp:     fsEvent.Timestamp, // 已经是格式化的字符串
		EventType:     fsEvent.EventType,
		FilePath:      fsEvent.Path,
		FileName:      baseName,
		FileSize:      fileSize,
		FileExtension: ext,
		ProcessInfo: ProcessInfo{
			PID:         "",
			ProcessName: "",
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
			"raw_operation": fsEvent.EventType,
			"category":      category,
			"source":        "fsevents_ipc", // 标记使用 FSEvents IPC 时间戳
		},
	}

	// 填充进程信息
	if processInfo != nil {
		event.ProcessInfo = *processInfo
	}

	// 添加到事件列表
	fm.events = append(fm.events, *event)

	// 写入文件
	if fm.logFile != nil {
		data, _ := json.Marshal(event)
		fm.logFile.Write(data)
		fm.logFile.WriteString("\n")
	}

	// 检查关键事件
	if fm.isKeyEvent(event) {
		fm.keyEvents = append(fm.keyEvents, fm.toKeyEvent(event))
	}

	// 打印日志
	log.Printf("📄 [%s] %s %s -> %s (精准时间戳)",
		event.EventType, event.AppName, event.FileName, event.FilePath)
}

// flushExpiredPendingEventsIPC 刷新超时的待处理 IPC 事件
func (fm *FileMonitor) flushExpiredPendingEventsIPC() {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	now := time.Now()
	var toDelete []string

	for key, pending := range fm.pendingEvents {
		if pending.Emitted {
			toDelete = append(toDelete, key)
			continue
		}

		// 3秒超时
		if now.Sub(pending.CreateTime) > 3*time.Second {
			// 没有进程信息也输出
			fm.emitEventIPC(pending.FSEvent, nil)
			pending.Emitted = true
			toDelete = append(toDelete, key)
		}
	}

	for _, key := range toDelete {
		delete(fm.pendingEvents, key)
	}

	// 清理过期的 fs_usage 进程信息缓存
	for key, cache := range fm.fsUsageProcessInfo {
		if now.Sub(cache.CaptureTime) > 5*time.Second {
			delete(fm.fsUsageProcessInfo, key)
		}
	}
}

// flushAllPendingEventsIPC 刷新所有待处理 IPC 事件
func (fm *FileMonitor) flushAllPendingEventsIPC() {
	fm.mutex.Lock()
	defer fm.mutex.Unlock()

	for key, pending := range fm.pendingEvents {
		if !pending.Emitted {
			fm.emitEventIPC(pending.FSEvent, nil)
		}
		delete(fm.pendingEvents, key)
	}
}
