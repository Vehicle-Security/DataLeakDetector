// server/window_monitor.go
// 活动窗口监控器 - 监控当前活动的应用程序和窗口标题
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// WindowEvent 窗口事件
type WindowEvent struct {
	Timestamp   float64 `json:"timestamp"`      // 相对于录制开始的时间（秒）
	AppName     string  `json:"app_name"`       // 应用名称（规范化后）
	RawAppName  string  `json:"raw_app_name"`   // 原始应用名称
	WindowTitle string  `json:"window_title"`   // 窗口标题
	Category    string  `json:"category"`       // 风险类别
	RiskLevel   string  `json:"risk_level"`     // 风险等级
	EventType   string  `json:"event_type"`     // 事件类型
}

// WindowMonitor 窗口监控器
type WindowMonitor struct {
	running      bool
	mutex        sync.Mutex
	events       []WindowEvent
	logFile      *os.File
	startTime    time.Time
	lastApp      string
	lastTitle    string
	pollInterval time.Duration
}

// NewWindowMonitor 创建窗口监控器
func NewWindowMonitor() *WindowMonitor {
	return &WindowMonitor{
		pollInterval: 500 * time.Millisecond, // 每500ms检查一次
	}
}

// IsRunning 检查是否正在运行
func (w *WindowMonitor) IsRunning() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	return w.running
}

// Start 开始监控
func (w *WindowMonitor) Start(outputPath string) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.running {
		return fmt.Errorf("窗口监控已在运行中")
	}

	// 创建输出文件
	var err error
	w.logFile, err = os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("创建窗口日志文件失败: %v", err)
	}

	w.events = []WindowEvent{}
	w.startTime = time.Now()
	w.lastApp = ""
	w.lastTitle = ""
	w.running = true

	// 写入 JSON 数组开始
	w.logFile.WriteString("[\n")

	log.Printf("🪟 开始窗口监控: %s", outputPath)

	// 启动监控协程
	go w.monitorLoop()

	return nil
}

// Stop 停止监控
func (w *WindowMonitor) Stop() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if !w.running {
		return fmt.Errorf("窗口监控未在运行")
	}

	w.running = false

	// 关闭日志文件
	if w.logFile != nil {
		w.logFile.WriteString("\n]")
		w.logFile.Close()
		w.logFile = nil
	}

	log.Printf("🪟 窗口监控已停止, 共 %d 个应用切换事件", len(w.events))
	return nil
}

// GetEvents 获取所有事件
func (w *WindowMonitor) GetEvents() []WindowEvent {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	result := make([]WindowEvent, len(w.events))
	copy(result, w.events)
	return result
}

// monitorLoop 监控循环
func (w *WindowMonitor) monitorLoop() {
	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	firstEntry := true

	for {
		w.mutex.Lock()
		if !w.running {
			w.mutex.Unlock()
			break
		}
		w.mutex.Unlock()

		// 获取当前活动窗口信息
		appName, windowTitle := w.getActiveWindow()
		
		if appName != "" && (appName != w.lastApp || windowTitle != w.lastTitle) {
			// 应用或窗口标题发生变化
			event := w.createEvent(appName, windowTitle)
			
			if event != nil {
				w.mutex.Lock()
				w.events = append(w.events, *event)
				
				// 写入文件
				if w.logFile != nil {
					data, _ := json.Marshal(event)
					if !firstEntry {
						w.logFile.WriteString(",\n")
					}
					w.logFile.Write(data)
					firstEntry = false
				}
				w.mutex.Unlock()

				// 打印日志
				if event.RiskLevel != "" {
					log.Printf("🚨 [%s] %s - %s (%s)", 
						event.RiskLevel, event.AppName, event.WindowTitle, event.Category)
				} else {
					log.Printf("🪟 应用切换: %s - %s", event.AppName, event.WindowTitle)
				}
			}
			
			w.lastApp = appName
			w.lastTitle = windowTitle
		}

		<-ticker.C
	}
}

// getActiveWindow 获取当前活动窗口信息
func (w *WindowMonitor) getActiveWindow() (appName, windowTitle string) {
	// 使用 AppleScript 获取当前活动应用和窗口标题
	script := `
	tell application "System Events"
		set frontApp to first application process whose frontmost is true
		set appName to name of frontApp
		try
			set windowTitle to name of front window of frontApp
		on error
			set windowTitle to ""
		end try
		return appName & "|||" & windowTitle
	end tell
	`
	
	cmd := exec.Command("osascript", "-e", script)
	output, err := cmd.Output()
	if err != nil {
		return "", ""
	}

	result := strings.TrimSpace(string(output))
	parts := strings.Split(result, "|||")
	if len(parts) >= 2 {
		return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
	} else if len(parts) == 1 {
		return strings.TrimSpace(parts[0]), ""
	}
	return "", ""
}

// createEvent 创建窗口事件
func (w *WindowMonitor) createEvent(appName, windowTitle string) *WindowEvent {
	timestamp := time.Since(w.startTime).Seconds()

	event := &WindowEvent{
		Timestamp:   timestamp,
		RawAppName:  appName,
		WindowTitle: windowTitle,
		EventType:   RiskTypeAppSwitch,
	}

	// 检查是否是黑名单应用
	if risk, exists := BlacklistApps[appName]; exists {
		event.AppName = risk.Name
		event.Category = risk.Category
		event.RiskLevel = "高"
	} else {
		event.AppName = normalizeAppName(appName)
		
		// 检查窗口标题中是否包含敏感网站
		for domain, risk := range BlacklistWebsites {
			if strings.Contains(strings.ToLower(windowTitle), domain) {
				event.AppName = risk.Name
				event.Category = risk.Category
				event.RiskLevel = "高"
				event.EventType = RiskTypeWebsiteVisit
				break
			}
		}
	}

	// 检查浏览器窗口标题中的网站
	if isBrowser(appName) {
		for domain, risk := range BlacklistWebsites {
			if strings.Contains(strings.ToLower(windowTitle), domain) ||
				strings.Contains(strings.ToLower(windowTitle), risk.Name) {
				event.AppName = risk.Name
				event.Category = risk.Category
				event.RiskLevel = "高"
				event.EventType = RiskTypeWebsiteVisit
				break
			}
		}
	}

	return event
}

// normalizeAppName 规范化应用名称
func normalizeAppName(appName string) string {
	// 常见应用名称映射
	nameMap := map[string]string{
		"Google Chrome":      "Chrome",
		"Safari":             "Safari",
		"Firefox":            "Firefox",
		"Microsoft Edge":     "Edge",
		"Arc":                "Arc",
		"Finder":             "Finder",
		"Preview":            "预览",
		"TextEdit":           "文本编辑",
		"Notes":              "备忘录",
		"Terminal":           "终端",
		"iTerm2":             "iTerm2",
		"Visual Studio Code": "VS Code",
		"Code":               "VS Code",
		"Xcode":              "Xcode",
	}

	if normalized, exists := nameMap[appName]; exists {
		return normalized
	}
	return appName
}

// isBrowser 检查是否是浏览器
func isBrowser(appName string) bool {
	browsers := []string{
		"Google Chrome", "Chrome", "Safari", "Firefox",
		"Microsoft Edge", "Edge", "Arc", "Opera", "Brave",
	}
	for _, browser := range browsers {
		if strings.EqualFold(appName, browser) {
			return true
		}
	}
	return false
}

// ClipboardMonitor 剪贴板监控器
type ClipboardMonitor struct {
	running      bool
	mutex        sync.Mutex
	events       []ClipboardEvent
	logFile      *os.File
	startTime    time.Time
	lastContent  string
	pollInterval time.Duration
}

// ClipboardEvent 剪贴板事件
type ClipboardEvent struct {
	Timestamp    float64 `json:"timestamp"`
	ContentType  string  `json:"content_type"`  // text, file, image
	ContentSize  int     `json:"content_size"`  // 内容大小
	Preview      string  `json:"preview"`       // 内容预览（前100字符）
	SourceApp    string  `json:"source_app"`    // 来源应用
	IsSensitive  bool    `json:"is_sensitive"`  // 是否敏感
	SensitiveKey string  `json:"sensitive_key"` // 匹配的敏感关键词
}

// NewClipboardMonitor 创建剪贴板监控器
func NewClipboardMonitor() *ClipboardMonitor {
	return &ClipboardMonitor{
		pollInterval: 1 * time.Second,
	}
}

// Start 开始监控
func (c *ClipboardMonitor) Start(outputPath string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.running {
		return fmt.Errorf("剪贴板监控已在运行中")
	}

	var err error
	c.logFile, err = os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("创建剪贴板日志文件失败: %v", err)
	}

	c.events = []ClipboardEvent{}
	c.startTime = time.Now()
	c.lastContent = ""
	c.running = true

	c.logFile.WriteString("[\n")

	log.Printf("📋 开始剪贴板监控: %s", outputPath)

	go c.monitorLoop()

	return nil
}

// Stop 停止监控
func (c *ClipboardMonitor) Stop() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if !c.running {
		return fmt.Errorf("剪贴板监控未在运行")
	}

	c.running = false

	if c.logFile != nil {
		c.logFile.WriteString("\n]")
		c.logFile.Close()
		c.logFile = nil
	}

	log.Printf("📋 剪贴板监控已停止, 共 %d 个事件", len(c.events))
	return nil
}

// monitorLoop 监控循环
func (c *ClipboardMonitor) monitorLoop() {
	ticker := time.NewTicker(c.pollInterval)
	defer ticker.Stop()

	firstEntry := true

	for {
		c.mutex.Lock()
		if !c.running {
			c.mutex.Unlock()
			break
		}
		c.mutex.Unlock()

		content := c.getClipboardContent()
		
		if content != "" && content != c.lastContent {
			event := c.createEvent(content)
			
			if event != nil {
				c.mutex.Lock()
				c.events = append(c.events, *event)
				
				if c.logFile != nil {
					data, _ := json.Marshal(event)
					if !firstEntry {
						c.logFile.WriteString(",\n")
					}
					c.logFile.Write(data)
					firstEntry = false
				}
				c.mutex.Unlock()

				if event.IsSensitive {
					log.Printf("🚨 敏感剪贴板: [%s] %s...", event.SensitiveKey, event.Preview)
				}
			}
			
			c.lastContent = content
		}

		<-ticker.C
	}
}

// getClipboardContent 获取剪贴板内容
func (c *ClipboardMonitor) getClipboardContent() string {
	cmd := exec.Command("pbpaste")
	output, err := cmd.Output()
	if err != nil {
		return ""
	}
	return string(output)
}

// createEvent 创建剪贴板事件
func (c *ClipboardMonitor) createEvent(content string) *ClipboardEvent {
	timestamp := time.Since(c.startTime).Seconds()

	preview := content
	if len(preview) > 100 {
		preview = preview[:100] + "..."
	}

	event := &ClipboardEvent{
		Timestamp:   timestamp,
		ContentType: "text",
		ContentSize: len(content),
		Preview:     preview,
	}

	// 检查是否包含敏感关键词
	contentLower := strings.ToLower(content)
	for _, keyword := range SensitiveFileKeywords {
		if strings.Contains(contentLower, strings.ToLower(keyword)) {
			event.IsSensitive = true
			event.SensitiveKey = keyword
			break
		}
	}

	return event
}

// BrowserHistoryMonitor 浏览器历史监控器（通过网络请求监控）
type BrowserHistoryMonitor struct {
	running   bool
	mutex     sync.Mutex
	events    []URLEvent
	logFile   *os.File
	startTime time.Time
	cmd       *exec.Cmd
}

// URLEvent URL访问事件
type URLEvent struct {
	Timestamp   float64 `json:"timestamp"`
	URL         string  `json:"url"`
	Domain      string  `json:"domain"`
	AppName     string  `json:"app_name"`
	Category    string  `json:"category"`
	RiskLevel   string  `json:"risk_level"`
	ProcessName string  `json:"process_name"`
}

// NewBrowserHistoryMonitor 创建浏览器历史监控器
func NewBrowserHistoryMonitor() *BrowserHistoryMonitor {
	return &BrowserHistoryMonitor{}
}

// Start 开始监控（使用 nettop 监控网络连接）
func (b *BrowserHistoryMonitor) Start(outputPath string) error {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	if b.running {
		return fmt.Errorf("网络监控已在运行中")
	}

	var err error
	b.logFile, err = os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("创建网络日志文件失败: %v", err)
	}

	b.events = []URLEvent{}
	b.startTime = time.Now()
	b.running = true

	b.logFile.WriteString("[\n")

	log.Printf("🌐 开始网络监控: %s", outputPath)

	// 使用 nettop 监控网络连接
	go b.monitorNetwork()

	return nil
}

// Stop 停止监控
func (b *BrowserHistoryMonitor) Stop() error {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	if !b.running {
		return fmt.Errorf("网络监控未在运行")
	}

	b.running = false

	if b.cmd != nil && b.cmd.Process != nil {
		b.cmd.Process.Signal(syscall.SIGTERM)
		b.cmd.Process.Kill()
	}

	if b.logFile != nil {
		b.logFile.WriteString("\n]")
		b.logFile.Close()
		b.logFile = nil
	}

	log.Printf("🌐 网络监控已停止, 共 %d 个事件", len(b.events))
	return nil
}

// monitorNetwork 监控网络连接
func (b *BrowserHistoryMonitor) monitorNetwork() {
	// 使用 lsof 监控网络连接
	b.cmd = exec.Command("lsof", "-i", "-n", "-P", "+c", "0")

	stdout, err := b.cmd.StdoutPipe()
	if err != nil {
		log.Printf("网络监控启动失败: %v", err)
		return
	}

	if err := b.cmd.Start(); err != nil {
		log.Printf("网络监控启动失败: %v", err)
		return
	}

	scanner := bufio.NewScanner(stdout)
	firstEntry := true
	processedHosts := make(map[string]bool)

	for scanner.Scan() {
		b.mutex.Lock()
		if !b.running {
			b.mutex.Unlock()
			break
		}
		b.mutex.Unlock()

		line := scanner.Text()
		event := b.parseLsofLine(line, processedHosts)
		
		if event != nil {
			b.mutex.Lock()
			b.events = append(b.events, *event)
			
			if b.logFile != nil {
				data, _ := json.Marshal(event)
				if !firstEntry {
					b.logFile.WriteString(",\n")
				}
				b.logFile.Write(data)
				firstEntry = false
			}
			b.mutex.Unlock()

			if event.RiskLevel != "" {
				log.Printf("🌐 [%s] 访问: %s (%s)", event.RiskLevel, event.Domain, event.Category)
			}
		}
	}

	b.cmd.Wait()
}

// parseLsofLine 解析 lsof 输出行
func (b *BrowserHistoryMonitor) parseLsofLine(line string, processed map[string]bool) *URLEvent {
	fields := strings.Fields(line)
	if len(fields) < 9 {
		return nil
	}

	processName := fields[0]
	
	// 只关注浏览器和可能发起网络请求的应用
	if !isBrowser(processName) && !isNetworkApp(processName) {
		return nil
	}

	// 提取目标地址
	nameField := fields[len(fields)-1]
	if !strings.Contains(nameField, "->") {
		return nil
	}

	parts := strings.Split(nameField, "->")
	if len(parts) < 2 {
		return nil
	}

	target := parts[1]
	// 移除端口号
	if idx := strings.LastIndex(target, ":"); idx != -1 {
		target = target[:idx]
	}

	// 去重
	key := processName + "-" + target
	if processed[key] {
		return nil
	}
	processed[key] = true

	// 检查是否是敏感网站
	for domain, risk := range BlacklistWebsites {
		if strings.Contains(target, domain) || strings.HasSuffix(target, domain) {
			return &URLEvent{
				Timestamp:   time.Since(b.startTime).Seconds(),
				Domain:      domain,
				AppName:     risk.Name,
				Category:    risk.Category,
				RiskLevel:   "高",
				ProcessName: processName,
			}
		}
	}

	return nil
}

// isNetworkApp 检查是否是可能发起网络请求的应用
func isNetworkApp(appName string) bool {
	networkApps := []string{
		"QQ", "WeChat", "微信", "DingTalk", "钉钉", "Feishu", "飞书",
		"Zoom", "TencentMeeting", "腾讯会议",
		"Doubao", "豆包", "yuanbao", "元宝",
	}
	for _, app := range networkApps {
		if strings.Contains(appName, app) {
			return true
		}
	}
	return false
}
