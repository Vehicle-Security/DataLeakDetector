// server/risk_analyzer.go
// 风险分析器 - 基于进程+文件+操作的综合风险评估
package main

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// RiskLevel 风险等级
type RiskLevel string

const (
	RiskLevelNone   RiskLevel = "none"
	RiskLevelLow    RiskLevel = "low"
	RiskLevelMedium RiskLevel = "medium"
	RiskLevelHigh   RiskLevel = "high"
)

// RiskEvent 风险事件
type RiskEvent struct {
	Timestamp   string    `json:"timestamp"`
	ProcessName string    `json:"process_name"`
	FilePath    string    `json:"file_path"`
	FileName    string    `json:"file_name"`
	Action      string    `json:"action"`
	RiskLevel   RiskLevel `json:"risk_level"`
	RiskReason  string    `json:"risk_reason"`
	Category    string    `json:"category"`     // 应用类别
	FileType    string    `json:"file_type"`    // 文件类型
	IsSensitive bool      `json:"is_sensitive"` // 是否敏感文件
}

// MonitorConfig 监控配置（从 YAML 加载）
type MonitorConfig struct {
	BlacklistApps       []AppConfig           `yaml:"blacklist_apps"`
	BlacklistWebsites   []WebsiteConfig       `yaml:"blacklist_websites"`
	SensitiveKeywords   []string              `yaml:"sensitive_keywords"`
	SensitiveExtensions []string              `yaml:"sensitive_extensions"`
	SystemWhitelist     SystemWhitelistConfig `yaml:"system_whitelist"`
}

// SystemWhitelistConfig 系统白名单配置
type SystemWhitelistConfig struct {
	IgnoreProcesses      []string `yaml:"ignore_processes"`      // 需要忽略的系统进程
	IgnorePathPrefixes   []string `yaml:"ignore_path_prefixes"`  // 需要忽略的路径前缀
	CorrelationProcesses []string `yaml:"correlation_processes"` // 需要保留用于关联分析的进程
}

// AppConfig 应用配置
type AppConfig struct {
	Name     string   `yaml:"name"`
	Aliases  []string `yaml:"aliases"`
	Category string   `yaml:"category"`
}

// WebsiteConfig 网站配置
type WebsiteConfig struct {
	Domain   string `yaml:"domain"`
	Name     string `yaml:"name"`
	Category string `yaml:"category"`
}

// RiskAnalyzer 风险分析器
type RiskAnalyzer struct {
	config            *MonitorConfig
	blacklistAppMap   map[string]*AppConfig // 快速查找 map
	blacklistWebMap   map[string]*WebsiteConfig
	sensitiveKeywords []string
	sensitiveExts     map[string]bool
	mutex             sync.RWMutex
}

// NewRiskAnalyzer 创建风险分析器（使用默认配置）
func NewRiskAnalyzer() *RiskAnalyzer {
	ra := &RiskAnalyzer{
		blacklistAppMap: make(map[string]*AppConfig),
		blacklistWebMap: make(map[string]*WebsiteConfig),
		sensitiveExts:   make(map[string]bool),
	}

	// 使用默认配置初始化
	ra.initDefaults()
	return ra
}

// NewRiskAnalyzerWithConfig 从配置文件创建风险分析器
func NewRiskAnalyzerWithConfig(configPath string) (*RiskAnalyzer, error) {
	ra := &RiskAnalyzer{
		blacklistAppMap: make(map[string]*AppConfig),
		blacklistWebMap: make(map[string]*WebsiteConfig),
		sensitiveExts:   make(map[string]bool),
	}

	if err := ra.LoadConfig(configPath); err != nil {
		// 加载失败时使用默认配置
		ra.initDefaults()
		return ra, err
	}

	return ra, nil
}

// initDefaults 初始化默认配置
func (ra *RiskAnalyzer) initDefaults() {
	// 使用现有的 config.go 中的配置
	for name, risk := range BlacklistApps {
		ra.blacklistAppMap[strings.ToLower(name)] = &AppConfig{
			Name:     risk.Name,
			Category: risk.Category,
		}
	}

	for domain, risk := range BlacklistWebsites {
		ra.blacklistWebMap[strings.ToLower(domain)] = &WebsiteConfig{
			Domain:   domain,
			Name:     risk.Name,
			Category: risk.Category,
		}
	}

	ra.sensitiveKeywords = SensitiveFileKeywords

	for _, ext := range SensitiveFileExtensions {
		ra.sensitiveExts[strings.ToLower(ext)] = true
	}
}

// LoadConfig 从 YAML 文件加载配置
func (ra *RiskAnalyzer) LoadConfig(configPath string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}

	var config MonitorConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return err
	}

	ra.mutex.Lock()
	defer ra.mutex.Unlock()

	ra.config = &config

	// 重建查找 map
	ra.blacklistAppMap = make(map[string]*AppConfig)
	for i := range config.BlacklistApps {
		app := &config.BlacklistApps[i]
		ra.blacklistAppMap[strings.ToLower(app.Name)] = app
		// 添加别名
		for _, alias := range app.Aliases {
			ra.blacklistAppMap[strings.ToLower(alias)] = app
		}
	}

	ra.blacklistWebMap = make(map[string]*WebsiteConfig)
	for i := range config.BlacklistWebsites {
		site := &config.BlacklistWebsites[i]
		ra.blacklistWebMap[strings.ToLower(site.Domain)] = site
	}

	ra.sensitiveKeywords = config.SensitiveKeywords

	ra.sensitiveExts = make(map[string]bool)
	for _, ext := range config.SensitiveExtensions {
		ra.sensitiveExts[strings.ToLower(ext)] = true
	}

	return nil
}

// AnalyzeRisk 分析文件操作风险
// 核心函数：基于进程名+文件路径+操作类型判断风险
func (ra *RiskAnalyzer) AnalyzeRisk(processName string, filePath string, action string) *RiskEvent {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	// 规范化输入
	processLower := strings.ToLower(processName)
	actionLower := strings.ToLower(action)
	fileName := filepath.Base(filePath)
	fileExt := strings.ToLower(filepath.Ext(filePath))

	// 1. 检查进程是否在黑名单
	appConfig := ra.findBlacklistApp(processLower)
	isBlacklistApp := appConfig != nil

	// 2. 检查是否为读取/打开操作
	isReadAction := ra.isReadOperation(actionLower)

	// 3. 检查文件是否敏感
	isSensitiveFile := ra.isSensitiveFile(fileName, fileExt)

	// 4. 综合判断风险等级
	var riskLevel RiskLevel
	var riskReason string
	var category string

	if appConfig != nil {
		category = appConfig.Category
	}

	switch {
	case isBlacklistApp && isReadAction && isSensitiveFile:
		// 黑名单应用 + 读取操作 + 敏感文件 = 高风险
		riskLevel = RiskLevelHigh
		riskReason = "黑名单应用读取敏感文件，疑似外发"

	case isBlacklistApp && isReadAction:
		// 黑名单应用 + 读取操作 = 中风险
		riskLevel = RiskLevelMedium
		riskReason = "黑名单应用读取文件"

	case isBlacklistApp && isSensitiveFile:
		// 黑名单应用 + 敏感文件（非读取操作）= 中风险
		riskLevel = RiskLevelMedium
		riskReason = "黑名单应用访问敏感文件"

	case isSensitiveFile && isReadAction:
		// 敏感文件被读取 = 低风险（需要结合其他信息）
		riskLevel = RiskLevelLow
		riskReason = "敏感文件被读取"

	default:
		// 无风险
		return nil
	}

	return &RiskEvent{
		Timestamp:   time.Now().Format("2006-01-02T15:04:05.000"),
		ProcessName: processName,
		FilePath:    filePath,
		FileName:    fileName,
		Action:      action,
		RiskLevel:   riskLevel,
		RiskReason:  riskReason,
		Category:    category,
		FileType:    fileExt,
		IsSensitive: isSensitiveFile,
	}
}

// findBlacklistApp 查找黑名单应用
func (ra *RiskAnalyzer) findBlacklistApp(processName string) *AppConfig {
	// 直接匹配
	if app, ok := ra.blacklistAppMap[processName]; ok {
		return app
	}

	// 模糊匹配（进程名可能包含版本号等）
	for name, app := range ra.blacklistAppMap {
		if strings.Contains(processName, name) || strings.Contains(name, processName) {
			return app
		}
	}

	return nil
}

// isReadOperation 判断是否为读取操作
func (ra *RiskAnalyzer) isReadOperation(action string) bool {
	readActions := []string{
		"open", "opened", "read", "openat", "pread",
		"file-read", "access", "stat", "fstat",
	}

	for _, readAction := range readActions {
		if strings.Contains(action, readAction) {
			return true
		}
	}

	return false
}

// isSensitiveFile 判断文件是否敏感
func (ra *RiskAnalyzer) isSensitiveFile(fileName, fileExt string) bool {
	// 检查扩展名
	if ra.sensitiveExts[fileExt] {
		return true
	}

	// 检查文件名关键词
	fileNameLower := strings.ToLower(fileName)
	for _, keyword := range ra.sensitiveKeywords {
		if strings.Contains(fileNameLower, strings.ToLower(keyword)) {
			return true
		}
	}

	return false
}

// IsBlacklistProcess 检查进程是否在黑名单（公开方法）
func (ra *RiskAnalyzer) IsBlacklistProcess(processName string) bool {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()
	return ra.findBlacklistApp(strings.ToLower(processName)) != nil
}

// GetAppCategory 获取应用类别
func (ra *RiskAnalyzer) GetAppCategory(processName string) string {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	if app := ra.findBlacklistApp(strings.ToLower(processName)); app != nil {
		return app.Category
	}
	return ""
}

// ShouldIgnoreProcess 检查进程是否应该被忽略（基于配置的系统白名单）
func (ra *RiskAnalyzer) ShouldIgnoreProcess(processName string) bool {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	if ra.config == nil {
		return false
	}

	processLower := strings.ToLower(processName)
	for _, ignoreProc := range ra.config.SystemWhitelist.IgnoreProcesses {
		if strings.ToLower(ignoreProc) == processLower ||
			strings.Contains(processLower, strings.ToLower(ignoreProc)) {
			return true
		}
	}
	return false
}

// ShouldIgnorePath 检查路径是否应该被忽略（基于配置的路径白名单）
func (ra *RiskAnalyzer) ShouldIgnorePath(filePath string) bool {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	if ra.config == nil {
		return false
	}

	for _, prefix := range ra.config.SystemWhitelist.IgnorePathPrefixes {
		if strings.HasPrefix(filePath, prefix) {
			return true
		}
	}
	return false
}

// IsCorrelationProcess 检查进程是否是需要关联分析的进程
func (ra *RiskAnalyzer) IsCorrelationProcess(processName string) bool {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	if ra.config == nil {
		return false
	}

	processLower := strings.ToLower(processName)
	for _, corrProc := range ra.config.SystemWhitelist.CorrelationProcesses {
		if strings.ToLower(corrProc) == processLower ||
			strings.Contains(processLower, strings.ToLower(corrProc)) {
			return true
		}
	}
	return false
}

// GetSystemWhitelist 获取系统白名单配置（用于外部访问）
func (ra *RiskAnalyzer) GetSystemWhitelist() *SystemWhitelistConfig {
	ra.mutex.RLock()
	defer ra.mutex.RUnlock()

	if ra.config == nil {
		return nil
	}
	return &ra.config.SystemWhitelist
}
