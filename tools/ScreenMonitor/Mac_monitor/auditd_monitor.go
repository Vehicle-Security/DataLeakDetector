// auditd_monitor.go
//
// 🥈 备选方案 (Alternative Solution)
//
// 原理: 使用 macOS 的 BSM (Basic Security Module) 审计系统。
// 优点:
// 1. ✅ 安全级别高，难以规避。
// 2. ✅ 包含用户和进程信息。
//
// 缺点:
// 1. ❌ 配置复杂 (可能需要修改 /etc/security/audit_control)。
// 2. ❌ 解析复杂 (praudit 输出格式)。
// 3. ❌ 性能开销略大。
//

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"os/user"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// AuditEvent 审计事件
type AuditEvent struct {
	Timestamp   time.Time
	Operation   string
	Path        string
	ProcessName string
	User        string
}

var (
	// 忽略的路径前缀
	ignorePrefixes = []string{
		"/dev/",
		"/sys",
		"/private/var/db",
		"/private/var/folders",
		"/private/tmp",
		"/bin/",
		"/usr/",
		"/sbin/",
		"/System/",
		"/tmp/",
		"/opt/",
		"/Applications/",
	}

	homeDir string
)

func main() {
	if os.Geteuid() != 0 {
		log.Fatal("❌ 请使用 sudo 运行此程序 (BSM 审计需要 root 权限)")
	}

	currentUser, _ := user.Current()
	username := "root"
	homeDir = os.Getenv("HOME")
	
	// 获取实际用户的 HOME
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" {
		if u, err := user.Lookup(sudoUser); err == nil {
			homeDir = u.HomeDir
			username = sudoUser
		}
	} else if currentUser != nil {
		username = currentUser.Username
	}

	fmt.Printf("🔍 BSM 审计监控启动... (用户: %s)\n", username)
	fmt.Println("📋 监控目录: ~/Desktop, ~/Documents, ~/Downloads")
	fmt.Println("💡 BSM 特点: macOS 原生审计系统、安全合规")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 检查并启动 BSM 审计
	if err := ensureAuditRunning(); err != nil {
		log.Fatal("❌ 无法启动 BSM 审计:", err)
	}

	fmt.Println("✅ BSM 审计系统运行中")
	fmt.Println("⚠️  注意: BSM 审计日志可能有延迟，且不会捕获所有文件操作")
	fmt.Println("💡 提示: 推荐使用 fs_usage 或 FSEvent 方案获得更好的实时性\n")

	// 捕获退出信号
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// 启动审计日志监控
	go monitorAuditLog()

	// 等待退出
	<-sigChan
	fmt.Println("\n\n🛑 停止监控...")
}

// 确保审计系统运行
func ensureAuditRunning() error {
	// 1. 尝试直接启动/重置审计服务 (audit -s)
	// 无论当前状态如何，运行 audit -s 通常是安全的，它会初始化或重载配置
	fmt.Println("📝 正在初始化/启动 BSM 审计系统...")
	startCmd := exec.Command("audit", "-s")
	if err := startCmd.Run(); err != nil {
		// 如果 -s 失败，可能是配置问题或权限问题
		log.Printf("⚠️ 警告: audit -s 启动返回错误 (可能是已运行或权限限制): %v", err)
	}
	
	// 等待一小会儿让服务就绪
	time.Sleep(500 * time.Millisecond)

	// 2. 验证审计是否运行 (audit -n)
	// audit -n 通知守护进程重读配置，如果守护进程没在跑，这里会报错
	checkCmd := exec.Command("audit", "-n")
	if err := checkCmd.Run(); err != nil {
		// 如果 audit -n 失败，说明守护进程没有响应
		return fmt.Errorf("审计守护进程未响应 (exit code 255)，请尝试手动运行 'sudo audit -s' 检查错误")
	}

	return nil
}

// 监控审计日志
func monitorAuditLog() {
	// 查找当前审计日志文件
	auditDir := "/var/audit"
	
	// 使用 praudit 实时解析审计日志
	// -l: 单行输出模式
	cmd := exec.Command("sh", "-c", fmt.Sprintf("tail -F %s/current 2>/dev/null | praudit -l", auditDir))
	
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("❌ 无法创建审计日志流: %v", err)
		log.Println("💡 提示: 可能需要配置审计策略，请参考 'man audit'")
		return
	}

	if err := cmd.Start(); err != nil {
		log.Printf("❌ 无法启动审计日志读取: %v", err)
		log.Println("💡 提示: 请确保有读取 /var/audit 的权限")
		return
	}

	fmt.Println("✅ 开始监控审计日志\n")

	scanner := bufio.NewScanner(stdout)
	buf := make([]byte, 0, 256*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		processAuditLine(line)
	}

	if err := scanner.Err(); err != nil {
		log.Printf("⚠️  读取审计日志出错: %v", err)
	}
}

// 处理审计日志行
func processAuditLine(line string) {
	// praudit -l 输出格式示例:
	// header,175,11,open(2),0,Thu Nov 28 20:00:00 2024, + 123 msec,path,/Users/xxx/file.txt,...

	// 必须包含路径信息
	if !strings.Contains(line, "path,") {
		return
	}

	// 提取操作类型
	var operation string
	var emoji string

	if strings.Contains(line, "open(2)") || strings.Contains(line, "AUE_OPEN") {
		return // open 太多，忽略
	} else if strings.Contains(line, "unlink") || strings.Contains(line, "AUE_UNLINK") {
		operation = "Deleted"
		emoji = "❌"
	} else if strings.Contains(line, "rename") || strings.Contains(line, "AUE_RENAME") {
		operation = "Renamed/Moved"
		emoji = "📦"
	} else if strings.Contains(line, "mkdir") || strings.Contains(line, "AUE_MKDIR") {
		operation = "New Folder"
		emoji = "📁"
	} else if strings.Contains(line, "rmdir") || strings.Contains(line, "AUE_RMDIR") {
		operation = "Deleted Folder"
		emoji = "🗑️"
	} else if strings.Contains(line, "create") || strings.Contains(line, "AUE_OPEN_W") {
		operation = "Created"
		emoji = "✨"
	} else if strings.Contains(line, "write") {
		operation = "Modified"
		emoji = "✏️"
	} else {
		return // 其他操作忽略
	}

	// 提取路径
	path := extractPath(line)
	if path == "" {
		return
	}

	// 过滤路径
	if shouldIgnorePath(path) {
		return
	}

	// 检测废纸篓
	if strings.Contains(path, ".Trash") {
		operation = "Moved to Trash"
		emoji = "🗑️"
	}

	// 提取进程名
	processName := extractProcessName(line)

	// 打印事件
	printEvent(path, operation, emoji, processName)
}

// 提取路径
func extractPath(line string) string {
	// 查找 "path," 后面的内容
	if idx := strings.Index(line, "path,"); idx != -1 {
		remaining := line[idx+5:] // 跳过 "path,"
		
		// 路径可能在引号中或以逗号结束
		if strings.HasPrefix(remaining, "\"") {
			// 引号包围的路径
			if endIdx := strings.Index(remaining[1:], "\""); endIdx != -1 {
				return remaining[1 : endIdx+1]
			}
		} else {
			// 逗号分隔的路径
			if endIdx := strings.Index(remaining, ","); endIdx != -1 {
				return remaining[:endIdx]
			}
			return remaining
		}
	}
	return ""
}

// 提取进程名
func extractProcessName(line string) string {
	// 审计日志中进程信息格式: subject,xxx,xxx,xxx,xxx,xxx,processname,...
	parts := strings.Split(line, ",")
	for i, part := range parts {
		if part == "subject" && i+6 < len(parts) {
			return parts[i+6]
		}
		if part == "process" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return "unknown"
}

// 检查是否应该忽略路径
func shouldIgnorePath(path string) bool {
	// 系统路径
	for _, prefix := range ignorePrefixes {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}

	// Library 目录
	if strings.Contains(path, "/Library/Caches") || 
	   strings.Contains(path, "/Library/Logs") ||
	   strings.Contains(path, "/Library/Application Support") {
		return true
	}

	// 隐藏文件
	baseName := filepath.Base(path)
	if strings.HasPrefix(baseName, ".") {
		return true
	}

	// 临时文件
	if strings.Contains(baseName, "~tmp") || strings.HasPrefix(baseName, "~$") {
		return true
	}

	// 临时扩展名
	ext := strings.ToLower(filepath.Ext(path))
	tempExts := []string{".tmp", ".lock", ".dat", ".plist", ".db", ".log", ".swp"}
	for _, te := range tempExts {
		if ext == te {
			return true
		}
	}

	// 只关注用户目录
	if !strings.Contains(path, "/Users/") {
		return true
	}

	// 只关注 Desktop, Documents, Downloads
	if !strings.Contains(path, "/Desktop") && 
	   !strings.Contains(path, "/Documents") && 
	   !strings.Contains(path, "/Downloads") {
		return true
	}

	return false
}

// 打印事件
func printEvent(path, operation, emoji, processName string) {
	// 简化路径
	displayPath := path
	if strings.HasPrefix(displayPath, homeDir) {
		displayPath = "~" + strings.TrimPrefix(displayPath, homeDir)
	}

	// 颜色
	colorReset := "\033[0m"
	colorTime := "\033[0;90m"
	colorApp := "\033[1;36m"
	colorOp := "\033[1;33m"
	colorFile := "\033[1;37m"

	if strings.Contains(operation, "Deleted") || strings.Contains(operation, "Trash") {
		colorOp = "\033[1;31m"
	} else if strings.Contains(operation, "Created") {
		colorOp = "\033[1;32m"
	} else if strings.Contains(operation, "Modified") {
		colorOp = "\033[1;35m"
	}

	timestamp := time.Now().Format("15:04:05.000")
	fmt.Printf("%s[%s]%s %s%s%s %s %s%s%s %s%s%s\n",
		colorTime, timestamp, colorReset,
		colorApp, processName, colorReset,
		emoji,
		colorOp, operation, colorReset,
		colorFile, displayPath, colorReset,
	)
}