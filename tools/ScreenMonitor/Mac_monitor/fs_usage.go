// fs_usage.go
//
// 🏆 推荐方案 (Recommended Solution)
//
// 原理: 使用 macOS 内置的 `fs_usage` 命令实时捕获文件系统调用。
// 优点:
// 1. ✅ 能获取进程信息 (Process Name & PID) - 解决了 "谁做了什么" 的核心问题。
// 2. ✅ 实时性高 - 直接 hook 内核事件。
// 3. ✅ 无需额外依赖 - fs_usage 是系统自带工具。
//
// 缺点:
// 1. 需要 sudo 权限。
// 2. 输出数据量大，需要精细过滤 (已在此代码中实现)。
//

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// LogEntry 结构化日志
type LogEntry struct {
	Timestamp   string
	Operation   string
	Path        string
	ProcessName string
	ProcessID   string
	User        string
}

// 文件操作缓存,用于推断真实意图
type FileOperation struct {
	Path      string
	Operation string
	Time      time.Time
}

var recentOps = make(map[string]*FileOperation)

// 关注的操作
var operationMap = map[string]string{
	"open":   "Opened",
	"create": "Created",
	"mkdir":  "New Folder",
	"rename": "Renamed/Moved",
	"unlink": "Deleted",
	"rmdir":  "Deleted Folder",
	"write":  "Modified",
}

// 必须忽略的系统路径前缀
var ignorePrefixes = []string{
	"/dev/",
	"/sys",
	"/private/var",
	"/private/tmp",
	"/bin/",
	"/usr/",
	"/sbin/",
	"/System/",
	"/tmp/",
	"/opt/",
	"/Library/",
	// "/Applications/", // 允许监控 Applications 目录下的文件变化，虽然通常我们监控的是用户数据
}

func main() {
	if os.Geteuid() != 0 {
		log.Fatal("请使用 sudo 运行此程序")
	}

	myPid := strconv.Itoa(os.Getpid())
	currentUser, _ := user.Current()
	username := "root"
	homeDir := ""
	if currentUser != nil {
		username = currentUser.Username
		homeDir = currentUser.HomeDir
	}

	fmt.Printf("🔍 启动fs_usage文件监控... (当前用户: %s)\n", username)
	fmt.Println("📋 过滤规则: 系统文件、应用缓存、隐藏文件、临时文件")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 启动 fs_usage
	cmd := exec.Command("fs_usage", "-f", "filesys", "-w")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(stdout)
	buf := make([]byte, 0, 128*1024)
	scanner.Buffer(buf, 1024*1024)

	// 跳过头部
	for i := 0; i < 5; i++ {
		scanner.Scan()
	}

	for scanner.Scan() {
		line := scanner.Text()

		// 1. 过滤自身
		if strings.Contains(line, "."+myPid) {
			continue
		}

		entry := parseAndFilter(line, homeDir)
		if entry != nil {
			entry.User = username
			printEntry(entry)
		}
	}
	cmd.Wait()
}

func parseAndFilter(line string, homeDir string) *LogEntry {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}
	fields := strings.Fields(line)
	if len(fields) < 4 {
		return nil
	}

	timestamp := fields[0]
	rawProcess := fields[len(fields)-1]

	processName := rawProcess
	processID := ""
	if idx := strings.LastIndex(processName, "."); idx != -1 {
		processID = processName[idx+1:]
		processName = processName[:idx]
	}

	// 过滤系统噪音进程
	systemProcesses := []string{"mdworker", "mds", "git", "notifyd", "bird", "com.apple", "kernel_task"}
	for _, sp := range systemProcesses {
		if processName == sp || strings.Contains(processName, sp) {
			return nil
		}
	}

	rawOp := fields[1]
	cleanOp := strings.Split(strings.Split(strings.ToLower(rawOp), "(")[0], "[")[0]

	readableOp, exists := operationMap[cleanOp]
	if !exists {
		return nil
	}

	// --- 改进的路径解析 ---
	// fs_usage 输出格式: TIMESTAMP OPERATION [optional_params] PATH [optional_time] PROCESS.PID
	// 需要找到路径部分,排除最后两个字段(时间和进程)
	pathFields := []string{}
	for i := 2; i < len(fields)-1; i++ {
		field := fields[i]
		// 跳过时间字段 (格式: 0.000123)
		if strings.Contains(field, ".") && len(field) < 10 {
			if _, err := strconv.ParseFloat(field, 64); err == nil {
				continue
			}
		}
		// 跳过 fd= 等参数
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

	// 确保是有效路径
	if !strings.HasPrefix(fullPathStr, "/") {
		slashIdx := strings.Index(fullPathStr, "/")
		if slashIdx == -1 {
			return nil
		}
		fullPathStr = fullPathStr[slashIdx:]
	}

	// --- 核心过滤逻辑 ---

	// 1. 忽略系统路径前缀
	for _, prefix := range ignorePrefixes {
		if strings.HasPrefix(fullPathStr, prefix) {
			return nil
		}
	}

	// 2. 忽略 Library (缓存、日志等)
	if strings.Contains(fullPathStr, "/Library/") {
		return nil
	}

	// 3. 忽略隐藏文件/文件夹
	baseName := filepath.Base(fullPathStr)
	if strings.HasPrefix(baseName, ".") {
		return nil
	}
	if strings.Contains(fullPathStr, "/.") {
		return nil
	}

	// 4. 忽略临时文件
	ext := strings.ToLower(filepath.Ext(fullPathStr))
	tempExts := []string{".tmp", ".lock", ".dat", ".plist", ".db", ".log", ".crdownload", ".download"}
	for _, te := range tempExts {
		if ext == te {
			return nil
		}
	}
	// WPS/Office 临时文件模式
	if strings.Contains(baseName, "~tmp") || strings.HasPrefix(baseName, "~$") {
		return nil
	}

	// 5. 过滤无关的 open 操作
	if cleanOp == "open" {
		// 宽松模式：捕获所有应用的open操作，只要它不是系统噪音
		// 不再限制只在 Documents/Desktop/Downloads

		// 进一步过滤掉毫无意义的 open (例如 .plist, .strings 资源文件读取)
		// 但为了保险起见，如果用户要求"任何场景"，我们先不过滤资源文件，或者只过滤非常明显的
		if strings.HasSuffix(fullPathStr, ".plist") || strings.HasSuffix(fullPathStr, ".strings") || strings.HasSuffix(fullPathStr, ".icns") {
			return nil
		}

		return &LogEntry{
			Timestamp:   timestamp,
			Operation:   "Opened", // 直接标记为 Opened
			Path:        fullPathStr,
			ProcessName: processName,
			ProcessID:   processID,
		}
	}

	// --- 智能操作推断 ---
	finalOp := smartInferOperation(fullPathStr, cleanOp, readableOp, processName)

	return &LogEntry{
		Timestamp:   timestamp,
		Operation:   finalOp,
		Path:        fullPathStr,
		ProcessName: processName,
		ProcessID:   processID,
	}
}

// 智能推断真实操作意图
func smartInferOperation(path, rawOp, readableOp, process string) string {
	// 检测是否移动到废纸篓
	if rawOp == "rename" {
		if strings.Contains(path, "/.Trash/") || strings.Contains(path, "/Trash/") {
			return "🗑️  Moved to Trash"
		}
		// 如果之前这个文件刚被创建,这可能是"保存"操作
		if recent, exists := recentOps[path]; exists {
			if time.Since(recent.Time) < 2*time.Second && recent.Operation == "create" {
				delete(recentOps, path)
				return "💾 Saved (Created)"
			}
		}
		return "📦 Renamed/Moved"
	}

	if rawOp == "unlink" {
		// 真正的删除(不经过废纸篓)
		return "❌ Deleted"
	}

	if rawOp == "create" {
		// 记录创建操作,用于后续推断
		recentOps[path] = &FileOperation{
			Path:      path,
			Operation: "create",
			Time:      time.Now(),
		}
		return "✨ Created"
	}

	if rawOp == "write" || rawOp == "modify" {
		return "✏️  Modified"
	}

	if rawOp == "mkdir" {
		return "📁 New Folder"
	}

	if rawOp == "rmdir" {
		return "🗑️  Deleted Folder"
	}

	return readableOp
}

func printEntry(e *LogEntry) {
	colorReset := "\033[0m"
	colorTime := "\033[0;90m" // 灰色
	colorApp := "\033[1;36m"  // 青色
	colorOp := "\033[1;33m"   // 黄色
	colorFile := "\033[1;37m" // 白色

	// 根据操作类型设置颜色
	if strings.Contains(e.Operation, "Deleted") || strings.Contains(e.Operation, "Trash") {
		colorOp = "\033[1;31m" // 红色
	} else if strings.Contains(e.Operation, "Created") || strings.Contains(e.Operation, "Saved") {
		colorOp = "\033[1;32m" // 绿色
	} else if strings.Contains(e.Operation, "Modified") {
		colorOp = "\033[1;35m" // 紫色
	}

	// 简化路径显示
	displayPath := e.Path
	if strings.HasPrefix(displayPath, "/Users/") {
		parts := strings.SplitN(displayPath, "/", 4)
		if len(parts) >= 4 {
			displayPath = "~/" + parts[3]
		}
	}

	fmt.Printf("%s[%s]%s %s%s%s %s%s%s %s%s%s\n",
		colorTime, e.Timestamp, colorReset,
		colorApp, e.ProcessName, colorReset,
		colorOp, e.Operation, colorReset,
		colorFile, displayPath, colorReset,
	)

	// 清理旧的操作记录(避免内存泄漏)
	if len(recentOps) > 1000 {
		for k, v := range recentOps {
			if time.Since(v.Time) > 10*time.Second {
				delete(recentOps, k)
			}
		}
	}
}
