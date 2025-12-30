// fsevent_monitor.go
//
// 🥉 基准方案 (Baseline Solution)
//
// 原理: 使用 macOS CoreServices 的 FSEvents API。
// 优点:
// 1. ✅ 极高的效率和性能。
// 2. ✅ macOS 原生推荐的文件监控方式。
//
// 缺点:
// 1. ❌ 无法获取进程信息 (No Process Info) - 不知道是谁修改了文件。
// 2. ❌ 只能监控目录变化，需要递归。
//

package main

/*
#cgo LDFLAGS: -framework CoreServices
#include <CoreServices/CoreServices.h>
#include <stdlib.h>
#include <stdio.h>

// 全局 Go 回调声明
extern void goFSEventCallback(int numEvents, void *eventPaths, void *eventFlags);

// C 回调函数 - 使用 static inline 避免重复定义
static inline void fsEventCallback(
    ConstFSEventStreamRef streamRef,
    void *clientCallBackInfo,
    size_t numEvents,
    void *eventPaths,
    const FSEventStreamEventFlags eventFlags[],
    const FSEventStreamEventId eventIds[])
{
    goFSEventCallback((int)numEvents, eventPaths, (void*)eventFlags);
}

// 创建 FSEventStream - 使用 static inline
static inline FSEventStreamRef createEventStream(CFArrayRef pathsToWatch) {
    FSEventStreamContext context = {0, NULL, NULL, NULL, NULL};
    
    FSEventStreamRef stream = FSEventStreamCreate(
        kCFAllocatorDefault,
        (FSEventStreamCallback)&fsEventCallback,
        &context,
        pathsToWatch,
        kFSEventStreamEventIdSinceNow,
        0.3,
        kFSEventStreamCreateFlagFileEvents | 
        kFSEventStreamCreateFlagNoDefer |
        kFSEventStreamCreateFlagWatchRoot
    );
    
    return stream;
}

// 启动监控 - 使用 static inline
static inline void startEventStream(FSEventStreamRef stream) {
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    FSEventStreamSetDispatchQueue(stream, queue);
    FSEventStreamStart(stream);
}

// 停止监控 - 使用 static inline
static inline void stopEventStream(FSEventStreamRef stream) {
    FSEventStreamStop(stream);
    FSEventStreamSetDispatchQueue(stream, NULL);
    FSEventStreamInvalidate(stream);
    FSEventStreamRelease(stream);
}

// 辅助函数 - 使用 static inline
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
	"fmt"
	"os"
	"os/signal"
	"os/user"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

// 事件类型标志
const (
	kFSEventStreamEventFlagItemCreated     = 0x00000100
	kFSEventStreamEventFlagItemRemoved     = 0x00000200
	kFSEventStreamEventFlagItemRenamed     = 0x00000800
	kFSEventStreamEventFlagItemModified    = 0x00001000
	kFSEventStreamEventFlagItemIsFile      = 0x00010000
	kFSEventStreamEventFlagItemIsDir       = 0x00020000
	kFSEventStreamEventFlagItemChangeOwner = 0x00004000
	kFSEventStreamEventFlagItemXattrMod    = 0x00008000
)

type FileEvent struct {
	Timestamp time.Time
	Operation string
	Path      string
	IsDir     bool
}

var (
	ignorePrefixes = []string{
		"/private/var",
		"/private/tmp",
		"/System/",
		"/Applications/",
		"/Volumes/",
		"/dev/",
		"/usr/",
	}

	recentEvents = make(map[string]time.Time)
	homeDir      string
	running      = true
)

func main() {
	currentUser, _ := user.Current()
	username := "unknown"
	homeDir = os.Getenv("HOME")

	if currentUser != nil {
		username = currentUser.Username
		if currentUser.HomeDir != "" {
			homeDir = currentUser.HomeDir
		}
	}

	fmt.Printf("🔍 FSEvent 文件监控启动... (用户: %s)\n", username)
	fmt.Println("📋 监控目录: ~/Desktop, ~/Documents, ~/Downloads")
	fmt.Println("💡 FSEvent 特点: 高效、低延迟、macOS 原生支持")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 监控的目录
	watchPaths := []string{
		filepath.Join(homeDir, "Desktop"),
		filepath.Join(homeDir, "Documents"),
		filepath.Join(homeDir, "Downloads"),
	}

	// 创建 CFArray
	pathsToWatch := createCFArrayFromPaths(watchPaths)
	defer C.CFRelease(C.CFTypeRef(pathsToWatch))

	// 创建事件流
	stream := C.createEventStream(pathsToWatch)
	if stream == nil {
		fmt.Println("❌ 无法创建 FSEvent 流")
		os.Exit(1)
	}

	fmt.Println("✅ 监控已启动\n")

	// 捕获退出信号
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// 在后台协程中启动监控
	go func() {
		C.startEventStream(stream)
	}()

	// 等待退出信号
	<-sigChan
	fmt.Println("\n\n🛑 停止监控...")
	running = false
	C.stopEventStream(stream)
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

	// 清理旧事件
	cleanupOldEvents()
}

// 处理单个事件
func handleEvent(path string, flags uint32) {
	// 过滤系统路径
	if shouldIgnorePath(path) {
		return
	}

	// 去重: 避免短时间内重复报告
	if lastTime, exists := recentEvents[path]; exists {
		if time.Since(lastTime) < 500*time.Millisecond {
			return
		}
	}
	recentEvents[path] = time.Now()

	// 判断操作类型
	operation := ""
	emoji := ""
	isDir := (flags & kFSEventStreamEventFlagItemIsDir) != 0

	if (flags & kFSEventStreamEventFlagItemCreated) != 0 {
		if isDir {
			operation = "New Folder"
			emoji = "📁"
		} else {
			operation = "Created"
			emoji = "✨"
		}
	} else if (flags & kFSEventStreamEventFlagItemRemoved) != 0 {
		if strings.Contains(path, ".Trash") {
			operation = "Moved to Trash"
			emoji = "🗑️"
		} else {
			operation = "Deleted"
			emoji = "❌"
		}
	} else if (flags & kFSEventStreamEventFlagItemRenamed) != 0 {
		if strings.Contains(path, ".Trash") {
			operation = "Moved to Trash"
			emoji = "🗑️"
		} else {
			operation = "Renamed/Moved"
			emoji = "📦"
		}
	} else if (flags & kFSEventStreamEventFlagItemModified) != 0 {
		operation = "Modified"
		emoji = "✏️"
	} else {
		return // 忽略其他类型的事件
	}

	// 打印事件
	printEvent(path, operation, emoji)
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
	if strings.Contains(path, "/Library/") {
		return true
	}

	// 隐藏文件
	baseName := filepath.Base(path)
	if strings.HasPrefix(baseName, ".") {
		return true
	}

	// 路径中包含隐藏目录
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

// 打印事件
func printEvent(path, operation, emoji string) {
	// 简化路径显示
	displayPath := path
	if strings.HasPrefix(displayPath, homeDir) {
		displayPath = "~" + strings.TrimPrefix(displayPath, homeDir)
	}

	// 颜色设置
	colorReset := "\033[0m"
	colorTime := "\033[0;90m"
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
	fmt.Printf("%s[%s]%s %s %s%s%s %s%s%s\n",
		colorTime, timestamp, colorReset,
		emoji,
		colorOp, operation, colorReset,
		colorFile, displayPath, colorReset,
	)
}

// 清理旧事件记录
func cleanupOldEvents() {
	if len(recentEvents) < 1000 {
		return
	}

	now := time.Now()
	for k, v := range recentEvents {
		if now.Sub(v) > 10*time.Second {
			delete(recentEvents, k)
		}
	}
}