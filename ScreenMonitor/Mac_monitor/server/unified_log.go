package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// UnifiedLogMonitor manages the external Rust binary for Unified Logs
type UnifiedLogMonitor struct {
	cmd     *exec.Cmd
	running bool
	mutex   sync.Mutex
	events  []LogEntry   // We reuse the main LogEntry struct
	logFile *FileMonitor // Optional: Link to main FileMonitor if we want to merge streams directly

	// Configuration
	binaryPath string
}

// UnifiedLogEvent match the JSON output from Rust
type UnifiedLogEvent struct {
	Timestamp  string `json:"timestamp"`
	Process    string `json:"process"`
	EventType  string `json:"event_type"`
	Filepath   string `json:"filepath"`
	RawMessage string `json:"raw_message"`
	Subsystem  string `json:"subsystem"`
	Category   string `json:"category"`
}

func NewUnifiedLogMonitor(binaryPath string) *UnifiedLogMonitor {
	return &UnifiedLogMonitor{
		binaryPath: binaryPath,
	}
}

func (ul *UnifiedLogMonitor) Start() error {
	ul.mutex.Lock()
	defer ul.mutex.Unlock()

	if ul.running {
		return fmt.Errorf("UnifiedLogMonitor already running")
	}

	// Start the Rust binary
	// Note: sudo might be needed for full log access, similar to fs_usage
	// For now we assume the main process has permissions or user runs with sudo
	ul.cmd = exec.Command(ul.binaryPath)

	stdout, err := ul.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}

	stderr, err := ul.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	if err := ul.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start unified_log_monitor: %v", err)
	}

	ul.running = true
	log.Printf("🚀 Unified Log Monitor started (pid: %d)", ul.cmd.Process.Pid)

	// Stream Processor
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			ul.processLine(line)
		}

		// If scanner ends, process likely died
		ul.mutex.Lock()
		ul.running = false
		ul.mutex.Unlock()
	}()

	// Error Logger
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("[UnifiedLog-StdErr] %s", scanner.Text())
		}
	}()

	return nil
}

func (ul *UnifiedLogMonitor) Stop() error {
	ul.mutex.Lock()
	defer ul.mutex.Unlock()

	if !ul.running || ul.cmd == nil {
		return nil
	}

	if ul.cmd.Process != nil {
		ul.cmd.Process.Signal(syscall.SIGTERM)
		time.Sleep(200 * time.Millisecond)
		ul.cmd.Process.Kill()
	}

	ul.running = false
	log.Println("🛑 Unified Log Monitor stopped")
	return nil
}

func (ul *UnifiedLogMonitor) processLine(line string) {
	var event UnifiedLogEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		log.Printf("Failed to unmarshal unified log: %v | Line: %s", err, line)
		return
	}

	// Convert to internal LogEntry format
	// This mapping is crucial. We translate "FileSelection" to "IsUpload=true" intent

	logEntry := LogEntry{
		Timestamp: time.Now().Format("2006-01-02T15:04:05.000"), // Re-stamp or use event.Timestamp
		EventType: "unified_log_event",
		FilePath:  event.Filepath,
		FileName:  event.Filepath,
		AppName:   event.Process,
		Extra: map[string]interface{}{
			"raw_message":  event.RawMessage,
			"subsystem":    event.Subsystem,
			"unified_type": event.EventType,
		},
	}

	// Special Handling for known types
	if event.EventType == "FileSelection - Selected" || event.EventType == "FileSelection - Dialog" {
		logEntry.UploadInfo = &UploadDetection{
			IsUpload:     true,
			AppName:      event.Process,
			UploadType:   "Dialog Selection/Upload",
			OriginalFile: event.Filepath,
		}
		log.Printf("🚨 DETECTED UPLOAD INTENT via File Dialog: %s -> %s", event.Process, event.Filepath)
	}

	// Note: For full event aggregation, consider creating a channel-based architecture
	// Track enhancement: Create GitHub Issue for "Unified Log event aggregation channel"
	// Current: Events are logged via Printf, which is sufficient for monitoring
	// Future: Could send to main event stream for unified dashboard display
}
