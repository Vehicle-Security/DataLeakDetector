use log::{error, info, warn, LevelFilter};
use macos_unifiedlogs::filesystem::LiveSystemProvider;
use macos_unifiedlogs::iterator::UnifiedLogIterator;
use macos_unifiedlogs::parser::{build_log, collect_timesync};
use macos_unifiedlogs::unified_log::{LogData, UnifiedLogData};
use macos_unifiedlogs::traits::FileProvider;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use serde::Serialize;
use regex::Regex;
use percent_encoding::percent_decode_str;

#[derive(Serialize)]
struct UnifiedLogEvent {
    timestamp: String,
    process: String,
    event_type: String,
    filepath: String,
    raw_message: String,
    subsystem: String,
    category: String,
}

/// Decode URL-encoded path (handles %20, %E4 etc.)
fn decode_file_path(encoded_path: &str) -> String {
    percent_decode_str(encoded_path)
        .decode_utf8()
        .unwrap_or_else(|_| encoded_path.into())
        .to_string()
}

fn main() {
    // Initialize logging (for debug/error, distinct from our JSON output)
    TermLogger::init(
        LevelFilter::Warn,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )
    .expect("Failed to initialize logger");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    info!("Starting Enhanced Unified Log Monitor...");
    
    // Connect to Live System
    let mut provider = LiveSystemProvider::default();
    
    let timesync_data = match collect_timesync(&provider) {
        Ok(data) => data,
        Err(e) => {
            error!("Failed to collect timesync data: {}", e);
            return;
        }
    };

    // ENHANCED REGEX: Matches file:// URLs with:
    // - Spaces (not excluded by \s)
    // - URL-encoded characters (%20, %E4%B8%AD etc.)
    // - Chinese/Unicode characters
    // - Stops at: ], ), ", ', >, newline, or end of string
    // Pattern explanation:
    //   file://  - literal prefix
    //   (        - capture group start
    //     /      - path must start with /
    //     [^\]"'>\n]+  - any chars except ], ", ', >, newline
    //   )        - capture group end
    let re_file_url = Regex::new(r#"file://(/[^\]"'>\n]+)"#).unwrap();
    
    // Secondary regex for generic path detection (e.g., in launchservices)
    let re_absolute_path = Regex::new(r#"(/Users/[^\]"'>\n:]+\.[a-zA-Z0-9]+)"#).unwrap();

    let mut oversize_strings = UnifiedLogData {
        header: Vec::new(),
        catalog_data: Vec::new(),
        oversize: Vec::new(),
    };
    
    // Main Loop
    for mut source in provider.tracev3_files() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let mut buf = Vec::new();
        if let Err(err) = source.reader().read_to_end(&mut buf) {
            error!("Failed to read tracev3 chunk: {:?}", err);
            continue;
        }

        let log_iterator = UnifiedLogIterator {
            data: buf,
            header: Vec::new(),
        };

        for mut chunk in log_iterator {
            chunk.oversize.append(&mut oversize_strings.oversize);
            
            let (results, _) = build_log(&chunk, &mut provider, &timesync_data, true);
            
            oversize_strings.oversize = chunk.oversize;

            process_log_batch(&results, &re_file_url, &re_absolute_path);
        }
    }
}

fn process_log_batch(logs: &[LogData], re_file_url: &Regex, re_absolute_path: &Regex) {
    for log in logs {
        
        // === 1. AirDrop / Sharing ===
        if log.subsystem == "com.apple.sharing" {
            let event_type = if log.message.contains("Transfer") { 
                "Transfer" 
            } else if log.message.contains("send") || log.message.contains("Send") {
                "Send"
            } else { 
                "Activity" 
            };
            
            // Try to extract file path from message
            let path = extract_path(&log.message, re_file_url, re_absolute_path);
            emit_json(log, "AirDrop/Sharing", event_type, &path);
            continue;
        }

        // === 2. File Selection Dialog (Powerbox / AppKit) ===
        if log.subsystem == "com.apple.AppKit" || log.subsystem == "com.apple.Powerbox" {
            if log.message.contains("NSOpenPanel") || log.message.contains("NSSavePanel") {
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                emit_json(log, "FileSelection", "Dialog", &path);
                continue;
            }
            
            // XPC document access (macOS 12+)
            if log.message.contains("openDocument") || log.message.contains("documentManager") {
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                emit_json(log, "FileSelection", "DocumentAccess", &path);
                continue;
            }
            
            // Generic file:// URL in AppKit
            if let Some(path) = extract_file_url(&log.message, re_file_url) {
                emit_json(log, "FileSelection", "Selected", &path);
                continue;
            }
        }

        // === 3. LaunchServices - Application opening files ===
        if log.subsystem == "com.apple.launchservices" {
            // Detect when an app opens a file
            if log.message.contains("LSOpenURL") || 
               log.message.contains("openURL") || 
               log.message.contains("Opening") {
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                if !path.is_empty() {
                    emit_json(log, "LaunchServices", "FileOpen", &path);
                    continue;
                }
            }
            
            // Detect file type handler registration/lookup
            if log.message.contains("handler") && log.message.contains("file") {
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                emit_json(log, "LaunchServices", "HandlerLookup", &path);
                continue;
            }
        }

        // === 4. Network Upload Detection (nsurlsessiond) ===
        if log.subsystem == "com.apple.nsurlsessiond" || 
           log.subsystem == "com.apple.CFNetwork" ||
           log.subsystem == "com.apple.network" {
            // Detect upload operations
            if log.message.contains("upload") || 
               log.message.contains("Upload") ||
               log.message.contains("POST") ||
               log.message.contains("multipart") {
                
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                emit_json(log, "NetworkUpload", "Detected", &path);
                continue;
            }
            
            // Detect file being sent over network
            if log.message.contains("sendBody") || log.message.contains("bodyStream") {
                emit_json(log, "NetworkUpload", "BodyStream", "");
                continue;
            }
        }

        // === 5. Sandbox file access (for sandboxed apps only) ===
        if log.subsystem == "com.apple.sandbox" || log.subsystem == "com.apple.security.sandbox" {
            if log.message.contains("file-read") || log.message.contains("file-write") {
                let path = extract_path(&log.message, re_file_url, re_absolute_path);
                let action = if log.message.contains("file-write") { "Write" } else { "Read" };
                emit_json(log, "Sandbox", action, &path);
                continue;
            }
        }
    }
}

/// Extract file path from message, trying multiple methods
fn extract_path(message: &str, re_file_url: &Regex, re_absolute_path: &Regex) -> String {
    // First try file:// URL
    if let Some(path) = extract_file_url(message, re_file_url) {
        return path;
    }
    
    // Then try absolute path pattern
    if let Some(caps) = re_absolute_path.captures(message) {
        if let Some(path_match) = caps.get(1) {
            return decode_file_path(path_match.as_str().trim());
        }
    }
    
    String::new()
}

/// Extract and decode file:// URL from message
fn extract_file_url(message: &str, re_file_url: &Regex) -> Option<String> {
    if let Some(caps) = re_file_url.captures(message) {
        if let Some(path_match) = caps.get(1) {
            let raw_path = path_match.as_str().trim();
            // Decode URL encoding (%20 -> space, %E4%B8%AD -> 中 etc.)
            let decoded = decode_file_path(raw_path);
            return Some(decoded);
        }
    }
    None
}

fn emit_json(log: &LogData, event_category: &str, event_type: &str, extracted_path: &str) {
    let event = UnifiedLogEvent {
        timestamp: format!("{}", log.time),
        process: log.process.clone(),
        event_type: format!("{} - {}", event_category, event_type),
        filepath: extracted_path.to_string(),
        raw_message: log.message.clone(),
        subsystem: log.subsystem.clone(),
        category: log.category.clone(),
    };

    if let Ok(json_str) = serde_json::to_string(&event) {
        println!("{}", json_str);
    }
}
