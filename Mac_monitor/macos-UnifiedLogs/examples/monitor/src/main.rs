use log::{error, info, LevelFilter};
use macos_unifiedlogs::filesystem::LiveSystemProvider;
use macos_unifiedlogs::iterator::UnifiedLogIterator;
use macos_unifiedlogs::parser::{build_log, collect_timesync};
use macos_unifiedlogs::unified_log::{LogData, UnifiedLogData};
use macos_unifiedlogs::traits::FileProvider;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use serde::Serialize;
use regex::Regex;

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

    info!("Starting Unified Log Monitor...");
    
    // Connect to Live System
    let mut provider = LiveSystemProvider::default();
    
    // In a real robust implementation we might need to handle timesync updates, 
    // but for now we grab it once at start.
    let timesync_data = match collect_timesync(&provider) {
        Ok(data) => data,
        Err(e) => {
            error!("Failed to collect timesync data: {}", e);
            return;
        }
    };

    // Pre-compile Regex
    // Matches: "navigating to [file:///path/to/file]" or similar common NSOpenPanel logs
    // Note: The exact message format varies by OS version, so we look for "Powerbox" signals primarily.
    let re_file_url = Regex::new(r"file://(/[^\]\s\)]+)").unwrap();

    let mut oversize_strings = UnifiedLogData {
        header: Vec::new(),
        catalog_data: Vec::new(),
        oversize: Vec::new(),
    };
    
    // Main Loop
    // LiveSystemProvider.tracev3_files() returns an iterator that blocks/waits for new log chunks
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
            header: Vec::new(), // Header is updated within iterator
        };

        for mut chunk in log_iterator {
            // Handle oversize strings (large log messages)
            chunk.oversize.append(&mut oversize_strings.oversize);
            
            // Build logs
            // We set include_missing=true because in live mode we want to see what we have now
            let (results, _) = build_log(&chunk, &mut provider, &timesync_data, true);
            
            // Cache oversize for next iteration
            oversize_strings.oversize = chunk.oversize;

            process_log_batch(&results, &re_file_url);
        }
    }
}

fn process_log_batch(logs: &[LogData], re_file: &Regex) {
    for log in logs {
        // Filter Strategy
        // 1. AirDrop / Sharing
        if log.subsystem == "com.apple.sharing" {
            // Simplified detection logic
            let event_type = if log.message.contains("Transfer") { "Transfer" } else { "Activity" };
            emit_json(log, "AirDrop/Sharing", event_type, "");
            continue;
        }

        // 2. File Selection (Powerbox / AppKit)
        // Subsystem: com.apple.AppKit
        // Category: FileService? OpenPanel?
        // We look for specific keywords in AppKit or Security logs
        if log.subsystem == "com.apple.AppKit" || log.subsystem == "com.apple.Powerbox" {
             // Look for NSOpenPanel or save panel interactions
             if log.message.contains("NSOpenPanel") || log.message.contains("NSSavePanel") {
                 emit_json(log, "FileSelection", "Dialog", "");
             } else if let Some(caps) = re_file.captures(&log.message) {
                 // If we catch a file URL in AppKit, it's likely a user selection
                 if let Some(path_match) = caps.get(1) {
                     emit_json(log, "FileSelection", "Selected", path_match.as_str());
                 }
             }
        }
    }
}

fn emit_json(log: &LogData, event_category: &str, event_type: &str, extracted_path: &str) {
    let event = UnifiedLogEvent {
        timestamp: format!("{}", log.time), // Raw absolute time, Go side can parse
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
