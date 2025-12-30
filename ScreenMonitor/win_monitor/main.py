# -*- coding: utf-8 -*-
"""
File monitoring system main entry point
"""
import time
import os
import sys
import socket
import traceback
from watchdog.observers import Observer

from core.platform import get_platform
from core.config import ConfigManager
from core.utils import get_drives, check_admin_rights, get_current_user
from core.stats import StatisticsCollector
from core.trackers import ProcessTracker, WindowSpy
from core.log_manager import ErrorLogger, LogEngine
from core.processor import EventBatchProcessor
from core.monitor import MonitorHandler
from core.upload_detector import UploadDetector
from core.browser_file_monitor import get_browser_file_monitor


def main():
    print("\n" + "=" * 100)
    print(" [MONITOR] File Monitoring System v3.0")
    print("=" * 100 + "\n")

    platform = get_platform()

    if not platform.is_admin():
        print("[WARNING] Not running as administrator/root")
        print("          Some system directories may not be monitored\n")

    config_mgr = ConfigManager()
    
    error_logger = ErrorLogger(config_mgr)
    stats_collector = StatisticsCollector(config_mgr)
    logger_engine = LogEngine(config_mgr)
    batch_processor = EventBatchProcessor(config_mgr, logger_engine, stats_collector)
    window_spy = WindowSpy(stats_collector)
    process_tracker = ProcessTracker(config_mgr)
    
    upload_detector = None
    if config_mgr.get("upload_detection.enabled", True):
        upload_detector = UploadDetector(config_mgr, stats_collector)
        print("[UPLOAD] Upload detection: Enabled")
    else:
        print("[UPLOAD] Upload detection: Disabled")
    
    # Check admin rights
    if platform.is_admin():
        print("✅ [ADMIN] Running with elevated privileges")
    else:
        print("⚠️  [ADMIN] Running with standard privileges")
        print("⚠️  [ADMIN] Some monitoring features might be restricted")
    
    
    try:
        browser_monitor = get_browser_file_monitor(
            config=config_mgr,
            stats=stats_collector,
            event_callback=batch_processor.add_event,
            error_logger=error_logger
        )
        print("[BROWSER_MONITOR] Browser file access monitoring: Started")
    except Exception as e:
        print(f"[BROWSER_MONITOR] Failed to start: {e}")
        browser_monitor = None

    event_handler = MonitorHandler(
        config_mgr,
        batch_processor,
        window_spy,
        process_tracker,
        error_logger,
        stats_collector,
        upload_detector
    )
    
    # Initialize File Dialog Detector via Platform
    if platform.start_file_dialog_monitor(batch_processor.add_event):
        print("[FILE_DIALOG] File dialog detection: Enabled")
    else:
        print("[FILE_DIALOG] File dialog detection: Disabled/Not Supported")

    observer = Observer()

    drives = get_drives()
    print(f"[DRIVES] Detected roots/drives: {', '.join(drives)}")
    print(f"[USER] Current user: {get_current_user()}")
    print(f"[HOST] Hostname: {socket.gethostname()}")
    print(f"[LOGS] Log directory: {os.path.abspath(logger_engine.log_dir)}")
    print(f"[ERROR] Error log: {os.path.abspath(error_logger.error_log_file)}")
    print()

    print("[CONFIG] Configuration:")
    print(f"   Batch processing: {'Enabled' if config_mgr.get('monitor_settings.batch_processing_enabled') else 'Disabled'}")
    print(f"   Batch size: {config_mgr.get('monitor_settings.batch_size')}")
    print(f"   Debounce time: {config_mgr.get('monitor_settings.debounce_seconds')}s")
    print(f"   Same file cooldown: {config_mgr.get('monitor_settings.same_file_cooldown')}s")
    print(f"   Full paths: {'Yes' if config_mgr.get('log_enrichment.show_full_paths') else 'No'}")
    print(f"   Statistics: {'Enabled' if config_mgr.get('advanced.enable_statistics') else 'Disabled'}")
    print()

    excluded_paths_count = len(config_mgr.get('filters.excluded_paths', []))
    excluded_files_count = len(config_mgr.get('filters.excluded_filenames', []))
    excluded_exts_count = len(config_mgr.get('filters.excluded_extensions', []))
    included_exts_count = len(config_mgr.get('filters.included_extensions', []))

    print("[FILTERS] Filter rules:")
    print(f"   Excluded paths: {excluded_paths_count}")
    print(f"   Excluded files: {excluded_files_count}")
    print(f"   Excluded extensions: {excluded_exts_count}")
    print(f"   Included extensions: {included_exts_count}")
    print()

    for drive in drives:
        try:
            observer.schedule(event_handler, drive, recursive=True)
            print(f"[OK] Monitoring: {drive}")
        except Exception as e:
            print(f"[ERROR] Cannot monitor {drive}: {e}")
            error_logger.log_error("drive_monitor", f"Cannot monitor {drive}", e)

    print("\n" + "=" * 100)
    print("[RUNNING] System running... (Press Ctrl+C to stop)")
    print("=" * 100 + "\n")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping system...")
        observer.stop()
        batch_processor.shutdown()

        if stats_collector.enabled:
            print("\n[STATISTICS] Final statistics:")
            print(f"   Total events: {stats_collector.total_events}")
            print(f"   Runtime: {(time.time() - stats_collector.start_time) / 60:.1f} minutes")

            error_stats = error_logger.get_statistics()
            if error_stats:
                print("\n[ERRORS] Error statistics:")
                for error_type, count in error_stats.items():
                    print(f"   {error_type}: {count}")

    observer.join()
    print("\n[STOPPED] System stopped\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] System error: {e}")
        traceback.print_exc()
        sys.exit(1)