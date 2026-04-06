# -*- coding: utf-8 -*-
import time
import threading
import psutil
from collections import defaultdict
from . import app_logger

class StatisticsCollector:
    """统计信息收集器"""

    def __init__(self, config):
        self.enabled = config.get("advanced.enable_statistics", True)
        self.report_interval = config.get("advanced.statistics_report_interval", 300)
        
        self.event_counts = defaultdict(int)
        self.total_events = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.enabled:
            self._start_reporter()

    def record_event(self, event_type):
        """记录事件"""
        if self.enabled:
            self.event_counts[event_type] += 1
            self.total_events += 1

    def record_cache_hit(self):
        """记录缓存命中"""
        if self.enabled:
            self.cache_hits += 1

    def record_cache_miss(self):
        """记录缓存未命中"""
        if self.enabled:
            self.cache_misses += 1

    def _start_reporter(self):
        """启动统计报告线程"""
        def reporter():
            while True:
                time.sleep(self.report_interval)
                self._print_statistics()

        thread = threading.Thread(target=reporter, daemon=True, name="StatsReporter")
        thread.start()

    def _print_statistics(self):
        """打印统计信息"""
        now = time.time()
        elapsed = now - self.start_time
        
        app_logger.info("\n" + "="*80)
        app_logger.info("[STATISTICS] 运行统计信息")
        app_logger.info("="*80)
        app_logger.info(f"运行时间: {elapsed/60:.1f} 分钟")
        app_logger.info(f"总事件数: {self.total_events}")
        app_logger.info(f"事件速率: {self.total_events/elapsed:.2f} 事件/秒")
        
        if self.event_counts:
            app_logger.info("\n事件类型分布:")
            for event_type, count in sorted(self.event_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_events * 100) if self.total_events > 0 else 0
                app_logger.info(f"  {event_type:12s}: {count:6d} ({percentage:5.1f}%)")
        
        total_cache = self.cache_hits + self.cache_misses
        if total_cache > 0:
            hit_rate = (self.cache_hits / total_cache * 100)
            app_logger.info(f"\n缓存命中率: {hit_rate:.1f}% ({self.cache_hits}/{total_cache})")
        
        # 内存使用
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        app_logger.info(f"内存使用: {mem_mb:.1f} MB")
        
        app_logger.info("="*80 + "\n")
        
        self.last_report_time = now
