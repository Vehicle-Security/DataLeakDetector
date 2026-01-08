# -*- coding: utf-8 -*-
import time
import threading
import queue

class EventBatchProcessor:
    """事件批处理器"""

    def __init__(self, config, logger, stats_collector):
        self.enabled = config.get("monitor_settings.batch_processing_enabled", True)
        self.batch_size = config.get("monitor_settings.batch_size", 50)
        self.batch_interval = config.get("monitor_settings.batch_interval_ms", 1000) / 1000.0
        
        self.logger = logger
        self.stats = stats_collector
        self.event_queue = queue.Queue()
        self.running = True
        
        if self.enabled:
            self._start_processor()

    def add_event(self, event_data):
        """添加事件到队列"""
        if self.enabled:
            self.event_queue.put(event_data)
        else:
            # 如果未启用批处理，直接写入
            self.logger.write(event_data)

    def _start_processor(self):
        """启动批处理线程"""
        def processor():
            batch = []
            last_flush = time.time()
            
            while self.running:
                try:
                    # 尝试获取事件，带超时
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        batch.append(event)
                    except queue.Empty:
                        pass

                    now = time.time()
                    
                    # 检查是否应该刷新批次
                    should_flush = (
                        len(batch) >= self.batch_size or
                        (batch and now - last_flush >= self.batch_interval)
                    )
                    
                    if should_flush:
                        self._flush_batch(batch)
                        batch = []
                        last_flush = now
                        
                except Exception as e:
                    print(f"[ERROR] 批处理错误: {e}")

        thread = threading.Thread(target=processor, daemon=True)
        thread.start()

    def _flush_batch(self, batch):
        """刷新批次到日志"""
        if not batch:
            return
            
        for event in batch:
            self.logger.write(event)

    def shutdown(self):
        """关闭批处理器"""
        self.running = False
