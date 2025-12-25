# -*- coding: utf-8 -*-
"""
浏览器文件访问监控器
监控浏览器进程打开的所有文件句柄，捕捉从任意位置上传的文件

使用psutil监控进程的文件句柄，检测浏览器读取的文件
"""
import os
import time
import threading
import psutil
from datetime import datetime
from collections import defaultdict


class BrowserFileMonitor:
    """监控浏览器进程访问的文件"""
    
    def __init__(self, config=None, stats=None, event_callback=None, error_logger=None):
        """初始化
        
        Args:
            config: 配置对象
            stats: 统计收集器
            event_callback: 事件回调函数(用于将事件发送到日志系统)
            error_logger: 错误日志记录器
        """
        self.config = config or {}
        self.stats = stats
        self.event_callback = event_callback
        self.error_logger = error_logger
        
        # Browser and blacklisted app processes
        self.browser_processes = [
            # Browsers
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
            "brave.exe",
            "opera.exe",
            # IM Apps
            "QQ.exe",
            "QQScLauncher.exe",
            "WeChat.exe",
            "Weixin.exe",
            "dingtalk.exe",
            # Cloud Storage
            "quark.exe",
            "Quark.exe",
            # Collaboration
            "Feishu.exe",
            "TencentMeeting.exe",
            "Meeting.exe",
            # AI Apps
            "doubao.exe",
            "Doubao.exe",
            "yuanbao.exe",
            "Chatbox.exe"
        ]
        
        # Monitor ALL file extensions (remove whitelist for DLP)
        self.monitored_extensions = None  # None = monitor all
        
        # 文件访问记录
        self.file_accesses = []
        self.file_lock = threading.Lock()
        self.max_history = 200
        
        # 已知的文件（避免重复记录）
        self.known_files = defaultdict(float)  # file_path: last_seen_time
        self.cleanup_interval = 60  # 清理间隔（秒）
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval=0.5):
        """
        开始监控
        
        Args:
            interval: 检查间隔（秒）
        """
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("=" * 80)
        print("[BROWSER_MONITOR] 浏览器文件访问监控已启动")
        print(f"[BROWSER_MONITOR] 轮询间隔: {interval}秒")
        print(f"[BROWSER_MONITOR] 监控进程: {', '.join(self.browser_processes)}")
        print(f"[BROWSER_MONITOR] 文件扩展名过滤: {'全部' if self.monitored_extensions is None else '已启用'}")
        print("=" * 80)
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        print("[BROWSER_MONITOR] 浏览器文件访问监控已停止")
    
    def _monitor_loop(self, interval):
        """监控循环"""
        last_cleanup = time.time()
        
        while self.is_monitoring:
            try:
                self._check_browser_files()
                
                # 定期清理旧记录
                if time.time() - last_cleanup > self.cleanup_interval:
                    self._cleanup_old_records()
                    last_cleanup = time.time()
                    
            except Exception:
                pass  # 静默处理错误
            
            time.sleep(interval)
    
    def _check_browser_files(self):
        """检查浏览器进程打开的文件"""
        found_processes = []
        try:
            # 遍历所有进程
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name']
                    if not proc_name or proc_name.lower() not in [p.lower() for p in self.browser_processes]:
                        continue
                    
                    found_processes.append(proc_name)
                    
                    # 获取进程对象
                    process = psutil.Process(proc.info['pid'])
                    
                    # 获取进程打开的文件
                    try:
                        open_files = process.open_files()
                    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
                        # 首次遇到权限拒绝时输出警告
                        if not hasattr(self, '_access_denied_warned'):
                            print(f"[BROWSER_MONITOR] ⚠️  无法访问进程 {proc_name} (PID: {proc.info['pid']}): {e}")
                            print(f"[BROWSER_MONITOR] ⚠️  请确保以管理员权限运行！")
                            self._access_denied_warned = True
                        continue
                    
                    # 检查每个打开的文件
                    for file_info in open_files:
                        file_path = file_info.path
                        
                        # 检查是否是我们关心的文件
                        if not self._is_interesting_file(file_path):
                            continue
                        
                        # 检查是否是新发现的文件或很久没见过的文件
                        now = time.time()
                        last_seen = self.known_files.get(file_path, 0)
                        
                        if now - last_seen > 5:  # 5秒内不重复记录
                            self._record_file_access(file_path, process, proc_name)
                            self.known_files[file_path] = now
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 定期输出监控状态
            if not hasattr(self, '_last_status_time'):
                self._last_status_time = 0
            now = time.time()
            if now - self._last_status_time > 30:  # 每30秒输出一次状态
                if found_processes:
                    print(f"[BROWSER_MONITOR] 监控中... 发现进程: {', '.join(set(found_processes))}")
                else:
                    print(f"[BROWSER_MONITOR] 监控中... 未发现目标进程")
                self._last_status_time = now
        
        except Exception as e:
            print(f"[BROWSER_MONITOR] 错误: {e}")
    
    
    def _is_interesting_file(self, file_path):
        """判断文件是否值得关注"""
        try:
            # 排除不存在的文件
            if not os.path.isfile(file_path):
                return False
            
            # Check extension (None = monitor all files for DLP)
            _, ext = os.path.splitext(file_path)
            if self.monitored_extensions is not None:
                if ext.lower() not in self.monitored_extensions:
                    return False
            
            # CRITICAL: 完全过滤 AppData\Local\Temp 目录下的所有 .tmp 文件
            # 这些是浏览器上传时的临时缓存,不是用户选择的原始文件
            if ext.lower() in ['.tmp', '.temp', '.crdownload', '.part']:
                if '\\AppData\\Local\\Temp\\' in file_path:
                    return False
                # 也过滤系统临时目录
                if '\\Windows\\Temp\\' in file_path or '\\Temp\\' in file_path:
                    return False
            
            # 排除系统文件扩展名（字体、语言包、系统数据）
            system_extensions = [
                '.ttf', '.ttc', '.otf',  # 字体文件
                '.mui', '.mun',          # 语言包/资源文件
                '.sdb', '.asar',         # 系统数据库/应用包
            ]
            if ext.lower() in system_extensions:
                return False
            
            # 排除Windows系统目录
            system_dirs = [
                '\\Windows\\Fonts\\',
                '\\Windows\\apppatch\\',
                '\\Windows\\SystemResources\\',
                '\\WindowsApps\\',
                '\\Program Files (x86)\\Microsoft\\Edge\\',
                '\\Program Files\\Microsoft\\Edge\\',
                '\\Program Files\\Tencent\\QQNT\\',
            ]
            for system_dir in system_dirs:
                if system_dir in file_path:
                    return False
            
            # 排除浏览器内部文件路径
            browser_internal_paths = [
                "\\AppData\\Local\\Microsoft\\Edge\\User Data\\",
                "\\AppData\\Local\\Google\\Chrome\\User Data\\",
                "\\AppData\\Roaming\\QQ\\Partitions\\",
                "\\AppData\\Roaming\\QQ\\Local Storage\\",
                "\\AppData\\Roaming\\QQ\\Session Storage\\",
                "\\AppData\\Roaming\\QQ\\IndexedDB\\",
                "\\AppData\\Roaming\\QQ\\Network\\",
                "\\AppData\\Roaming\\QQ\\Shared Dictionary\\",
                "\\AppData\\Roaming\\QQ\\DawnWebGPUCache\\",
                "\\AppData\\Roaming\\QQ\\DawnGraphiteCache\\",
                "\\AppData\\Roaming\\QQ\\Dictionaries\\",
                "\\AppData\\Roaming\\QQ\\Cache\\",
                "\\AppData\\Roaming\\QQ\\SharedStorage",
                "\\AppData\\LocalLow\\Intel\\ShaderCache\\",
                "\\ProgramData\\kingsoft\\office6\\mtfont\\",
            ]
            
            for pattern in browser_internal_paths:
                if pattern in file_path:
                    return False
            
            # 排除QQ/微信数据库文件
            db_patterns = [
                "\\Tencent Files\\",  # QQ数据库和缓存
            ]
            for pattern in db_patterns:
                if pattern in file_path:
                    # 但如果是用户主动上传的文件（常见文档格式）则保留
                    user_doc_extensions = ['.doc', '.docx', '.pdf', '.xlsx', '.xls', '.ppt', '.pptx', '.txt', '.zip', '.rar', '.7z', '.jpg', '.png', '.mp4', '.avi']
                    if ext.lower() in user_doc_extensions:
                        return True  # 用户文档，即使在Tencent Files也保留
                    return False
            
            # 排除浏览器内部文件扩展名
            browser_internal_extensions = [
                '.pak', '.bin', '.dat', '.bdic', '.ldb', '.log', 
                '.pma', '.db-wal', '.db-shm', '.db-journal',
                '.crc', '.mmap3', '.store'
            ]
            if ext.lower() in browser_internal_extensions:
                return False
            
            # Exclude specific filenames
            filename = os.path.basename(file_path).lower()
            exclude_filenames = [
                'lock', 'manifest-', 'quotamanager', 'cookies', 'history',
                'login data', 'web data', 'shortcuts', 'favicons',
                'trust tokens', 'reporting and nel', 'safe browsing',
                'extensionactivity', 'networkactionpredictor', 'sharedstorage',
                'iconcache'  # Windows Explorer icon cache files
            ]
            for pattern in exclude_filenames:
                if pattern in filename:
                    return False
            
            # Exclude browser cache and system files
            exclude_paths = [
                "\\GPUCache\\",
                "\\Code Cache\\",
                "\\Service Worker\\",
                "\\DawnCache\\",
                "scoped_dir"
            ]
            
            for pattern in exclude_paths:
                if pattern in file_path:
                    return False
            
            return True
        
        except Exception:
            return False
    
    def get_app_name_from_window(self, window_title):
        """从窗口标题提取应用名称"""
        if not window_title:
            return None
        
        # 应用识别规则（按优先级排序） - Enterprise DLP Coverage
        app_patterns = [
            # AI Assistants
            ('豆包', ['豆包', 'doubao']),
            ('KIMI', ['kimi', '月之暗面', 'moonshot']),
            ('ChatGPT', ['chatgpt', 'openai']),
            ('Claude', ['claude', 'anthropic']),
            ('Gemini', ['gemini', 'bard', 'google ai']),
            ('通义千问', ['通义', 'qwen', 'tongyi']),
            ('文心一言', ['文心', 'ernie', 'wenxin']),
            ('智谱清言', ['智谱', 'chatglm', 'zhipu']),
            ('讯飞星火', ['星火', 'spark', 'xunfei']),
            ('腾讯混元', ['混元', 'hunyuan']),
            # Cloud Storage & File Sharing
            ('百度网盘', ['百度网盘', 'baiduyun', 'baidu pan']),
            ('阿里云盘', ['阿里云盘', 'aliyun', 'alipan']),
            ('腾讯微云', ['微云', 'weiyun']),
            ('OneDrive', ['onedrive', 'microsoft drive']),
            ('Dropbox', ['dropbox']),
            ('Google Drive', ['google drive', 'drive.google']),
            # Communication Platforms
            ('QQ', ['qq', '腾训qq']),
            ('微信', ['微信', 'wechat', 'weixin']),
            ('钉钉', ['钉钉', 'dingtalk']),
            ('飞书', ['飞书', 'feishu', 'lark']),
            ('企业微信', ['企业微信', 'wework']),
        ]
        
        window_lower = window_title.lower()
        for app_name, keywords in app_patterns:
            for keyword in keywords:
                if keyword.lower() in window_lower:
                    return app_name
        
        return None
    
    def get_app_name_from_process(self, proc_name):
        """从进程名提取应用名称"""
        if not proc_name:
            return None
        
        proc_lower = proc_name.lower()
        
        # 进程名映射规则
        process_map = {
            'qq.exe': 'QQ',
            'qqscl auncher.exe': 'QQ',
            'wechat.exe': '微信',
            'weixin.exe': '微信',
            'dingtalk.exe': '钉钉',
            'feishu.exe': '飞书',
            'lark.exe': '飞书',
            'msedge.exe': 'Edge浏览器',
            'chrome.exe': 'Chrome浏览器',
            'firefox.exe': 'Firefox浏览器',
        }
        
        return process_map.get(proc_lower, None)
    
    def _validate_event(self, event):
        """验证事件结构完整性 - 防止日志丢失"""
        required_fields = ['timestamp', 'event_type', 'file_path', 
                          'process_info', 'window_info', 'user_info', 'disk_info']
        for field in required_fields:
            if field not in event:
                print(f"[BROWSER_MONITOR] ⚠️  事件缺少必需字段: {field}")
                return False
        return True
    
    def _write_to_fallback_log(self, event):
        """写入fallback日志 - 当主日志失败时使用"""
        try:
            import json
            fallback_file = "d:\\code\\win_monitor\\logs\\fallback_browser_events.json"
            with open(fallback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
            print(f"[BROWSER_MONITOR] 💾 事件已写入fallback日志")
        except Exception as e:
            print(f"[BROWSER_MONITOR] ❌ Fallback日志也失败: {e}")
    
    def _record_file_access(self, file_path, process, proc_name):
        """记录文件访问"""
        try:
            # 获取窗口标题
            window_title = ""
            try:
                import win32gui
                import win32process
                
                def enum_windows_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == process.pid:
                            title = win32gui.GetWindowText(hwnd)
                            if title:
                                results.append(title)
                
                titles = []
                win32gui.EnumWindows(enum_windows_callback, titles)
                if titles:
                    window_title = titles[0]
            except Exception:
                pass
            
            # 提取app_name
            app_name = self.get_app_name_from_window(window_title)
            if not app_name:
                app_name = self.get_app_name_from_process(proc_name)
            
            # 构建完整事件 - 匹配MonitorHandler结构
            file_info = {
                "timestamp": datetime.now().isoformat(timespec='milliseconds'),
                "event_type": "modified",  # CRITICAL: batch_processor needs this!
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "file_extension": os.path.splitext(file_path)[1],
                "process_info": {
                    "pid": process.pid,
                    "process_name": proc_name,
                    "process_path": None,
                    "cmdline": None
                },
                "window_info": {
                    "window_handle": None,
                    "window_title": window_title,
                    "window_class": None
                },
                "user_info": {
                    "username": None,  # batch_processor will fill
                    "hostname": None
                },
                "disk_info": {
                    "drive_letter": os.path.splitdrive(file_path)[0],
                    "disk_type": "Fixed"
                },
                "detection_method": "browser_file_monitor",
                "app_name": app_name
            }
            
            with self.file_lock:
                self.file_accesses.append(file_info)
                
                # 限制历史记录大小
                if len(self.file_accesses) > self.max_history:
                    self.file_accesses = self.file_accesses[-self.max_history:]
                
                print(f"[BROWSER_MONITOR] ✅ 检测到文件访问!")
                print(f"[BROWSER_MONITOR]    文件: {os.path.basename(file_path)}")
                print(f"[BROWSER_MONITOR]    路径: {file_path}")
                print(f"[BROWSER_MONITOR]    进程: {proc_name}")
                if window_title:
                    print(f"[BROWSER_MONITOR]    窗口: {window_title[:60]}")
                if app_name:
                    print(f"[BROWSER_MONITOR]    应用: {app_name}")
                
                # CRITICAL: Validate before sending - prevent log loss
                if not self._validate_event(file_info):
                    print(f"[BROWSER_MONITOR] ❌ 事件验证失败，写入fallback: {file_path}")
                    self._write_to_fallback_log(file_info)
                    return
                
                # CRITICAL: Send event to logging system via callback
                if self.event_callback:
                    try:
                        self.event_callback(file_info)
                    except Exception as e:
                        print(f"[BROWSER_MONITOR] ❌ 回调失败: {e}")
                        # Write to fallback on callback failure
                        self._write_to_fallback_log(file_info)
                        # Also log error
                        if hasattr(self, 'error_logger'):
                            self.error_logger.log_error("event_callback", 
                                f"Failed to log event: {file_path}", e)
        
        except Exception as e:
            print(f"[BROWSER_MONITOR] ❌ 记录文件访问异常: {e}")
            if hasattr(self, 'error_logger'):
                self.error_logger.log_error("record_file_access", str(e), e)
    def _cleanup_old_records(self):
        """清理旧的已知文件记录"""
        now = time.time()
        cleanup_age = 300  # 5分钟
        
        with self.file_lock:
            # 清理超过5分钟未见过的文件
            old_files = [
                path for path, last_seen in self.known_files.items()
                if now - last_seen > cleanup_age
            ]
            for path in old_files:
                del self.known_files[path]
    
    def get_recent_accesses(self, seconds=60):
        """
        获取最近访问的文件
        
        Args:
            seconds: 时间范围（秒）
            
        Returns:
            list: 文件访问记录列表
        """
        with self.file_lock:
            now = datetime.now()
            recent = []
            for file_info in reversed(self.file_accesses):
                try:
                    file_time = datetime.fromisoformat(file_info["timestamp"])
                    if (now - file_time).total_seconds() <= seconds:
                        recent.append(file_info)
                except:
                    pass
            return recent
    
    def clear_history(self):
        """清空历史记录"""
        with self.file_lock:
            self.file_accesses.clear()
            self.known_files.clear()


# 全局实例
_global_monitor = None


def get_browser_file_monitor(config=None, stats=None, event_callback=None, error_logger=None):
    """获取全局浏览器文件监控器实例
    
    Args:
        config: 配置对象
        stats: 统计收集器
        event_callback: 事件回调函数
        error_logger: 错误日志记录器
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BrowserFileMonitor(config, stats, event_callback, error_logger)
        _global_monitor.start_monitoring()
    return _global_monitor


if __name__ == "__main__":
    # 测试
    print("=== 浏览器文件访问监控器测试 ===")
    print("请在浏览器中上传文件...")
    
    monitor = BrowserFileMonitor()
    monitor.start_monitoring(interval=1.0)
    
    try:
        # 运行60秒
        for i in range(60):
            time.sleep(1)
            recent = monitor.get_recent_accesses(10)
            if recent and i % 5 == 0:
                print(f"\n最近10秒检测到 {len(recent)} 个文件访问:")
                for f in recent[-5:]:  # 显示最近5个
                    print(f"  - {f['file_name']} ({f['window_title'][:30] if f['window_title'] else 'N/A'})")
    except KeyboardInterrupt:
        print("\n测试中断")
    finally:
        monitor.stop_monitoring()
        print("测试完成")
