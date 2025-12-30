# -*- coding: utf-8 -*-
"""
上传检测器 - 检测文件上传操作
通过监控应用程序临时目录和文件对话框来识别上传行为
"""
import os
import time
import threading
from datetime import datetime
import win32gui


class UploadDetector:
    """检测文件上传操作"""
    
    def __init__(self, config, stats_collector=None):
        self.config = config
        self.stats = stats_collector
        
        # 应用程序配置
        self.app_configs = config.get("upload_detection.monitored_apps", {})
        
        # 临时文件映射: {temp_file_path: original_file_info}
        self.temp_file_map = {}
        self.temp_file_lock = threading.Lock()
        
        # 文件对话框检测
        self.last_dialog_files = []
        self.dialog_check_interval = config.get("upload_detection.dialog_check_interval", 0.5)
        
        # 文件选择跟踪: {timestamp: {file_path, window_title}}
        self.file_selections = []
        self.file_selection_lock = threading.Lock()
        self.max_selection_history = 100  # 最多保留100条记录
        
        # 应用名称映射规则（扩展版 - 支持40+应用）
        self.app_name_patterns = {
            # AI类应用
            "kimi": ["kimi", "moonshot", "月之暗面", "www.kimi.com"],
            "tongyi": ["tongyi", "通义千问", "tongyi.aliyun.com"],
            "yiyan": ["文心一言", "yiyan", "yiyan.baidu.com"],
            "doubao": ["豆包", "doubao", "www.doubao.com", "doubao.exe"],
            "chatgpt": ["chatgpt", "chat.openai.com", "openai"],
            "deepseek": ["deepseek", "www.deepseek.com"],
            "yuanbao": ["元宝", "yuanbao", "yuanbao.exe"],
            "cherry_studio": ["cherry studio", "cherry"],
            "chatbox": ["chatbox", "chatbox.exe"],
            
            # 网盘类
            "baidu_pan": ["百度网盘", "pan.baidu.com", "baiduyun"],
            "quark": ["夸克", "quark", "quark.exe"],
            "aliyun_drive": ["阿里云盘", "aliyundrive", "www.aliyundrive.com"],
            "weiyun": ["腾讯微云", "weiyun", "weiyun.com"],
            "115_pan": ["115", "115网盘", "115.com"],
            "jianguoyun": ["坚果云", "jianguoyun", "nutstore"],
            
            # 邮箱类
            "qq_mail": ["qq邮箱", "mail.qq.com"],
            "163_mail": ["网易邮箱", "mail.163.com", "163邮箱"],
            
            # 即时通讯类
            "qq": ["qq", "腾讯qq", "qq.exe"],
            "wechat": ["微信", "wechat", "weixin", "wechat.exe", "weixin.qq.com"],
            "dingtalk": ["钉钉", "dingtalk", "dingtalk.exe"],
            
            # 会议类
            "zoom": ["zoom", "zoom.us"],
            "tencent_meeting": ["腾讯会议", "tencent meeting", "tencentmeeting.exe"],
            "dingtalk_meeting": ["钉钉会议", "meeting.exe"],
            
            # 协作办公类
            "feishu": ["飞书", "feishu", "lark", "feishu.exe"],
            
            # 代码托管类
            "github": ["github", "github.com"],
            "gitee": ["gitee", "gitee.com"],
            
            # 技术社区类
            "csdn": ["csdn", "www.csdn.net"],
            
            # 笔记类
            "youdao": ["有道云笔记", "youdao", "note.youdao.com"],
            
            # 工具类
            "audio2edit": ["audio2edit", "audio2edit.com", "文本转语音"],
            "smallpdf": ["smallpdf", "smallpdf.com"],
            "ilovepdf": ["ilovepdf", "www.ilovepdf.com"],
            "online_convert": ["online-convert", "www.online-convert.com"]
        }
        
        # 启动文件对话框监控
        if config.get("upload_detection.enable_dialog_detection", True):
            self._start_dialog_monitor()
    
    def _start_dialog_monitor(self):
        """启动文件对话框监控线程"""
        def monitor_worker():
            while True:
                try:
                    self._check_file_dialogs()
                    time.sleep(self.dialog_check_interval)
                except Exception as e:
                    pass
        
        thread = threading.Thread(target=monitor_worker, daemon=True)
        thread.start()
    
    def _check_file_dialogs(self):
        """检查是否有文件选择对话框打开"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return
            
            window_title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            
            # 检测常见的文件对话框
            dialog_indicators = [
                "#32770",  # 标准对话框类名
                "打开", "Open", "选择文件", "Select File",
                "上传", "Upload", "发送", "Send"
            ]
            
            is_dialog = any(indicator in class_name or indicator in window_title 
                          for indicator in dialog_indicators)
            
            if is_dialog:
                # 记录可能的文件选择操作
                if self.stats:
                    self.stats.record_event("file_dialog_detected")
                
                # 尝试获取背后的父窗口（浏览器）
                try:
                    parent_hwnd = win32gui.GetParent(hwnd)
                    if parent_hwnd:
                        parent_title = win32gui.GetWindowText(parent_hwnd)
                        # 识别上传目标应用
                        app =  self.identify_upload_target(parent_title)
                        if app:
                            # 记录文件对话框与应用的关联
                            with self.file_selection_lock:
                                self.file_selections.append({
                                    "timestamp": time.time(),
                                    "window_title": parent_title,
                                    "app_name": app,
                                    "dialog_title": window_title
                                })
                except Exception:
                    pass
                    
        except Exception:
            pass
    
    def is_temp_file_for_upload(self, file_path, process_name):
        """
        判断文件是否是上传操作的临时文件
        
        Args:
            file_path: 文件路径
            process_name: 操作进程名称
            
        Returns:
            dict: 如果是上传临时文件,返回关联信息;否则返回None
        """
        if not process_name:
            return None
        
        # 检查是否是监控的应用程序
        for app_name, app_config in self.app_configs.items():
            if not app_config.get("enabled", True):
                continue
                
            # 检查进程名是否匹配
            process_patterns = app_config.get("process_names", [])
            if not any(pattern.lower() in process_name.lower() for pattern in process_patterns):
                continue
            
            # 检查是否在临时目录中
            temp_dirs = app_config.get("temp_directories", [])
            for temp_dir in temp_dirs:
                # 展开环境变量
                expanded_dir = os.path.expandvars(temp_dir)
                if expanded_dir.lower() in file_path.lower():
                    return {
                        "app_name": app_name,
                        "app_display_name": app_config.get("display_name", app_name),
                        "temp_directory": expanded_dir,
                        "upload_type": app_config.get("upload_type", "unknown")
                    }
        
        return None
    
    def try_associate_original_file(self, temp_file_path, file_name):
        """
        尝试关联临时文件与原始文件
        
        Args:
            temp_file_path: 临时文件路径
            file_name: 文件名
            
        Returns:
            str: 可能的原始文件路径,如果无法确定则返回None
        """
        # 策略1: 在常见位置搜索同名文件
        common_dirs = [
            os.path.expandvars("%USERPROFILE%\\Documents"),
            os.path.expandvars("%USERPROFILE%\\Desktop"),
            os.path.expandvars("%USERPROFILE%\\Downloads")
        ]
        
        for search_dir in common_dirs:
            if not os.path.exists(search_dir):
                continue
                
            # 搜索同名文件
            for root, dirs, files in os.walk(search_dir):
                if file_name in files:
                    potential_path = os.path.join(root, file_name)
                    # 检查文件大小是否接近(允许10%误差)
                    try:
                        temp_size = os.path.getsize(temp_file_path)
                        orig_size = os.path.getsize(potential_path)
                        if abs(temp_size - orig_size) / max(temp_size, orig_size) < 0.1:
                            return potential_path
                    except Exception:
                        pass
        
        return None
    
    def register_temp_file(self, temp_file_path, original_file_path, upload_info):
        """
        注册临时文件与原始文件的映射
        
        Args:
            temp_file_path: 临时文件路径
            original_file_path: 原始文件路径
            upload_info: 上传信息
        """
        with self.temp_file_lock:
            self.temp_file_map[temp_file_path] = {
                "original_path": original_file_path,
                "upload_info": upload_info,
                "timestamp": time.time()
            }
    
    def get_original_file_info(self, temp_file_path):
        """
        获取临时文件对应的原始文件信息
        
        Args:
            temp_file_path: 临时文件路径
            
        Returns:
            dict: 原始文件信息,如果不存在则返回None
        """
        with self.temp_file_lock:
            return self.temp_file_map.get(temp_file_path)
    
    def identify_upload_target(self, window_title):
        """
        从窗口标题识别上传目标应用
        增强版：支持多标签页浏览器窗口标题智能解析
        
        Args:
            window_title: 窗口标题
            
        Returns:
            str: 应用名称（如"Kimi", "QQ邮箱"），如果无法识别则返回None
        """
        if not window_title:
            return None
        
        # 友好显示名称映射
        friendly_names = {
            # AI类
            "kimi": "Kimi",
            "tongyi": "通义千问",
            "yiyan": "文心一言",
            "doubao": "豆包",
            "chatgpt": "ChatGPT",
            "deepseek": "DeepSeek",
            "yuanbao": "元宝",
            "cherry_studio": "Cherry Studio",
            "chatbox": "Chatbox",
            # 网盘类
            "baidu_pan": "百度网盘",
            "quark": "夸克网盘",
            "aliyun_drive": "阿里云盘",
            "weiyun": "腾讯微云",
            "115_pan": "115网盘",
            "jianguoyun": "坚果云",
            # 邮箱类
            "qq_mail": "QQ邮箱",
            "163_mail": "网易邮箱",
            # 即时通讯类
            "qq": "QQ",
            "wechat": "微信",
            "dingtalk": "钉钉",
            # 会议类
            "zoom": "Zoom",
            "tencent_meeting": "腾讯会议",
            "dingtalk_meeting": "钉钉会议",
            # 协作办公
            "feishu": "飞书",
            # 代码托管
            "github": "GitHub",
            "gitee": "Gitee",
            # 技术社区
            "csdn": "CSDN",
            # 笔记类
            "youdao": "有道云笔记",
            # 工具类
            "audio2edit": "文本转语音",
            "smallpdf": "SmallPDF",
            "ilovepdf": "iLovePDF",
            "online_convert": "OnlineConvert"
        }
        
        # 解析多标签页浏览器窗口标题
        # 格式: "标题1 和另外 N 个页面 - 浏览器名" 或 "标题1 - 标题2 - 标题3 - 浏览器名"
        tab_titles = self._parse_browser_tabs(window_title)
        
        # 为每个检测到的应用计算置信度分数
        app_scores = {}  # {app_key: (score, friendly_name)}
        
        for idx, title in enumerate(tab_titles):
            title_lower = title.lower()
            
            for app_key, patterns in self.app_name_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in title_lower:
                        # 计算置信度分数
                        # 第一个标签（通常是活动标签）权重最高
                        score = 100 - (idx * 20)  # 第1个标签100分，第2个80分，依此类推
                        
                        # 精确匹配加分
                        if pattern.lower() == title_lower.strip():
                            score += 50
                        
                        # 域名匹配加分（更可靠）
                        if '.' in pattern and pattern in title_lower:
                            score += 30
                        
                        # 更新最高分数
                        if app_key not in app_scores or score > app_scores[app_key][0]:
                            app_scores[app_key] = (score, friendly_names.get(app_key, app_key))
        
        # 返回置信度最高的应用
        if app_scores:
            best_app = max(app_scores.items(), key=lambda x: x[1][0])
            return best_app[1][1]  # 返回友好名称
        
        return None
    
    def _parse_browser_tabs(self, window_title):
        """
        解析浏览器窗口标题，提取所有标签页标题
        
        Args:
            window_title: 完整的窗口标题
            
        Returns:
            list: 标签页标题列表，按重要性排序（第一个通常是活动标签）
        """
        if not window_title:
            return []
        
        tabs = []
        
        # 模式1: "标题 和另外 N 个页面 - 浏览器" (Edge/Chrome常见格式)
        if "和另外" in window_title or "另外" in window_title:
            # 提取第一个标签标题（可能包含多部分）
            parts = window_title.split(" 和另外")
            if parts:
                first_part = parts[0].strip()
                # 分割可能的多段式标题: "页面标题 - 应用名"
                if " - " in first_part:
                    sub_parts = first_part.split(" - ")
                    # 添加所有子部分（按顺序）
                    for sub_part in sub_parts:
                        clean_part = sub_part.strip()
                        if clean_part:
                            tabs.append(clean_part)
                else:
                    tabs.append(first_part)
        
        # 模式2: "标题1 - 标题2 - ... - 浏览器"（多个可见标签）
        elif " - " in window_title:
            parts = window_title.split(" - ")
            # 排除最后的浏览器名称和账户信息
            browser_keywords = ["edge", "chrome", "firefox", "个人", "microsoft", "google"]
            for part in parts:
                part_lower = part.lower()
                # 跳过浏览器相关关键词
                if not any(keyword in part_lower for keyword in browser_keywords):
                    clean_part = part.strip()
                    if clean_part:
                        tabs.append(clean_part)
        
        # 模式3: 单标签或无法解析
        if not tabs:
            # 尝试简单分割去除浏览器信息
            clean_title = window_title.split(" - ")[0].strip() if " - " in window_title else window_title.strip()
            tabs.append(clean_title)
        
        return tabs
    
    def extract_uploaded_filename(self, temp_file_path):
        """
        从临时文件路径提取原始上传文件名
        尝试多种策略找到原始文件
        
        Args:
            temp_file_path: 临时文件路径
            
        Returns:
            tuple: (original_filename, full_path) 或 (None, None)
        """
        if not os.path.exists(temp_file_path):
            return None, None
        
        # 提取文件名
        temp_filename = os.path.basename(temp_file_path)
        
        # 策略1: 检查最近的文件选择记录
        with self.file_selection_lock:
            # 查找最近5秒内的文件选择
            recent_time = time.time() - 5.0
            recent_selections = [
                sel for sel in self.file_selections
                if sel["timestamp"] > recent_time
            ]
            
        # 策略2: 使用现有的关联方法
        original_path = self.try_associate_original_file(temp_file_path, temp_filename)
        if original_path:
            return os.path.basename(original_path), original_path
        
        # 策略3: 如果临时文件名本身就是有意义的文件名
        # （有些浏览器会保留原始文件名）
        if "." in temp_filename and not temp_filename.startswith("tmp"):
            return temp_filename, temp_file_path
        
        return None, None
    
    def track_file_selection(self, file_path, window_info=None):
        """
        跟踪用户选择的文件（当检测到文件访问时调用）
        
        Args:
            file_path: 被访问的文件路径
            window_info: 窗口信息字典
        """
        # 检查文件是否是常见文档类型
        common_exts = [".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx", 
                       ".txt", ".zip", ".rar", ".jpg", ".png", ".mp4"]
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in common_exts:
            return
        
        # 识别窗口标题中的应用
        app_name = None
        window_title = ""
        if window_info:
            window_title = window_info.get("window_title", "")
            app_name = self.identify_upload_target(window_title)
        
        with self.file_selection_lock:
            self.file_selections.append({
                "timestamp": time.time(),
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "window_title": window_title,
                "app_name": app_name
            })
            
            # 限制历史记录大小
            if len(self.file_selections) > self.max_selection_history:
                self.file_selections = self.file_selections[-self.max_selection_history:]
    
    def enrich_upload_event(self, temp_file_path, window_info, process_info):
        """
        丰富上传事件信息，添加目标应用和原始文件名
        
        Args:
            temp_file_path: 临时文件路径
            window_info: 窗口信息
            process_info: 进程信息
            
        Returns:
            dict: 包含完整上传信息的字典
        """
        upload_event = {
            "event_type": "upload_detected",
            "timestamp": datetime.now().isoformat(),
            "temp_file_path": temp_file_path,
            "temp_file_name": os.path.basename(temp_file_path)
        }
        
        # 添加窗口信息
        if window_info:
            window_title = window_info.get("window_title", "")
            upload_event["window_title"] = window_title
            upload_event["window_class"] = window_info.get("window_class", "")
            
            # 识别目标应用
            app_name = self.identify_upload_target(window_title)
            if app_name:
                upload_event["app_name"] = app_name
        
        # 添加进程信息
        if process_info:
            upload_event["process_name"] = process_info.get("process_name", "")
            upload_event["process_path"] = process_info.get("process_path", "")
        
        # 提取原始文件名
        original_filename, original_path = self.extract_uploaded_filename(temp_file_path)
        if original_filename:
            upload_event["uploaded_file"] = original_filename
            if original_path:
                upload_event["original_file_path"] = original_path
        
        # 尝试从最近的文件选择记录中找到匹配
        with self.file_selection_lock:
            recent_time = time.time() - 10.0  # 10秒内
            for selection in reversed(self.file_selections):
                if selection["timestamp"] < recent_time:
                    break
                
                # 如果应用名称匹配
                if upload_event.get("app_name") == selection.get("app_name"):
                    if "uploaded_file" not in upload_event:
                        upload_event["uploaded_file"] = selection["file_name"]
                        upload_event["original_file_path"] = selection["file_path"]
                    break
        
        return upload_event
    
    def cleanup_old_mappings(self, max_age_seconds=3600):
        """
        清理过期的文件映射和选择记录
        
        Args:
            max_age_seconds: 最大保留时间(秒)
        """
        with self.temp_file_lock:
            now = time.time()
            expired_keys = [
                key for key, value in self.temp_file_map.items()
                if now - value["timestamp"] > max_age_seconds
            ]
            for key in expired_keys:
                del self.temp_file_map[key]
        
        # 清理过期的文件选择记录
        with self.file_selection_lock:
            now = time.time()
            self.file_selections = [
                sel for sel in self.file_selections
                if now - sel["timestamp"] <= max_age_seconds
            ]
