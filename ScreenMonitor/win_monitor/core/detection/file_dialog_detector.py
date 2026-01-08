# -*- coding: utf-8 -*-
"""
文件对话框检测器 (UIA版)
使用 UI Automation 准确获取文件路径
"""
import time
import threading
import os
import psutil
try:
    import uiautomation as auto
except ImportError:
    auto = None

from .recent_file_tracker import get_recent_file_tracker

class FileDialogDetector:
    """监测文件选择对话框并记录选中的文件"""
    
    def __init__(self, event_callback=None):
        """初始化"""
        self.event_callback = event_callback
        self.running = False
        self.monitor_thread = None
        self.detected_files = [] 
        self.file_lock = threading.Lock()
        self.last_dialog_handle = 0
        self.last_check_time = 0
        
        # 已知的对话框标题
        self.dialog_titles = ["打开", "Open", "选择文件", "Select File", "上传", "Upload", "保存", "Save As"]
        
    def start_monitoring(self, interval=1.0):
        """启动监控线程"""
        if self.running:
            return
            
        if not auto:
            print("[FILE_DIALOG] ❌ uiautomation module not found. Monitoring disabled.")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("[FILE_DIALOG] Monitoring started (UIA mode)")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self, interval):
        """监控循环"""
        if not auto:
            return

        try:
            # 在线程中使用 UIA 需要初始化
            with auto.UIAutomationInitializerInThread():
                while self.running:
                    try:
                        self._check_with_uia()
                    except Exception as e:
                        # print(f"[FILE_DIALOG] Error: {e}")
                        pass
                    time.sleep(interval)
        except Exception as e:
            print(f"[FILE_DIALOG] Thread error: {e}")
            
    def _check_with_uia(self):
        """使用 UI Automation 检查对话框"""
        # 查找所有顶层窗口（优化：只查找在前台的或最近激活的）
        # 这里直接查找类名为 #32770 的窗口，通常是标准文件对话框
        # 现代应用也可能使用 Chromium 的对话框，但它们看起来也是 #32770 或者 Chrome_WidgetWin_1
        
        # 查找特定的 #32770 对话框
        # ControlType: WindowControl
        # ClassName: #32770
        
        # 限制查找深度和范围以提高性能
        root = auto.GetRootControl()
        # 查找当前活动窗口，如果它是一个文件对话框
        window = auto.GetForegroundControl()
        
        if not window:
            return

        try:
            # 检查是否是对话框
            is_dialog = False
            if window.ClassName == "#32770":
                is_dialog = True
            elif window.Name in self.dialog_titles:
                is_dialog = True
            
            # 如果当前窗口不是，可能是子窗口（例如在Edge中）
            if not is_dialog:
                # 尝试找 #32770 子窗口
                dialog = window.WindowControl(ClassName="#32770", searchDepth=2)
                if dialog.Exists(maxSearchSeconds=0.1):
                    window = dialog
                    is_dialog = True

            if not is_dialog:
                return
                
            # 检查标题
            title = window.Name
            if not any(t in title for t in self.dialog_titles):
                return
                
            # 获取句柄以避免重复处理
            handle = window.NativeWindowHandle
            if handle == self.last_dialog_handle and time.time() - self.last_check_time < 2:
                return

            # print(f"[FILE_DIALOG] Found dialog: {title}")
            
            # 尝试获取文件名输入框
            # 通常是 ComboBox 名为 "文件名:" (File name:) 或者 Edit
            # 需要支持中英文
            
            file_path = None
            
            # 策略1: 查找名为"文件名:"的ComboBox
            # Group "文件名:" -> ComboBox -> Edit
            
            # 常见的标签名
            labels = ["文件名:", "File name:", "文件名(N):", "File name:"]
            
            for label in labels:
                # 查找 Name=label 的控件，通常是 TextControl 或 ComboBox
                # 有时 ComboBox 直接叫这个名字
                combo = window.ComboBoxControl(Name=label)
                if combo.Exists(maxSearchSeconds=0.1):
                    if combo.GetValuePattern():
                        val = combo.GetValuePattern().Value
                        if val:
                            file_path = val
                            break
                    # 也可以尝试找它下面的Edit
                    edit = combo.EditControl()
                    if edit.Exists(maxSearchSeconds=0.1):
                         # ValuePattern for Edit
                        try:
                            file_path = edit.GetValuePattern().Value
                        except:
                            pass
                        if file_path: break

            # 策略2: 如果没找到，找第一个 Value 不为空的 ComboBoxEx32 或 ComboBox
            if not file_path:
                combos = window.GetChildren()
                for child in combos:
                    if child.ControlTypeName == "ComboBoxControl" or child.ClassName == "ComboBoxEx32":
                         try:
                            val = child.GetValuePattern().Value
                            if val and ("." in val or "\\" in val):
                                file_path = val
                                break
                         except:
                             pass
            
            # 策略3: 查找 EditControl 显示完整路径的
            if not file_path:
                edits = window.GetChildren() # 这里只会浅层搜索，可能需要深度
                # 现代对话框路径在上面的地址栏? 不，我们要在下面的文件名框
                pass

            
            # 如果获取到了路径
            if file_path:
                # 验证路径
                # 这里的 file_path 可能只是文件名，如果是在当前目录下
                # UI Automation 很难直接获取"当前目录"，除非去读地址栏
                # 但是现代对话框通常会在点击"打开"时，文件框里有完整路径（有时）
                # 或者我们需要结合 RecentFileTracker 或者是地址栏
                
                # 尝试获取地址栏当前路径
                folder_path = ""
                # 地址栏通常在上面，Toolbar -> Address Band -> ...
                # 简便方法：查找 "地址: *" 的 Text? 
                # 实际上很难通用。
                
                # 但是！如果用户选择了文件，通常 file_path 只是文件名。
                # 如果我们能获取到完整路径最好
                
                # 让我们先记录提取到的内容
                # print(f"[FILE_DIALOG] Extracted text: {file_path}")
                
                if not os.path.isabs(file_path):
                    # 尝试推断目录
                    # 使用 RecentFileTracker 查找名为 file_path 的文件
                    try:
                        from .recent_file_tracker import get_recent_file_tracker
                        tracker = get_recent_file_tracker()
                        # 使用新的 find_file_by_name 方法，不依赖 atime
                        found_info = tracker.find_file_by_name(file_path)
                        if found_info:
                            full_path = found_info['path']
                            print(f"[FILE_DIALOG] ℹ️  通过文件名找到完整路径: {full_path}")
                    except Exception as e:
                        print(f"[FILE_DIALOG] Path inference error: {e}")
                
                # 如果仍然不是绝对路径，且存在于系统中
                if not os.path.isabs(full_path):
                     # 如果实在找不到，只能忽略或记录文件名
                     pass

                if os.path.isabs(full_path) and os.path.exists(full_path):
                     self._handle_file_selected(full_path, window)
                     self.last_dialog_handle = handle
                     self.last_check_time = time.time()
                elif file_path and not os.path.isabs(file_path):
                     # 即使是相对路径（文件名），也记录下来，结合app_name
                     # 尝试在 recent_tracker 中再次搜索?
                     pass
                     
        except Exception as e:
            # print(f"[FILE_DIALOG] Error processing window: {e}")
            pass

    def _handle_file_selected(self, file_path, window_control):
        """处理文件选择"""
        with self.file_lock:
             # 去重
             for f in self.detected_files[-5:]:
                 if f['file_path'] == file_path and time.time() - f.get('timestamp', 0) < 5:
                     return

             # 获取应用名
             try:
                 process_id = window_control.ProcessId
                 process = psutil.Process(process_id)
                 app_name = process.name()
                 
                 # 如果是浏览器进程，获取窗口标题
                 window_title = window_control.Name
                 
                # 修正Edge/Chrome应用名
                 if app_name.lower() in ['msedge.exe', 'chrome.exe', 'webviewhost.exe']:
                     try:
                         # 1. 尝试向上查找父进程，直到找到非浏览器进程
                         parent = process.parent()
                         while parent:
                             p_name = parent.name().lower()
                             if p_name not in ['msedge.exe', 'chrome.exe', 'explorer.exe', 'svchost.exe', 'services.exe']:
                                 app_name = parent.name()
                                 break
                             parent = parent.parent()
                             
                         # 2. 如果还是浏览器，尝试通过窗口标题映射
                         if app_name.lower() in ['msedge.exe', 'chrome.exe']:
                              # 常见应用映射
                              title_lower = window_title.lower()
                              if '豆包' in title_lower:
                                  app_name = 'Doubao.exe'
                              elif '微信' in title_lower or 'wechat' in title_lower:
                                  app_name = 'WeChat.exe'
                              elif '钉钉' in title_lower or 'dingtalk' in title_lower:
                                  app_name = 'DingTalk.exe'
                     except:
                         pass
             except:
                 app_name = "Unknown"
                 window_title = "Unknown"

             event_info = {
                "timestamp": time.time(),
                "event_type": "file_selected",
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "app_name": app_name,
                "window_title": window_title,
                "detection_method": "file_dialog"
             }
             
             self.detected_files.append(event_info)
             print(f"[FILE_DIALOG] ✅ Detected file: {file_path}")
             
             if self.event_callback:
                 self.event_callback(event_info)

# 单例
_global_detector = None

def get_file_dialog_detector(event_callback=None):
    global _global_detector
    if _global_detector is None:
        _global_detector = FileDialogDetector(event_callback)
    return _global_detector
