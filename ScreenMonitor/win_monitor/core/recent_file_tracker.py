# -*- coding: utf-8 -*-
"""
Recent File Tracker - 跟踪最近访问的文件
支持按时间扫描和按文件名查找
"""
import os
import time
from pathlib import Path


class RecentFileTracker:
    """跟踪最近被访问的用户文件"""
    
    def __init__(self):
        # 跟踪目录
        self.tracked_directories = [
            "D:\\code\\win_monitor\\test",  # 用户文件位置
            "D:\\test",
        ]
        
        # 添加用户标准目录
        home = Path.home()
        common_dirs = ["Desktop", "Documents", "Downloads"]
        for d in common_dirs:
            self.tracked_directories.append(str(home / d))
        
    def scan_recent_files(self, seconds=10):
        """
        扫描最近访问的文件
        
        Args:
            seconds: 查找最近N秒内访问的文件
            
        Returns:
            list: 最近访问的文件列表
        """
        cutoff_time = time.time() - seconds
        candidates = []
        
        for directory in self.tracked_directories:
            if not os.path.exists(directory):
                continue
            
            try:
                # 只扫描目录下的直接文件
                for item in os.listdir(directory):
                    file_path = os.path.join(directory, item)
                    
                    # 跳过目录
                    if not os.path.isfile(file_path):
                        continue
                    
                    try:
                        # 获取访问时间
                        atime = os.path.getatime(file_path)
                        mtime = os.path.getmtime(file_path)
                        
                        # 如果 atime 无效（比如没更新），可以考虑 mtime
                        # 但这里我们只记录，由调用者判断
                        
                        if atime > cutoff_time:
                            # 只关注用户文档格式
                            _, ext = os.path.splitext(file_path)
                            if ext.lower() in ['.docx', '.pdf', '.xlsx', '.pptx', '.txt', '.doc', '.xls', '.ppt', '.zip', '.rar']:
                                candidates.append({
                                    'path': file_path,
                                    'name': item,
                                    'access_time': atime,
                                    'modify_time': mtime,
                                    'size': os.path.getsize(file_path)
                                })
                    except Exception:
                        pass
            except Exception as e:
                print(f"[TRACKER] Error scanning {directory}: {e}")
        
        # 按访问时间排序
        candidates.sort(key=lambda x: x['access_time'], reverse=True)
        return candidates

    def find_file_by_name(self, filename):
        """
        在跟踪目录中查找指定文件名的文件
        支持模糊匹配（无扩展名匹配有扩展名文件）
        
        Args:
            filename: 文件名 (e.g. "test.docx" 或 "test")
            
        Returns:
            dict: 文件信息或None (包含path)
        """
        print(f"[TRACKER] 正在查找文件: {filename}")
        
        # 常见文档扩展名
        common_exts = ['.docx', '.pdf', '.xlsx', '.pptx', '.doc', '.xls', '.ppt', '.txt', '.zip', '.rar', '.7z']
        
        # 1. 优先扫描特定测试目录
        priority_dirs = ["D:\\code\\win_monitor\\test", "D:\\test"]
        for d in priority_dirs:
            if not os.path.exists(d):
                continue
                
            # A. 精确匹配
            target = os.path.join(d, filename)
            if os.path.isfile(target):
                print(f"[TRACKER] 在优先目录找到(精确): {target}")
                return {'path': target, 'name': filename}
            
            # B. 扩展名尝试匹配 (如果filename没有扩展名)
            if '.' not in filename:
                for ext in common_exts:
                    target = os.path.join(d, filename + ext)
                    if os.path.isfile(target):
                        print(f"[TRACKER] 在优先目录找到(补全ext): {target}")
                        return {'path': target, 'name': filename + ext}
                        
            # C. 遍历目录模糊查找
            try:
                for item in os.listdir(d):
                    # 忽略大小写，忽略扩展名比对
                    item_name, item_ext = os.path.splitext(item)
                    if item == filename or item_name == filename:
                         full_path = os.path.join(d, item)
                         if os.path.isfile(full_path):
                             print(f"[TRACKER] 在优先目录找到(模糊): {full_path}")
                             return {'path': full_path, 'name': item}
            except:
                pass

        # 2. 扫描所有跟踪目录
        for directory in self.tracked_directories:
            if not os.path.exists(directory):
                continue
                
            target_path = os.path.join(directory, filename)
            if os.path.isfile(target_path):
                print(f"[TRACKER] 在跟踪目录找到: {target_path}")
                return {'path': target_path, 'name': filename}
                
            # 尝试补全扩展名
            if '.' not in filename:
                for ext in common_exts:
                    target = os.path.join(directory, filename + ext)
                    if os.path.isfile(target):
                        print(f"[TRACKER] 在跟踪目录找到(补全ext): {target}")
                        return {'path': target, 'name': filename + ext}
                        
        return None

    def get_likely_uploaded_file(self, time_window=5):
        """获取最可能被上传的文件(基于时间)"""
        candidates = self.scan_recent_files(time_window)
        if candidates:
            return candidates[0]
        return None


# 全局实例
_tracker = None

def get_recent_file_tracker():
    """获取全局tracker实例"""
    global _tracker
    if _tracker is None:
        _tracker = RecentFileTracker()
    return _tracker
