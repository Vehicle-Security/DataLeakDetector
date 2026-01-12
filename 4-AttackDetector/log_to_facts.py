# -*- coding: utf-8 -*-
"""
log_to_facts.py - 日志转 Datalog 事实转换器

将 win_monitor 生成的 JSON 日志转换为 Soufflé 可以处理的 .facts 文件。
每种事件类型生成对应的事实文件。
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Set


class LogToFactsConverter:
    """日志到 Datalog 事实的转换器"""
    
    # 敏感文件扩展名
    SENSITIVE_EXTENSIONS = {
        '.docx', '.doc', '.pdf', '.xlsx', '.xls', '.pptx', '.ppt',
        '.txt', '.csv', '.rtf', '.odt', '.key', '.numbers', '.pages'
    }
    
    # AI/上传网站关键词
    UPLOAD_SITE_KEYWORDS = {
        'doubao', 'chatgpt', 'claude', 'gemini', 'copilot', 'poe.com',
        'kimi', 'tongyi', 'wenxin', 'spark', 'moonshot',
        'dropbox', 'onedrive', 'drive.google', 'wetransfer', 'sendspace'
    }
    
    # 敏感应用进程名
    SENSITIVE_APPS = {
        'winword', 'excel', 'powerpnt', 'acrord32', 'acrobat',
        'notepad', 'notepad++', 'code', 'sublime_text'
    }
    
    # 风险应用进程名
    RISKY_APPS = {
        'chrome', 'msedge', 'firefox', 'opera', 'brave',
        'wechat', 'qq', 'telegram', 'slack', 'teams'
    }
    
    def __init__(self, output_dir: str = "output/facts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 事实收集器
        self.facts = {
            'open_file': [],
            'create_file': [],
            'modify_file': [],
            'rename_file': [],
            'delete_file': [],
            'app_switch': [],
            'browser_access': [],
            'clipboard_text': [],
            'clipboard_image': [],
            'clipboard_paste': [],
            'sensitive_extension': set(),
            'upload_site': set(),
            'sensitive_app': set(),
            'risky_app': set(),
            'im_app': set(),
        }
        
        # 初始化静态事实
        self._init_static_facts()
    
    def _init_static_facts(self):
        """初始化静态事实（敏感扩展名、网站等）"""
        for ext in self.SENSITIVE_EXTENSIONS:
            self.facts['sensitive_extension'].add(ext)
        
        for keyword in self.UPLOAD_SITE_KEYWORDS:
            self.facts['upload_site'].add(keyword)
        
        for app in self.SENSITIVE_APPS:
            self.facts['sensitive_app'].add(app)
        
        for app in self.RISKY_APPS:
            self.facts['risky_app'].add(app)
    
    def _parse_timestamp(self, ts_str: str) -> int:
        """将时间戳字符串转换为 Unix 时间戳（秒）"""
        try:
            # 处理 ISO 格式: 2026-01-09T10:30:00.000
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except:
            return 0
    
    def _escape_string(self, s: str) -> str:
        """转义字符串用于 Datalog"""
        if not s:
            return '""'
        # 替换特殊字符
        s = s.replace('\\', '/')  # 路径分隔符统一
        s = s.replace('"', '\\"')
        s = s.replace('\n', ' ')
        s = s.replace('\t', ' ')
        return f'"{s}"'
    
    def _get_extension(self, path: str) -> str:
        """获取文件扩展名"""
        if not path:
            return ""
        _, ext = os.path.splitext(path)
        return ext.lower()
    
    def _normalize_process(self, process_name: str) -> str:
        """规范化进程名"""
        if not process_name:
            return "unknown"
        # 移除 .exe 后缀并转小写
        name = process_name.lower()
        if name.endswith('.exe'):
            name = name[:-4]
        return name
    
    def _is_browser(self, process_name: str) -> bool:
        """判断是否是浏览器进程"""
        browsers = {'chrome', 'msedge', 'firefox', 'opera', 'brave', 'safari'}
        return self._normalize_process(process_name) in browsers
    
    def _extract_url_from_title(self, title: str) -> str:
        """从窗口标题提取 URL（简化实现）"""
        # 浏览器标题通常包含网站名或 URL
        if not title:
            return ""
        # 简单提取：查找常见域名模式
        url_pattern = r'(https?://[^\s]+|[a-zA-Z0-9.-]+\.(com|cn|org|net|io))'
        match = re.search(url_pattern, title)
        if match:
            return match.group(1)
        return title[:50]  # 使用标题前50字符作为标识
    
    def convert_event(self, event: Dict[str, Any]):
        """转换单个事件为 Datalog 事实"""
        event_type = event.get('event_type', '')
        timestamp = self._parse_timestamp(event.get('timestamp', ''))
        
        # 提取进程信息
        process_info = event.get('process_info', {})
        process_name = self._normalize_process(process_info.get('process_name', ''))
        
        # 提取文件信息
        file_path = event.get('file_path', '')
        file_name = event.get('file_name', '')
        
        # 根据事件类型生成事实
        if event_type == 'opened':
            self.facts['open_file'].append((
                self._escape_string(process_name),
                self._escape_string(file_path),
                timestamp
            ))
        
        elif event_type == 'created':
            self.facts['create_file'].append((
                self._escape_string(process_name),
                self._escape_string(file_path),
                timestamp
            ))
        
        elif event_type == 'modified':
            self.facts['modify_file'].append((
                self._escape_string(process_name),
                self._escape_string(file_path),
                timestamp
            ))
        
        elif event_type == 'renamed':
            dest_path = event.get('destination_path', '')
            self.facts['rename_file'].append((
                self._escape_string(process_name),
                self._escape_string(file_path),
                self._escape_string(dest_path),
                timestamp
            ))
        
        elif event_type == 'deleted':
            self.facts['delete_file'].append((
                self._escape_string(process_name),
                self._escape_string(file_path),
                timestamp
            ))
        
        elif event_type == 'app_switch':
            # 从窗口标题提取信息
            window_info = event.get('window_info', {})
            window_title = window_info.get('window_title', '')
            
            # 如果是浏览器，记录浏览器访问
            if self._is_browser(process_name):
                url = self._extract_url_from_title(window_title)
                if url:
                    self.facts['browser_access'].append((
                        self._escape_string(process_name),
                        self._escape_string(url),
                        timestamp
                    ))
            
            # 记录应用切换
            app_name = event.get('app_name', process_name)
            self.facts['app_switch'].append((
                self._escape_string("prev"),  # 简化：不追踪前一个应用
                self._escape_string(app_name),
                timestamp
            ))
        
        elif event_type == 'website_visit':
            window_info = event.get('window_info', {})
            window_title = window_info.get('window_title', '')
            url = self._extract_url_from_title(window_title)
            
            self.facts['browser_access'].append((
                self._escape_string(process_name),
                self._escape_string(url),
                timestamp
            ))
        
        # 剪贴板事件
        elif event_type == 'clipboard_text':
            content_hash = event.get('content_hash', '')
            self.facts['clipboard_text'].append((
                self._escape_string(process_name),
                self._escape_string(content_hash),
                timestamp
            ))
        
        elif event_type == 'clipboard_image':
            content_hash = event.get('content_hash', '')
            self.facts['clipboard_image'].append((
                self._escape_string(process_name),
                self._escape_string(content_hash),
                timestamp
            ))
        
        elif event_type == 'clipboard_paste':
            content_hash = event.get('content_hash', '')
            self.facts['clipboard_paste'].append((
                self._escape_string(process_name),
                self._escape_string(content_hash),
                timestamp
            ))
    
    def convert_log_file(self, log_path: str) -> int:
        """转换整个日志文件"""
        event_count = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试解析为 JSON 数组
            try:
                events = json.loads(content)
                if isinstance(events, list):
                    for event in events:
                        self.convert_event(event)
                        event_count += 1
                else:
                    self.convert_event(events)
                    event_count = 1
            except json.JSONDecodeError:
                # 尝试按行解析（JSON Lines 格式）
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('[') and not line.startswith(']'):
                        # 移除可能的逗号
                        if line.endswith(','):
                            line = line[:-1]
                        try:
                            event = json.loads(line)
                            self.convert_event(event)
                            event_count += 1
                        except:
                            continue
        
        print(f"[LOG_TO_FACTS] 转换了 {event_count} 个事件")
        return event_count
    
    def write_facts(self) -> Dict[str, str]:
        """写入所有事实文件"""
        files_written = {}
        
        # 写入动态事实
        fact_configs = [
            ('open_file', 3),      # (process, path, time)
            ('create_file', 3),
            ('modify_file', 3),
            ('rename_file', 4),    # (process, src, dst, time)
            ('delete_file', 3),
            ('app_switch', 3),     # (from, to, time)
            ('browser_access', 3), # (browser, url, time)
            ('clipboard_text', 3), # (process, hash, time)
            ('clipboard_image', 3), # (process, hash, time)
            ('clipboard_paste', 3), # (process, hash, time)
        ]
        
        for fact_name, _ in fact_configs:
            facts = self.facts.get(fact_name, [])
            if facts:
                path = os.path.join(self.output_dir, f"{fact_name}.facts")
                with open(path, 'w', encoding='utf-8') as f:
                    for fact in facts:
                        f.write('\t'.join(str(x) for x in fact) + '\n')
                files_written[fact_name] = path
                print(f"[LOG_TO_FACTS] 写入 {len(facts)} 条 {fact_name} 事实")
        
        # 写入静态事实
        static_facts = ['sensitive_extension', 'upload_site', 'sensitive_app', 'risky_app']
        for fact_name in static_facts:
            facts = self.facts.get(fact_name, set())
            if facts:
                path = os.path.join(self.output_dir, f"{fact_name}.facts")
                with open(path, 'w', encoding='utf-8') as f:
                    for value in facts:
                        f.write(f'"{value}"\n')
                files_written[fact_name] = path
        
        return files_written


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON logs to Datalog facts')
    parser.add_argument('--input', '-i', required=True, help='Input JSON log file')
    parser.add_argument('--output', '-o', default='output/facts', help='Output directory for facts')
    
    args = parser.parse_args()
    
    converter = LogToFactsConverter(output_dir=args.output)
    converter.convert_log_file(args.input)
    converter.write_facts()
    
    print(f"\n[LOG_TO_FACTS] 事实文件已写入: {args.output}")


if __name__ == '__main__':
    main()
