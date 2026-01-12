"""
ThreatDetector LLM Prompt Templates
根据 FrameAnalyzer 和 ScreenMonitor 的实际格式设计
"""

from typing import List, Dict
from datetime import datetime


class PromptTemplates:
    """LLM Prompt 模板管理"""
    
    SYSTEM_PROMPT = """你是数据安全分析专家，专门从系统日志和视频帧分析中识别数据泄露行为。

你的任务是:
1. 理解系统日志中的文件操作
2. 结合视频帧分析理解用户行为
3. 生成结构化的Datalog事实，用于污点追踪

## Datalog 关系定义

### OpenFile(OperationID, Process, File)
含义: 进程打开/访问了文件
示例: OpenFile("op_1_20260110T100000", "Excel", "/Users/admin/Documents/工资表.xlsx")

### TransferFile(OperationID, Process, SrcFile, DstFile, Timestamp)
含义: 文件从源位置转移到目标位置（同进程内）
典型场景:
- 复制粘贴: TransferFile("op_2", "Excel", "工资表.xlsx", "Clipboard")
- 文件重命名: TransferFile("op_3", "Finder", "old.txt", "new.txt")
- 模态转换: TransferFile("op_4", "Python", "文档.pdf", "文档.wav")

### CrossProcessTransfer(OperationID, FromProc, ToProc, SharedData, Timestamp) ⭐
含义: 数据在不同进程间传输（重要！）
典型场景:
- 剪贴板复制粘贴: CrossProcessTransfer("op_3", "Excel", "WeChat", "Clipboard")
- 共享内存: CrossProcessTransfer("op_5", "AppA", "AppB", "SharedMem")
- 管道通信: CrossProcessTransfer("op_6", "Producer", "Consumer", "Pipe")

关键: 当看到"复制→粘贴"跨应用时，使用CrossProcessTransfer！

### LeakFile(OperationID, Process, File, LeakChannel, Timestamp)
含义: 文件被泄露到外部（网络、云端、外部设备）
典型场景:
- 网络发送: LeakFile("op_6", "WeChat", "Clipboard")
- 文件上传: LeakFile("op_7", "Chrome", "机密文档.pdf")
- 云端同步: LeakFile("op_8", "Dropbox", "项目资料.zip")

## 输出格式

必须返回JSON数组，每个元素包含:
- relation: "OpenFile" | "TransferFile" | "LeakFile"
- operation_id: 格式 "op_{序号}_{ISO时间戳}"
- process: 进程名称
- file: 文件路径或抽象名称（如"Clipboard"）
- dst_file: (仅TransferFile需要) 目标文件
- description: 中文描述，说明操作的安全含义

## 分析规则

1. **文件访问**: opened/read/access → OpenFile
2. **数据传输**: copy/paste/rename/convert → TransferFile
3. **外部泄露**: network_send/upload/sync → LeakFile
4. **剪贴板**: 视为数据传输的中间媒介，用"Clipboard"表示
5. **模态转换**: 文件格式变化也是TransferFile (如PDF→音频)
6. **时间戳**: 使用日志中的timestamp，格式化为 ISO 8601

## 安全思维

- 敏感文件: 工资表、财务、机密、源代码等
- 高危操作: 敏感文件 + 网络发送/外部上传
- 隐蔽泄露: 模态转换后上传（如PDF转音频）
- 跨进程传播: 数据在多个应用间传递
"""
    
    @staticmethod
    def build_user_prompt(logs: List[Dict], video_frames: List[Dict]) -> str:
        """
        构建用户提示词，动态插入日志和视频帧数据
        
        Args:
            logs: 系统日志列表，格式参考 ScreenMonitor
            video_frames: 视频帧分析列表，格式参考 FrameAnalyzer
        
        Returns:
            格式化的用户提示词
        """
        
        # 格式化日志
        logs_section = PromptTemplates._format_logs(logs)
        
        # 格式化视频帧
        frames_section = PromptTemplates._format_video_frames(video_frames)
        
        user_prompt = f"""请分析以下系统监控数据，生成Datalog事实。

## 1. 系统日志
来源: ScreenMonitor 监控模块
{logs_section}

## 2. 视频帧分析 
来源: FrameAnalyzer 行为分析
{frames_section}

## 3. 分析任务

请根据以上信息：
1. 识别所有文件操作（打开、转移、泄露）
2. 推断进程间的数据流动
3. 特别关注敏感文件的外泄路径
4. 生成完整的Datalog事实链

## 4. 输出要求

返回JSON数组，示例:
```json
[
  {{
    "relation": "OpenFile",
    "operation_id": "op_1_2026-01-10T10:00:00.000",
    "process": "Excel",
    "file": "/Users/admin/Documents/机密工资表.xlsx",
    "description": "用户使用Excel打开了机密工资表文件"
  }},
  {{
    "relation": "TransferFile",
    "operation_id": "op_2_2026-01-10T10:00:15.000",
    "process": "Excel",
    "file": "/Users/admin/Documents/机密工资表.xlsx",
    "dst_file": "Clipboard",
    "description": "用户从Excel复制了工资表内容到剪贴板"
  }},
  {{
    "relation": "CrossProcessTransfer",
    "operation_id": "op_3_2026-01-10T10:00:20.000",
    "from_process": "Excel",
    "to_process": "WeChat",
    "shared_data": "Clipboard",
    "description": "通过剪贴板从Excel传输数据到WeChat"
  }},
  {{
    "relation": "LeakFile",
    "operation_id": "op_4_2026-01-10T10:00:25.000",
    "process": "WeChat",
    "file": "Clipboard",
    "description": "微信将消息发送到网络，可能导致机密数据泄露"
  }}
]
```

**重要**: 
- TransferFile 用于同进程内的数据传输（需要 dst_file 字段）
- CrossProcessTransfer 用于跨进程传输（需要 from_process, to_process, shared_data 字段）
- LeakFile 用于数据外泄（需要 file 字段）

请开始分析:"""
        
        return user_prompt
    
    # ==================== 辅助方法 ====================
    
    @staticmethod
    def _format_logs(logs: List[Dict]) -> str:
        """
        格式化系统日志
        
        参考格式 (ScreenMonitor):
        {
            "timestamp": "2026-01-10T10:00:00.000",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/salary.xlsx",
            "process_info": {
                "process_name": "Excel",
                "pid": "1234"
            },
            "description": "用户打开了文件"
        }
        """
        if not logs:
            return "（无系统日志）"
        
        formatted_logs = []
        for i, log in enumerate(logs, 1):
            timestamp = log.get('timestamp', '')
            event_type = log.get('event_type', '未知')
            file_path = log.get('file_path', '')
            process_name = log.get('process_info', {}).get('process_name', '未知进程')
            description = log.get('description', '')
            
            # 提取时间部分（去掉日期）
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp
            
            log_entry = f"""[{i}] 时间: {time_str}
   事件: {event_type}
   进程: {process_name}
   文件: {file_path}
   描述: {description}"""
            
            formatted_logs.append(log_entry)
        
        return "\n\n".join(formatted_logs)
    
    @staticmethod
    def _format_video_frames(frames: List[Dict]) -> str:
        """
        格式化视频帧分析
        
        参考格式 (FrameAnalyzer):
        {
            "timestamp": "2025-12-28 18:41:54",
            "app_name": "iLovePDF",
            "behavior_category": "潜在隐藏行为",
            "operation_type": "格式转换",
            "description": "用户在 iLovePDF 网站上将文件转换为PDF格式"
        }
        """
        if not frames:
            return "（无视频帧分析）"
        
        formatted_frames = []
        for i, frame in enumerate(frames, 1):
            timestamp = frame.get('timestamp', '')
            app_name = frame.get('app_name', '')
            operation_type = frame.get('operation_type', '')
            behavior_category = frame.get('behavior_category', '')
            description = frame.get('description', '')
            
            # 提取时间部分
            try:
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                else:
                    # 格式如 "2025-12-28 18:41:54"
                    time_str = timestamp.split()[1] if ' ' in timestamp else timestamp
            except:
                time_str = timestamp
            
            frame_entry = f"""[帧{i}] {time_str}"""
            if app_name:
                frame_entry += f" | 应用: {app_name}"
            if operation_type:
                frame_entry += f" | 操作: {operation_type}"
            if behavior_category:
                frame_entry += f" | 类别: {behavior_category}"
            frame_entry += f"\n   {description}"
            
            formatted_frames.append(frame_entry)
        
        return "\n\n".join(formatted_frames)
    
    # ==================== 便捷方法 ====================
    
    @staticmethod
    def get_messages(logs: List[Dict], video_frames: List[Dict]) -> List[Dict]:
        """
        获取完整的消息列表，用于LLM API调用
        
        Returns:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]
        """
        return [
            {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.build_user_prompt(logs, video_frames)}
        ]


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 测试数据
    test_logs = [
        {
            "timestamp": "2026-01-10T10:00:00.000",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/机密工资表.xlsx",
            "process_info": {"process_name": "Excel", "pid": "1234"},
            "description": "用户使用Excel打开了机密工资表文件"
        },
        {
            "timestamp": "2026-01-10T10:00:15.000",
            "event_type": "clipboard_copy",
            "file_path": "",
            "process_info": {"process_name": "Excel"},
            "description": "用户从Excel复制了工资表内容到剪贴板"
        }
    ]
    
    test_frames = [
        {
            "timestamp": "2026-01-10T10:00:00.000",
            "app_name": "Excel",
            "operation_type": "文件打开",
            "behavior_category": "正常操作",
            "description": "Excel打开机密工资表，显示员工薪资数据"
        },
        {
            "timestamp": "2026-01-10T10:00:15.000",
            "app_name": "Excel",
            "operation_type": "复制数据",
            "behavior_category": "潜在风险",
            "description": "用户选中数据并Ctrl+C复制"
        }
    ]
    
    # 生成消息
    messages = PromptTemplates.get_messages(test_logs, test_frames)
    
    print("=" * 80)
    print("系统提示词:")
    print("=" * 80)
    print(messages[0]["content"][:500] + "...")
    
    print("\n" + "=" * 80)
    print("用户提示词:")
    print("=" * 80)
    print(messages[1]["content"])
