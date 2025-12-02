"""
配置文件
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 记忆数据库路径
MEMORY_DB_PATH = str(DATA_DIR / "memory.json")

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = str(LOG_DIR / "agent.log")

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "[%(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": LOG_FILE,
            "mode": "a",
            "encoding": "utf-8"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

# Agent配置
AGENT_CONFIG = {
    # 事件总线配置
    "event_bus": {
        "max_history": 10000  # 最大事件历史记录数
    },
    
    # 记忆系统配置
    "memory": {
        "db_path": MEMORY_DB_PATH,
        "auto_save_interval": 300  # 自动保存间隔（秒）
    },
    
    # N2战术手册引擎配置
    "n2_engine": {
        "playbook_dir": str(PROJECT_ROOT / "playbooks"),  # 剧本目录
        "enable": True  # 是否启用N2引擎
    },
    
    # N3侦探引擎配置
    "n3_engine": {
        "max_iterations": 10,  # 单个侦查的最大迭代次数
        "max_concurrent_investigations": 5,  # 最大并发侦查数
        "llm_model": "qwen2.5-72b-instruct",  # LLM模型名称（可选：qwen-turbo/qwen-plus/qwen-max/qwen2.5-72b-instruct）
        "llm_api_key": os.getenv("LLM_API_KEY", "")  # LLM API密钥
    },
    
    # 污点追踪配置
    "taint_tracker": {
        "enable": True,  # 是否启用自动污点追踪
        "time_window": 10,  # 事件关联时间窗口（秒）
        "max_recent_events": 100  # 最大最近事件缓存
    },
    
    # 工具箱配置
    "toolbox": {
        "video_search_engine": "../deformed_image_search"  # 视频搜索引擎路径
    }
}

# 敏感数据源配置
SENSITIVE_SOURCES = [
    "database_query",
    "confidential_file_open",
    "secure_storage_access",
    "credential_access"
]

# 已知的跨模态转换模式
MODAL_CONVERSIONS = {
    ("text", "audio"): ["text_to_speech", "tts", "audio_convert", "speech_synthesis"],
    ("text", "image"): ["screenshot", "text_to_image", "render", "ocr_reverse"],
    ("document", "pdf"): ["pdf_export", "print_to_pdf", "save_as_pdf"],
    ("video", "images"): ["video_to_frames", "frame_extract", "snapshot"],
    ("audio", "text"): ["speech_to_text", "transcribe", "voice_recognition"]
}

# 告警配置
ALERT_CONFIG = {
    "enable_siem": False,  # 是否发送到SIEM
    "siem_endpoint": "http://localhost:5144",
    "enable_email": False,  # 是否发送邮件告警
    "email_recipients": [],
    "alert_threshold": "medium"  # 告警阈值：low, medium, high
}
