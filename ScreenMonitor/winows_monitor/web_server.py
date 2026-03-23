# -*- coding: utf-8 -*-
"""
web_server.py - Win Monitor Web 控制界面
提供 RESTful API 和 Web UI 来控制监控引擎

API 端点:
- GET  /              : 主页（Web UI）
- GET  /sessions      : 会话列表页
- GET  /session/<id>  : 会话详情页
- POST /api/start     : 启动监控
- POST /api/stop      : 停止监控
- GET  /api/status    : 获取状态
- GET  /api/logs      : 获取最近日志
- GET  /api/sessions  : 获取会话列表
- GET  /api/sessions/<id> : 获取会话详情
- GET  /api/sessions/<id>/events : 获取会话事件
"""

import os
import sys
import json
import re
import threading
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_from_directory, Response
from flask_cors import CORS
import logging

# 禁用 Flask/Werkzeug 请求日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Ensure console logging and print() do not fail in legacy code pages.
def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


_configure_stdio()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.config_loader import ConfigLoader
from services.recorder_service import RecorderService
from core.monitors.engine import Engine, State
from core.monitors.file_system_monitor import FileSystemMonitor, get_file_system_monitor
from core.detection.rule_matcher import MatchResult


# Flask 应用
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
CORS(app)

# 全局引擎实例
_engine: Engine = None
_engine_lock = threading.Lock()

# 全局文件系统监控器
_fs_monitor: FileSystemMonitor = None

# 会话目录（支持多个位置）
SESSION_DIRS = []


def get_engine() -> Engine:
    """获取或创建引擎实例（单例）"""
    global _engine, SESSION_DIRS
    with _engine_lock:
        if _engine is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            output_dir = os.path.join(os.path.dirname(__file__), "recordings")
            
            config = ConfigLoader(config_path)
            recorder = RecorderService(fps=10)
            
            _engine = Engine(
                config_loader=config,
                recorder_service=recorder,
                output_dir=output_dir
            )
            
            # 设置会话目录
            SESSION_DIRS = [
                output_dir,
                os.path.join(os.path.dirname(__file__), "..", ""),  # ScreenMonitor 目录
            ]
        return _engine


def scan_sessions():
    """扫描所有会话目录"""
    sessions = []
    seen_ids = set()
    
    for base_dir in SESSION_DIRS:
        if not os.path.exists(base_dir):
            continue
            
        for name in os.listdir(base_dir):
            if not name.startswith("session_"):
                continue
            
            session_id = name.replace("session_", "")
            if session_id in seen_ids:
                continue
            seen_ids.add(session_id)
            
            session_path = os.path.join(base_dir, name)
            if not os.path.isdir(session_path):
                continue
            
            session = parse_session(session_id, session_path)
            if session:
                sessions.append(session)
    
    # 按 ID 倒序排列
    sessions.sort(key=lambda x: x["id"], reverse=True)
    return sessions


def parse_session(session_id: str, session_path: str) -> dict:
    """解析单个会话目录"""
    session = {
        "id": session_id,
        "path": session_path,
        "start_time": None,
        "duration": None,
        "risk_events": 0,
        "status": "completed",
        "video_path": None,
        "log_path": None,
        "events_path": None,
    }
    
    # 解析时间
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        session["start_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        session["start_time"] = session_id
    
    # 查找视频
    video_dir = os.path.join(session_path, "video")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                session["video_path"] = f"/api/video/{session_id}/{f}"
                break
    
    # 查找日志
    logs_dir = os.path.join(session_path, "logs")
    if os.path.exists(logs_dir):
        logs_json = os.path.join(logs_dir, "logs.json")
        if os.path.exists(logs_json):
            session["log_path"] = logs_json
        else:
            for f in sorted(os.listdir(logs_dir)):
                if f.endswith(".json"):
                    session["log_path"] = os.path.join(logs_dir, f)
                    break

        keyevents_json = os.path.join(logs_dir, "keyevents.json")
        if os.path.exists(keyevents_json):
            session["events_path"] = keyevents_json
            session["risk_events"] = _count_json_events(keyevents_json)

    # 兼容旧 key_events 目录
    if session["events_path"] is None:
        events_dir = os.path.join(session_path, "key_events")
        if os.path.exists(events_dir):
            for f in sorted(os.listdir(events_dir)):
                if "key_events" in f and f.endswith(".json"):
                    session["events_path"] = os.path.join(events_dir, f)
                    session["risk_events"] = _count_json_events(session["events_path"])
                    break

    session["duration"] = _read_session_duration(session_path, session_id)
    
    return session


def _count_json_events(json_path: str) -> int:
    """统计 JSON 数组文件中的事件数量。"""
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            events = json.load(handle)
            return len(events) if isinstance(events, list) else 0
    except Exception:
        return 0


def _read_session_duration(session_path: str, session_id: str):
    """优先从 INDEX.md 读取新结构时长，回退到旧 summary 文件。"""
    index_path = os.path.join(session_path, "INDEX.md")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            match = re.search(r"\*\*Duration\*\*:\s*([0-9]+(?:\.[0-9]+)?)\s+seconds", content)
            if match:
                return float(match.group(1))
        except Exception:
            pass

    summary_path = os.path.join(session_path, "key_events", f"summary_{session_id}.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
                return summary.get("duration_seconds", 0)
        except Exception:
            return None

    return None


# ====== Web UI 路由 ======

@app.route('/')
def index():
    """主页 - Web UI"""
    return render_template('index.html')


@app.route('/sessions')
def sessions_page():
    """会话列表页"""
    return render_template('sessions.html')


@app.route('/session/<session_id>')
def session_detail_page(session_id):
    """会话详情页"""
    return render_template('session_detail.html')


# ====== API 路由 ======

@app.route('/api/start', methods=['POST'])
def api_start():
    """启动监控"""
    global _fs_monitor
    engine = get_engine()
    
    if engine.running:
        return jsonify({
            "success": False,
            "message": "监控已在运行中"
        }), 400
    
    result = engine.start_monitoring()
    
    # 注意: 文件系统监控由 Engine 内部管理，无需在此启动
        

    
    return jsonify({
        "success": result,
        "message": "监控已启动 (手动持续录制模式)" if result else "启动失败"
    })


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """停止监控"""
    global _fs_monitor
    engine = get_engine()
    
    if not engine.running:
        return jsonify({
            "success": False,
            "message": "监控未在运行"
        }), 400
    
    # 注意: 文件系统监控由 Engine 内部管理
    

    
    result = engine.stop_monitoring()
    
    return jsonify({
        "success": result,
        "message": "监控已停止" if result else "停止失败"
    })


@app.route('/api/status', methods=['GET'])
def api_status():
    """获取当前状态"""
    engine = get_engine()
    status = engine.get_status()
    status["is_running"] = engine.running
    
    if not engine.running:
        status["display_state"] = "空闲"
    elif status["state"] == "recording":
        status["display_state"] = "录制中"
    else:
        status["display_state"] = "监控中"
    
    return jsonify(status)


@app.route('/api/logs', methods=['GET'])
def api_logs():
    """获取最近日志"""
    engine = get_engine()
    count = request.args.get('count', 50, type=int)
    logs = engine.get_recent_logs(count)
    
    return jsonify({
        "logs": logs,
        "total": len(logs)
    })


@app.route('/api/sessions', methods=['GET'])
def api_sessions():
    """获取会话列表"""
    get_engine()  # 确保初始化
    sessions = scan_sessions()
    
    return jsonify({
        "success": True,
        "sessions": sessions,
        "total": len(sessions)
    })


@app.route('/api/sessions/<session_id>', methods=['GET'])
def api_session_detail(session_id):
    """获取会话详情"""
    get_engine()
    sessions = scan_sessions()
    
    for s in sessions:
        if s["id"] == session_id:
            return jsonify({
                "success": True,
                "session": s
            })
    
    return jsonify({
        "success": False,
        "message": "会话不存在"
    }), 404


@app.route('/api/sessions/<session_id>/events', methods=['GET'])
def api_session_events(session_id):
    """获取会话完整日志"""
    get_engine()
    sessions = scan_sessions()
    
    for s in sessions:
        if s["id"] == session_id:
            events = []
            
            # 读取完整日志 (logs.json)
            logs_dir = os.path.join(s["path"], "logs")
            logs_file = os.path.join(logs_dir, "logs.json")
            
            if os.path.exists(logs_file):
                try:
                    with open(logs_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        # 尝试 JSON Array 格式
                        if content.startswith('['):
                            events = json.loads(content)
                        else:
                            # JSON Lines 格式
                            for line in content.split('\n'):
                                if line.strip():
                                    try:
                                        events.append(json.loads(line))
                                    except:
                                        pass
                except Exception as e:
                    print(f"[ERROR] 读取 logs.json 失败: {e}")
            
            # 如果没有 logs.json，尝试旧的日志格式
            elif s.get("log_path") and os.path.exists(s["log_path"]):
                try:
                    with open(s["log_path"], "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content.startswith('['):
                            events = json.loads(content)
                        else:
                            for line in content.split('\n'):
                                if line.strip():
                                    try:
                                        events.append(json.loads(line))
                                    except:
                                        pass
                except Exception as e:
                    print(f"[ERROR] 读取原始日志失败: {e}")
            
            return jsonify({
                "success": True,
                "events": events if isinstance(events, list) else [],
                "total": len(events) if isinstance(events, list) else 0
            })
    
    return jsonify({
        "success": False,
        "message": "会话不存在"
    }), 404


@app.route('/api/sessions/<session_id>/keyevents', methods=['GET'])
def api_session_keyevents(session_id):
    """获取会话关键事件"""
    get_engine()
    sessions = scan_sessions()
    
    for s in sessions:
        if s["id"] == session_id:
            events = []
            
            # 读取关键事件 (keyevents.json)
            logs_dir = os.path.join(s["path"], "logs")
            keyevents_file = os.path.join(logs_dir, "keyevents.json")
            
            if os.path.exists(keyevents_file):
                try:
                    with open(keyevents_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content.startswith('['):
                            events = json.loads(content)
                        else:
                            # JSON Lines 备用
                            for line in content.split('\n'):
                                if line.strip():
                                    try:
                                        events.append(json.loads(line))
                                    except:
                                        pass
                except Exception as e:
                    print(f"[ERROR] 读取 keyevents.json 失败: {e}")
            
            # 备用：读取旧的 key_events 目录
            elif s.get("events_path") and os.path.exists(s["events_path"]):
                try:
                    with open(s["events_path"], "r", encoding="utf-8") as f:
                        events = json.load(f)
                except Exception as e:
                    print(f"[ERROR] 读取 key_events 失败: {e}")
            
            return jsonify({
                "success": True,
                "events": events if isinstance(events, list) else [],
                "total": len(events) if isinstance(events, list) else 0
            })
    
    return jsonify({
        "success": False,
        "message": "会话不存在"
    }), 404


@app.route('/api/video/<session_id>/<filename>')
def api_video(session_id, filename):
    """提供视频文件"""
    get_engine()
    
    for base_dir in SESSION_DIRS:
        video_path = os.path.join(base_dir, f"session_{session_id}", "video")
        if os.path.exists(os.path.join(video_path, filename)):
            return send_from_directory(video_path, filename)
    
    return "Video not found", 404


@app.route('/api/config', methods=['GET'])
def api_config():
    """获取配置信息"""
    engine = get_engine()
    config = engine.config
    
    return jsonify({
        "blacklist_apps_count": len(config.config.blacklist_apps) if config.config else 0,
        "blacklist_websites_count": len(config.config.blacklist_websites) if config.config else 0,
        "poll_interval": engine.poll_interval,
        "buffer_time": engine.buffer_time
    })


# ====== 主入口 ======

def main():
    """主函数"""
    from core.utils import app_logger
    
    app_logger.info("=" * 60)
    app_logger.info("  Win Monitor - Web 控制界面")
    app_logger.info("=" * 60)
    
    global _engine
    _engine = get_engine()
    
    app_logger.info(f"📊 配置加载完成")
    app_logger.info(f"🌐 Web UI: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
