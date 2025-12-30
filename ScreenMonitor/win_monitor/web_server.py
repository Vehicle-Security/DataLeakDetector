# -*- coding: utf-8 -*-
"""
Web服务器 - 为文件监控录制系统提供Web界面
直接控制录制，避免进程终止问题
"""
import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.screen_recorder import ScreenRecorder
from core.key_log_extractor import KeyLogExtractor

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
CORS(app)  # 启用CORS以支持跨域请求

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# 全局变量：录制相关
recorder = None
extractor = None
recording_session_id = None
recording_start_time = None
recording_thread = None
monitor_process = None
session_dir = None
video_path = None


@app.route('/')
def index():
    """主页"""
    return send_from_directory('web', 'index.html')


@app.route('/sessions')
def sessions_page():
    """Session列表页"""
    return send_from_directory('web', 'sessions.html')


@app.route('/session/<session_id>')
def session_detail(session_id):
    """Session详情页"""
    return send_from_directory('web', 'session_detail.html')


# ==================== API路由 ====================

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """启动录制 - 直接在服务器内控制"""
    global recorder, recording_session_id, recording_start_time, recording_thread
    global monitor_process, session_dir, video_path, extractor
    
    if recorder and getattr(recorder, 'is_recording_flag', False):
        return jsonify({"error": "Recording already in progress"}), 400
    
    data = request.json or {}
    fps = data.get('fps', 10)
    
    # 生成session ID
    recording_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_start_time = datetime.now()
    
    # 创建会话目录
    session_dir = OUTPUT_DIR / f"session_{recording_session_id}"
    video_dir = session_dir / "video"
    logs_dir = session_dir / "logs"
    key_events_dir = session_dir / "key_events"
    
    for directory in [video_dir, logs_dir, key_events_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 初始化录制器和提取器
    try:
        config = {}
        config_path = PROJECT_ROOT / "config_recording.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        recorder = ScreenRecorder(config)
        extractor = KeyLogExtractor(config)
        
        # 启动文件监控（在后台）
        monitor_process = start_monitor()
        
        # 启动录制
        video_filename = f"recording_{recording_session_id}.mp4"
        video_path = recorder.start_recording(
            output_dir=str(video_dir),
            fps=fps,
            filename=video_filename
        )
        
        print(f"[WEB] Recording started: {recording_session_id}")
        
        return jsonify({
            "status": "recording",
            "session_id": recording_session_id,
            "manual_mode": True,
            "fps": fps,
            "start_time": recording_start_time.isoformat()
        })
    except Exception as e:
        print(f"[WEB]Error starting recording: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """停止录制 - 直接控制，确保视频正确保存"""
    global recorder, recording_session_id, monitor_process, session_dir, extractor
    
    if not recorder or not getattr(recorder, 'is_recording_flag', False):
        return jsonify({"error": "No recording in progress"}), 400
    
    try:
        print(f"[WEB] Stopping recording: {recording_session_id}")
        
        
        print("[WEB] Stopping screen recorder...")
        recorder.stop_recording()
        print("[WEB] Screen recorder stopped successfully")
        
        # 2. 停止文件监控
        print("[WEB] Stopping file monitor...")
        stop_monitor(monitor_process)
        
        # 3. 整理输出文件
        print("[WEB] Organizing output files...")
        organize_outputs(recording_session_id, session_dir, extractor)
        
        # 4. 创建索引文件
        print("[WEB] Creating index...")
        create_index_file(recording_session_id, session_dir)
        
        print(f"[WEB] Recording {recording_session_id} completed successfully")
        
        return jsonify({
            "status": "stopped",
            "session_id": recording_session_id
        })
    except Exception as e:
        print(f"[WEB] Error stopping recording: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        recorder = None
        monitor_process = None


@app.route('/api/recording/status')
def recording_status():
    """获取录制状态"""
    global recorder, recording_session_id, recording_start_time
    
    is_recording = recorder and getattr(recorder, 'is_recording_flag', False)
    
    if is_recording and recording_start_time:
        elapsed = (datetime.now() - recording_start_time).total_seconds()
    else:
        elapsed = 0
    
    return jsonify({
        "is_recording": is_recording,
        "session_id": recording_session_id if is_recording else None,
        "elapsed_seconds": int(elapsed)
    })


# ==================== 辅助函数 ====================

def start_monitor():
    """启动文件监控"""
    main_script = PROJECT_ROOT / "main.py"
    if not main_script.exists():
        print("[WEB] Monitor script not found, skipping")
        return None
    
    try:
        print(f"[WEB] 启动监控进程...")
        process = subprocess.Popen(
            [sys.executable, str(main_script)],
            # 不重定向输出，让调试信息直接显示在控制台
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        print(f"[WEB] Monitor started (PID: {process.pid})")
        time.sleep(2)  # 等待监控初始化
        return process
    except Exception as e:
        print(f"[WEB] Failed to start monitor: {e}")
        return None


def stop_monitor(process):
    """停止文件监控"""
    if process:
        try:
            import signal
            if sys.platform == 'win32':
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
            print("[WEB] Monitor stopped")
        except Exception as e:
            print(f"[WEB] Force terminating monitor: {e}")
            process.kill()



def organize_outputs(session_id, session_dir, extractor):
    """整理输出文件"""
    try:
        # 查找监控日志
        logs_dir = PROJECT_ROOT / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("monitor_*.json"))
            if log_files:
                # 使用最新的日志
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                
                # 复制到session目录
                import shutil
                dest_log = session_dir / "logs" / f"monitor_{session_id}.json"
                shutil.copy2(latest_log, dest_log)
                print(f"[WEB] Log copied: {dest_log.name}")
                
                # 提取关键事件
                events = extractor.extract_key_events(str(latest_log))
                events_file = session_dir / "key_events" / f"key_events_{session_id}.json"
                with open(events_file, 'w', encoding='utf-8') as f:
                    json.dump(events, f, indent=2, ensure_ascii=False)
                print(f"[WEB] Extracted {len(events)} events")
                
                # 生成摘要
                summary = extractor.generate_summary(events)
                summary_file = session_dir / "key_events" / f"summary_{session_id}.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"[WEB] Summary created")
    except Exception as e:
        print(f"[WEB] Error organizing outputs: {e}")


def create_index_file(session_id, session_dir):
    """创建索引文件"""
    try:
        content = f"""# Recording Session Index

**Session ID**: {session_id}  
**Recording Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  

## File List

### Video Files
- `video/recording_{session_id}.mp4` - Recorded screen video

### Original Logs
- `logs/monitor_{session_id}.json` - Complete monitoring log

### Key Events
- `key_events/key_events_{session_id}.json` - Extracted key events
- `key_events/summary_{session_id}.json` - Event statistics summary

---
*Auto-generated by Web Recorder*
"""
        index_path = session_dir / "INDEX.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[WEB] Index created")
    except Exception as e:
        print(f"[WEB] Error creating index: {e}")


# ==================== Session API ====================

@app.route('/api/sessions')
def get_sessions():
    """获取所有录制会话列表"""
    if not OUTPUT_DIR.exists():
        return jsonify([])
    
    sessions = []
    for session_dir in sorted(OUTPUT_DIR.glob("session_*"), reverse=True):
        try:
            session_id = session_dir.name.replace("session_", "")
            
            # 读取summary文件
            summary_file = session_dir / "key_events" / f"summary_{session_id}.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
            else:
                summary = {}
            
            # 读取INDEX文件获取录制时间
            index_file = session_dir / "INDEX.md"
            start_time = None
            if index_file.exists():
                content = index_file.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    if '录制时间' in line or 'Recording Time' in line:
                        parts = line.split(': ')
                        if len(parts) > 1:
                            start_time = parts[1].strip()
                        break
            
            sessions.append({
                "id": session_id,
                "start_time": start_time or summary.get("time_range", {}).get("start"),
                "total_events": summary.get("total_events", 0),
                "apps": summary.get("apps", {}),
                "upload_count": summary.get("upload_count", 0)
            })
        except Exception as e:
            print(f"[WEB] Error reading session {session_dir}: {e}")
            continue
    
    return jsonify(sessions)


@app.route('/api/sessions/<session_id>')
def get_session_detail(session_id):
    """获取单个会话详情"""
    session_dir = OUTPUT_DIR / f"session_{session_id}"
    
    if not session_dir.exists():
        return jsonify({"error": "Session not found"}), 404
    
    try:
        # 读取关键事件
        events_file = session_dir / "key_events" / f"key_events_{session_id}.json"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
        else:
            events = []
        
        # 读取摘要
        summary_file = session_dir / "key_events" / f"summary_{session_id}.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # 检查视频文件
        video_file = session_dir / "video" / f"recording_{session_id}.mp4"
        has_video = video_file.exists()
        
        return jsonify({
            "id": session_id,
            "events": events,
            "summary": summary,
            "has_video": has_video,
            "video_url": f"/api/sessions/{session_id}/video" if has_video else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sessions/<session_id>/video')
def get_session_video(session_id):
    """获取会话视频文件"""
    video_file = OUTPUT_DIR / f"session_{session_id}" / "video" / f"recording_{session_id}.mp4"
    
    if not video_file.exists():
        return jsonify({"error": "Video not found"}), 404
    
    return send_file(video_file, mimetype='video/mp4')


@app.route('/api/sessions/<session_id>/events')
def get_session_events(session_id):
    """获取会话事件（支持筛选）"""
    session_dir = OUTPUT_DIR / f"session_{session_id}"
    events_file = session_dir / "key_events" / f"key_events_{session_id}.json"
    
    if not events_file.exists():
        return jsonify([])
    
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    # 筛选参数
    app_filter = request.args.get('app')
    file_type_filter = request.args.get('file_type')
    search_query = request.args.get('q', '').lower()
    
    filtered_events = []
    for event in events:
        # 应用筛选
        if app_filter and event.get('app_name') != app_filter:
            continue
        
        # 文件类型筛选
        if file_type_filter and event.get('file_extension') != file_type_filter:
            continue
        
        # 搜索查询
        if search_query:
            searchable = f"{event.get('file_name', '')} {event.get('app_name', '')} ".lower()
            if search_query not in searchable:
                continue
        
        filtered_events.append(event)
    
    return jsonify(filtered_events)


@app.route('/api/sessions/<session_id>/download')
def download_session_events(session_id):
    """下载会话事件JSON文件"""
    events_file = OUTPUT_DIR / f"session_{session_id}" / "key_events" / f"key_events_{session_id}.json"
    
    if not events_file.exists():
        return jsonify({"error": "Events file not found"}), 404
    
    return send_file(events_file, as_attachment=True, download_name=f"events_{session_id}.json")


def main():
    """启动Web服务器"""
    print("=" * 80)
    print(" 文件监控录制系统 - Web界面 (直接控制版)")
    print("=" * 80)
    print()
    print(f" 项目目录: {PROJECT_ROOT}")
    print(f" 输出目录: {OUTPUT_DIR}")
    print()
    print("=" * 80)
    print(" 服务器启动中...")
    print("=" * 80)
    print()
    print(" 访问地址: http://localhost:5000")
    print()
    print("按 Ctrl+C 停止服务器")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
