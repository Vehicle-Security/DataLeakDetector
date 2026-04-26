// 主页JavaScript - 录制控制和会话列表

const API_BASE = '/api';

// DOM元素
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const currentSessionId = document.getElementById('currentSessionId');
const elapsedTime = document.getElementById('elapsedTime');
const recentSessions = document.getElementById('recentSessions');

let statusCheckInterval = null;

// 开始录制
startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_BASE}/recording/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fps: 10 })
        });

        const data = await response.json();

        if (response.ok) {
            // 更新UI
            startBtn.disabled = true;
            stopBtn.disabled = false;
            recordingStatus.style.display = 'flex';
            currentSessionId.textContent = data.session_id;

            // 开始状态轮询
            startStatusPolling();
        } else {
            alert('启动录制失败: ' + data.error);
        }
    } catch (error) {
        alert('启动录制失败: ' + error.message);
    }
});

// 停止录制
stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_BASE}/recording/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            stopRecording();
            // 延迟刷新以等待文件整理完成
            setTimeout(() => {
                loadRecentSessions();
            }, 3000);
        } else {
            alert('停止录制失败: ' + data.error);
        }
    } catch (error) {
        alert('停止录制失败: ' + error.message);
    }
});

// 状态轮询
function startStatusPolling() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/recording/status`);
            const data = await response.json();

            if (data.is_recording) {
                elapsedTime.textContent = data.elapsed_seconds;
            } else {
                stopRecording();
                loadRecentSessions();
            }
        } catch (error) {
            console.error('状态检查失败:', error);
        }
    }, 1000);
}

function stopRecording() {
    startBtn.disabled = false;
    stopBtn.disabled = true;
    recordingStatus.style.display = 'none';
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

// 加载最近的会话
async function loadRecentSessions() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        const sessions = await response.json();

        recentSessions.innerHTML = '';

        if (sessions.length === 0) {
            recentSessions.innerHTML = '<div class="loading">暂无录制会话</div>';
            return;
        }

        sessions.slice(0, 6).forEach(session => {
            const card = createSessionCard(session);
            recentSessions.appendChild(card);
        });
    } catch (error) {
        recentSessions.innerHTML = '<div class="loading">加载失败</div>';
    }
}

function createSessionCard(session) {
    const card = document.createElement('div');
    card.className = 'session-card';
    card.onclick = () => window.location.href = `/session/${session.id}`;

    const apps = session.apps || {};
    const appList = Object.keys(apps).slice(0, 3).join(', ') || '无';

    card.innerHTML = `
        <div class="session-id">Session ${session.id}</div>
        <div class="session-meta">${session.start_time || '未知时间'}</div>
        <div class="session-stats">
            <span class="badge">${session.total_events} 个事件</span>
            <span class="badge">应用: ${appList}</span>
        </div>
    `;

    return card;
}

// 初始化
loadRecentSessions();
setInterval(loadRecentSessions, 30000); // 每30秒刷新
