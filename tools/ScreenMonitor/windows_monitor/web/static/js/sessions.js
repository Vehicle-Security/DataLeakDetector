// Sessions列表页JavaScript

const API_BASE = '/api';

const sessionsList = document.getElementById('sessionsList');
const searchInput = document.getElementById('searchInput');
const sortBy = document.getElementById('sortBy');

let allSessions = [];

// 加载所有会话
async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        allSessions = await response.json();
        renderSessions(allSessions);
    } catch (error) {
        sessionsList.innerHTML = '<div class="loading">加载失败</div>';
    }
}

// 渲染会话列表
function renderSessions(sessions) {
    sessionsList.innerHTML = '';

    if (sessions.length === 0) {
        sessionsList.innerHTML = '<div class="loading">暂无录制会话</div>';
        return;
    }

    sessions.forEach(session => {
        const card = createSessionCard(session);
        sessionsList.appendChild(card);
    });
}

// 创建会话卡片
function createSessionCard(session) {
    const card = document.createElement('div');
    card.className = 'session-card';
    card.onclick = () => window.location.href = `/session/${session.id}`;

    const apps = session.apps || {};
    const appList = Object.keys(apps).map(app => `${app} (${apps[app]})`).join(', ') || '无';

    card.innerHTML = `
        <div class="session-id">Session ${session.id}</div>
        <div class="session-meta">${session.start_time || '未知时间'}</div>
        <div class="session-stats">
            <span class="badge">${session.total_events} 个事件</span>
            <span class="badge">上传: ${session.upload_count}</span>
            <span class="badge">应用: ${appList}</span>
        </div>
    `;

    return card;
}

// 搜索过滤
searchInput.addEventListener('input', () => {
    const query = searchInput.value.toLowerCase();
    const filtered = allSessions.filter(s =>
        s.id.toLowerCase().includes(query)
    );
    renderSessions(filtered);
});

// 排序
sortBy.addEventListener('change', () => {
    const sorted = [...allSessions];
    if (sortBy.value === 'date') {
        sorted.sort((a, b) => b.id.localeCompare(a.id));
    } else if (sortBy.value === 'events') {
        sorted.sort((a, b) => b.total_events - a.total_events);
    }
    allSessions = sorted;
    renderSessions(sorted);
});

// 初始化
loadSessions();
