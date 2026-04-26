// Session详情页JavaScript - 视频播放和事件时间线

const API_BASE = '/api';
const sessionId = window.location.pathname.split('/').pop();

// DOM元素
const sessionVideo = document.getElementById('sessionVideo');
const appFilter = document.getElementById('appFilter');
const fileTypeFilter = document.getElementById('fileTypeFilter');
const searchQuery = document.getElementById('searchQuery');
const eventsTimeline = document.getElementById('eventsTimeline');
const totalEventsSpan = document.getElementById('totalEvents');
const filteredEventsSpan = document.getElementById('filteredEvents');
const downloadEvents = document.getElementById('downloadEvents');
const downloadVideo = document.getElementById('downloadVideo');

let allEvents = [];

// 加载Session详情
async function loadSessionDetail() {
    try {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        // 设置视频
        if (data.has_video) {
            sessionVideo.src = data.video_url;
        } else {
            document.querySelector('.video-section').innerHTML = '<p>该会话没有视频</p>';
        }

        // 存储事件
        allEvents = data.events;
        totalEventsSpan.textContent = allEvents.length;

        // 构建筛选选项
        buildFilters(allEvents);

        // 渲染事件
        renderEvents(allEvents);

    } catch (error) {
        alert('加载Session失败: ' + error.message);
    }
}

// 构建筛选器选项
function buildFilters(events) {
    const apps = new Set();
    const fileTypes = new Set();

    events.forEach(event => {
        if (event.app_name) apps.add(event.app_name);
        if (event.file_extension) fileTypes.add(event.file_extension);
    });

    // 应用筛选器
    apps.forEach(app => {
        const option = document.createElement('option');
        option.value = app;
        option.textContent = app;
        appFilter.appendChild(option);
    });

    // 文件类型筛选器
    fileTypes.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        fileTypeFilter.appendChild(option);
    });
}

// 渲染事件列表
function renderEvents(events) {
    eventsTimeline.innerHTML = '';
    filteredEventsSpan.textContent = events.length;

    if (events.length === 0) {
        eventsTimeline.innerHTML = '<div class="loading">没有匹配的事件</div>';
        return;
    }

    events.forEach(event => {
        const eventItem = createEventElement(event);
        eventsTimeline.appendChild(eventItem);
    });
}

// 创建事件元素
function createEventElement(event) {
    const div = document.createElement('div');
    div.className = 'event-item';

    const time = new Date(event.timestamp).toLocaleTimeString('zh-CN');
    const fullTime = new Date(event.timestamp).toLocaleString('zh-CN');

    // 应用标签 - 高亮显示
    const appBadge = event.app_name
        ? `<span class="event-app" style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;">📱 ${event.app_name}</span>`
        : '';

    // 文件大小格式化
    const fileSize = event.file_size ? formatFileSize(event.file_size) : '';
    const fileSizeBadge = fileSize ? `<span class="event-size">${fileSize}</span>` : '';

    // 事件类型标签
    const eventTypeColors = {
        'created': '#2196F3',
        'modified': '#FF9800',
        'deleted': '#F44336',
        'moved': '#9C27B0'
    };
    const eventTypeColor = eventTypeColors[event.event_type] || '#757575';
    const eventTypeBadge = `<span class="event-type" style="background: ${eventTypeColor}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">${event.event_type}</span>`;

    // 文件扩展名标签
    const extBadge = event.file_extension
        ? `<span class="event-ext" style="background: #757575; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">${event.file_extension}</span>`
        : '';

    // 检测方法（如果是ETW检测到的上传，特别标注）
    const detectionBadge = (event.detection_method === 'sliding_window_correlation' || event.upload_detection)
        ? `<span class="detection-badge" style="background: #E91E63; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">🔍 Upload Detected</span>`
        : '';

    // 完整路径（可展开）
    const filePathShort = event.file_path ? event.file_path.substring(event.file_path.lastIndexOf('\\') + 1) : event.file_name;
    const filePath = event.file_path || '';

    div.innerHTML = `
        <div class="event-header" style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div class="event-time" style="font-weight: bold; color: #666;" title="${fullTime}">${time}</div>
            ${eventTypeBadge}
            ${extBadge}
            ${appBadge}
            ${detectionBadge}
            ${fileSizeBadge}
        </div>
        <div class="event-body">
            <div class="event-file" style="font-size: 1.1em; font-weight: 500;">📄 ${event.file_name}</div>
            ${filePath ? `<div class="event-path" style="font-size: 0.9em; color: #888; margin-top: 4px; cursor: pointer;" title="点击复制路径">${filePath}</div>` : ''}
        </div>
    `;

    // 点击路径复制
    const pathEl = div.querySelector('.event-path');
    if (pathEl) {
        pathEl.onclick = (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(filePath).then(() => {
                pathEl.style.color = '#4CAF50';
                setTimeout(() => pathEl.style.color = '#888', 1000);
            });
        };
    }

    // 点击事件展开更多详情
    div.onclick = () => {
        const existing = div.querySelector('.event-details');
        if (existing) {
            existing.remove();
            return;
        }

        const details = document.createElement('div');
        details.className = 'event-details';
        details.style.cssText = 'margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 4px; font-size: 0.9em;';

        let detailsHTML = '<div style="font-weight: bold; margin-bottom: 6px;">📋 详细信息:</div>';

        if (event.window_info && event.window_info.window_title) {
            detailsHTML += `<div><strong>窗口标题:</strong> ${event.window_info.window_title}</div>`;
        }

        if (event.process_info && event.process_info.process_name) {
            detailsHTML += `<div><strong>进程名:</strong> ${event.process_info.process_name}</div>`;
        }

        if (event.process_info && event.process_info.process_path) {
            detailsHTML += `<div><strong>进程路径:</strong> ${event.process_info.process_path}</div>`;
        }

        if (event.user_info) {
            detailsHTML += `<div><strong>用户:</strong> ${event.user_info.username}@${event.user_info.hostname}</div>`;
        }

        details.innerHTML = detailsHTML;
        div.appendChild(details);
    };

    return div;
}

// 文件大小格式化
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 筛选事件
function filterEvents() {
    const app = appFilter.value;
    const fileType = fileTypeFilter.value;
    const search = searchQuery.value.toLowerCase();

    const filtered = allEvents.filter(event => {
        if (app && event.app_name !== app) return false;
        if (fileType && event.file_extension !== fileType) return false;
        if (search && !event.file_name.toLowerCase().includes(search)) return false;
        return true;
    });

    renderEvents(filtered);
}

// 事件监听器
appFilter.addEventListener('change', filterEvents);
fileTypeFilter.addEventListener('change', filterEvents);
searchQuery.addEventListener('input', filterEvents);

downloadEvents.addEventListener('click', () => {
    window.location.href = `${API_BASE}/sessions/${sessionId}/download`;
});

downloadVideo.addEventListener('click', () => {
    window.location.href = `${API_BASE}/sessions/${sessionId}/video`;
});

// 视频倍速控制
sessionVideo.addEventListener('loadedmetadata', () => {
    // 添加倍速控制
    sessionVideo.playbackRate = 1.0;

    // 可以添加倍速选择器
    const controls = document.querySelector('.video-section');
    const speedControl = document.createElement('div');
    speedControl.style.marginTop = '1rem';
    speedControl.innerHTML = `
        <label>播放速度: </label>
        <select id="speedSelector">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
        </select>
    `;
    controls.appendChild(speedControl);

    document.getElementById('speedSelector').addEventListener('change', (e) => {
        sessionVideo.playbackRate = parseFloat(e.target.value);
    });
});

// 初始化
loadSessionDetail();
