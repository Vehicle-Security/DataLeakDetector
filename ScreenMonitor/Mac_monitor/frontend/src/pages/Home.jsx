import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Square, Calendar, Clock, AlertTriangle, FileText, ChevronRight, ChevronLeft } from 'lucide-react';
import { format } from 'date-fns';

const Home = () => {
    const navigate = useNavigate();
    const [isRecording, setIsRecording] = useState(false);
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Status and Sessions
    useEffect(() => {
        fetchStatus();
        fetchSessions();
        const interval = setInterval(() => {
            fetchStatus();
            if (isRecording) fetchSessions();
        }, 3000);
        return () => clearInterval(interval);
    }, [isRecording]);

    const fetchStatus = async () => {
        try {
            const res = await fetch('/api/recording/status');
            const data = await res.json();
            if (data.success) {
                setIsRecording(data.data.recording);
            }
        } catch (err) {
            console.error('Failed to fetch status:', err);
        }
    };

    const fetchSessions = async () => {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            if (data.success) {
                const sorted = (data.data || []).sort((a, b) => b.id.localeCompare(a.id));
                setSessions(sorted);
            }
        } catch (err) {
            console.error('Failed to fetch sessions:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleStart = async () => {
        try {
            const res = await fetch('/api/recording/start', { method: 'POST', body: JSON.stringify({ fps: 10 }) });
            const data = await res.json();
            if (data.success) {
                setIsRecording(true);
                fetchSessions();
            } else {
                setError(data.message);
            }
        } catch (err) {
            setError('Failed to start recording');
        }
    };

    const handleStop = async () => {
        try {
            const res = await fetch('/api/recording/stop', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                setIsRecording(false);
                fetchSessions();
            } else {
                setError(data.message);
            }
        } catch (err) {
            setError('Failed to stop recording');
        }
    };

    const formatDuration = (seconds) => {
        if (!seconds) return '0s';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    };

    return (
        <div className="p-6 max-w-7xl mx-auto">
            {/* Header / Hero */}
            <div className="flex flex-col items-center justify-center py-12 mb-12 bg-gradient-to-b from-[#161b22] to-[#0f111a] rounded-2xl border border-gray-800 shadow-xl">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-4">
                    macOS 数据泄露行为监控系统
                </h1>
                <p className="text-gray-400 mb-8">
                    实时监控敏感文件操作、应用切换、剪贴板使用
                </p>

                <div className="flex items-center gap-6">
                    <button
                        onClick={handleStart}
                        disabled={isRecording}
                        className={`group relative flex items-center justify-center gap-3 px-8 py-4 rounded-full font-bold text-lg transition-all ${isRecording
                                ? 'bg-gray-800 text-gray-500 cursor-not-allowed opacity-50'
                                : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg shadow-green-900/20 hover:shadow-green-900/40 hover:scale-105 active:scale-95'
                            }`}
                    >
                        {isRecording ? (
                            <>
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                                </span>
                                正在录制...
                            </>
                        ) : (
                            <>
                                <Play fill="currentColor" size={24} />
                                开始监控
                            </>
                        )}
                    </button>

                    <button
                        onClick={handleStop}
                        disabled={!isRecording}
                        className={`flex items-center justify-center gap-3 px-8 py-4 rounded-full font-bold text-lg transition-all ${!isRecording
                                ? 'bg-gray-800 text-gray-500 cursor-not-allowed opacity-50'
                                : 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-900/20 hover:shadow-red-900/40 hover:scale-105 active:scale-95'
                            }`}
                    >
                        <Square fill="currentColor" size={24} />
                        停止监控
                    </button>
                </div>

                {error && (
                    <div className="mt-4 text-red-400 bg-red-900/20 px-4 py-2 rounded-lg border border-red-900/50">
                        {error}
                    </div>
                )}
            </div>

            {/* Session List */}
            <div className="mb-6">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Clock className="text-blue-400" /> Cloud Recordings
                </h2>
            </div>

            {loading ? (
                <div className="text-center py-20 text-gray-500">加载中...</div>
            ) : sessions.length === 0 ? (
                <div className="text-center py-20 bg-gray-800 rounded-xl border border-gray-700">
                    <p className="text-gray-400">暂无录制记录</p>
                </div>
            ) : (
                <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden shadow-xl">
                    <table className="w-full text-left">
                        <thead className="bg-gray-900/50 text-gray-400 text-sm">
                            <tr>
                                <th className="px-6 py-4 font-medium">会话 ID</th>
                                <th className="px-6 py-4 font-medium">开始时间</th>
                                <th className="px-6 py-4 font-medium">时长</th>
                                <th className="px-6 py-4 font-medium">风险事件</th>
                                <th className="px-6 py-4 font-medium">状态</th>
                                <th className="px-6 py-4 font-medium text-right">操作</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-700">
                            {sessions.map((session) => (
                                <tr
                                    key={session.id}
                                    className="hover:bg-gray-700/50 transition-colors group cursor-pointer"
                                    onClick={() => navigate(`/session/${session.id}`)}
                                >
                                    <td className="px-6 py-4 font-mono text-blue-400 text-sm">
                                        {session.id}
                                    </td>
                                    <td className="px-6 py-4 text-gray-300">
                                        <div className="flex items-center gap-2">
                                            <Calendar size={14} className="text-gray-500" />
                                            {format(new Date(session.start_time), 'yyyy-MM-dd HH:mm:ss')}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-300">
                                        <div className="flex items-center gap-2">
                                            <Clock size={14} className="text-gray-500" />
                                            {formatDuration(session.duration)}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        {session.risk_events > 0 ? (
                                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-900/30 text-red-400 border border-red-900/50">
                                                <AlertTriangle size={12} />
                                                {session.risk_events}
                                            </span>
                                        ) : (
                                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-900/30 text-green-400 border border-green-900/50">
                                                <FileText size={12} />
                                                Safe
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-6 py-4">
                                        {session.status === 'recording' ? (
                                            <span className="flex items-center gap-2 text-yellow-400 animate-pulse">
                                                <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                                                录制中
                                            </span>
                                        ) : (
                                            <span className="text-gray-400">已完成</span>
                                        )}
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button className="text-gray-400 group-hover:text-blue-400 transition-colors">
                                            <ChevronRight size={20} />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default Home;
