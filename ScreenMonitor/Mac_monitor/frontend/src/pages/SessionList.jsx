import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Calendar, Clock, AlertTriangle, FileText, ChevronRight, ChevronLeft } from 'lucide-react';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

const SessionList = () => {
    const navigate = useNavigate();
    const [sessions, setSessions] = useState([]);
    const [filteredSessions, setFilteredSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [page, setPage] = useState(1);
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');

    const ITEMS_PER_PAGE = 10;

    useEffect(() => {
        fetchSessions();
    }, []);

    useEffect(() => {
        filterSessions();
    }, [sessions, startDate, endDate]);

    const fetchSessions = async () => {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            if (data.success) {
                // Sort by ID (timestamp) descending
                const sorted = (data.data || []).sort((a, b) => b.id.localeCompare(a.id));
                setSessions(sorted);
            }
        } catch (error) {
            console.error('Failed to fetch sessions:', error);
        } finally {
            setLoading(false);
        }
    };

    const filterSessions = () => {
        let filtered = [...sessions];
        if (startDate) {
            filtered = filtered.filter(s => s.start_time >= startDate);
        }
        if (endDate) {
            // Include the end date fully
            const nextDay = new Date(endDate);
            nextDay.setDate(nextDay.getDate() + 1);
            const nextDayStr = nextDay.toISOString().split('T')[0];
            filtered = filtered.filter(s => s.start_time < nextDayStr);
        }
        setFilteredSessions(filtered);
        setPage(1); // Reset to first page
    };

    const totalPages = Math.ceil(filteredSessions.length / ITEMS_PER_PAGE);
    const paginatedSessions = filteredSessions.slice(
        (page - 1) * ITEMS_PER_PAGE,
        page * ITEMS_PER_PAGE
    );

    const formatDuration = (seconds) => {
        if (!seconds) return '0s';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    };

    return (
        <div className="p-6 max-w-7xl mx-auto">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        监控会话记录
                    </h1>
                    <p className="text-gray-400 mt-2">查看所有的录屏与行为审计日志</p>
                </div>
                <div className="flex gap-4 items-end bg-gray-800 p-4 rounded-xl border border-gray-700">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">开始日期</label>
                        <input
                            type="date"
                            className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">结束日期</label>
                        <input
                            type="date"
                            className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                        />
                    </div>
                    <button
                        onClick={() => { setStartDate(''); setEndDate(''); }}
                        className="text-sm text-gray-400 hover:text-white pb-1"
                    >
                        重置
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-20 text-gray-500">加载中...</div>
            ) : filteredSessions.length === 0 ? (
                <div className="text-center py-20 bg-gray-800 rounded-xl border border-gray-700">
                    <p className="text-gray-400">没有找到匹配的记录</p>
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
                            {paginatedSessions.map((session) => (
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

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex justify-center items-center gap-4 mt-8">
                    <button
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        disabled={page === 1}
                        className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 disabled:opacity-50 hover:bg-gray-700 disabled:hover:bg-gray-800 transition-colors"
                    >
                        <ChevronLeft size={20} />
                    </button>
                    <span className="text-gray-400 text-sm">
                        Page <span className="text-white font-medium">{page}</span> of {totalPages}
                    </span>
                    <button
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                        disabled={page === totalPages}
                        className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 disabled:opacity-50 hover:bg-gray-700 disabled:hover:bg-gray-800 transition-colors"
                    >
                        <ChevronRight size={20} />
                    </button>
                </div>
            )}
        </div>
    );
};

export default SessionList;
