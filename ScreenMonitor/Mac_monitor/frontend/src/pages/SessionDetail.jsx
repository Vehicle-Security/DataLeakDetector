import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Search, Download, Clock, Video, FileText, AlertTriangle } from 'lucide-react';

const SessionDetail = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [session, setSession] = useState(null);
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [error, setError] = useState(null);
    const videoRef = useRef(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch session info
                const sessionRes = await fetch(`/api/sessions/${id}`);
                const sessionData = await sessionRes.json();

                if (sessionData.success) {
                    setSession(sessionData.data);
                } else {
                    setError(sessionData.message);
                }

                // Fetch key events (comprehensive file operation logs)
                const keyEventsRes = await fetch(`/api/key-events/${id}`);
                if (keyEventsRes.ok) {
                    // The API returns raw JSON array, not a wrapped Response object
                    const keyEvents = await keyEventsRes.json();
                    setLogs(keyEvents || []);
                }

            } catch (err) {
                console.error("Failed to load session details", err);
                setError("无法加载会话详情");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [id]);

    const filteredLogs = logs.filter(log => {
        if (!searchTerm) return true;
        const term = searchTerm.toLowerCase();
        return (
            (log.file_path && log.file_path.toLowerCase().includes(term)) ||
            (log.process_info?.process_name && log.process_info.process_name.toLowerCase().includes(term)) ||
            (log.event_type && log.event_type.toLowerCase().includes(term)) ||
            (log.file_name && log.file_name.toLowerCase().includes(term))
        );
    });

    const formatTime = (ts) => {
        try {
            return new Date(ts).toLocaleTimeString('zh-CN', { hour12: false });
        } catch (e) {
            return ts;
        }
    };

    if (loading) return <div className="text-center py-20 text-gray-500">加载中...</div>;
    if (error) return <div className="text-center py-20 text-red-500">{error}</div>;
    if (!session) return <div className="text-center py-20 text-gray-500">会话不存在</div>;

    return (
        <div className="h-screen flex flex-col bg-[#0f111a] text-gray-300">
            {/* Header */}
            <div className="h-16 border-b border-gray-800 bg-[#161b22] px-6 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/')}
                        className="p-2 hover:bg-gray-800 rounded-full transition-colors text-gray-400 hover:text-white"
                    >
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <h1 className="font-semibold text-white flex items-center gap-2">
                            Session <span className="font-mono text-blue-400">{id}</span>
                        </h1>
                        <p className="text-xs text-gray-500">
                            {new Date(session.start_time).toLocaleString()} • {session.duration?.toFixed(0)}s
                        </p>
                    </div>
                </div>
            </div>

            {/* Content Grid */}
            <div className="flex-1 flex overflow-hidden">
                {/* Left Panel: Video */}
                <div className="w-2/3 flex flex-col border-r border-gray-800 bg-black">
                    <div className="flex-1 flex items-center justify-center p-4">
                        {session.full_video_path ? (
                            <div className="relative w-full max-h-full aspect-video bg-[#0a0a0a] rounded-lg overflow-hidden border border-gray-800 shadow-2xl">
                                <video
                                    ref={videoRef}
                                    src={session.full_video_path}
                                    controls
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        ) : (
                            <div className="text-gray-500 flex flex-col items-center">
                                <Video size={48} className="mb-4 opacity-50" />
                                <p>未找到录屏文件</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel: Logs */}
                <div className="w-1/3 flex flex-col bg-[#11141d]">
                    <div className="h-14 border-b border-gray-800 px-4 flex items-center gap-3 shrink-0">
                        <Search size={16} className="text-gray-500" />
                        <input
                            type="text"
                            placeholder="搜索日志 (进程, 文件, 操作)..."
                            className="bg-transparent border-none focus:outline-none text-sm w-full text-white placeholder-gray-600"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>

                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        <table className="w-full text-left text-xs">
                            <thead className="bg-[#1a1f2e] text-gray-400 sticky top-0">
                                <tr>
                                    <th className="px-4 py-2 font-medium w-20">时间</th>
                                    <th className="px-4 py-2 font-medium w-24">进程</th>
                                    <th className="px-4 py-2 font-medium w-16">操作</th>
                                    <th className="px-4 py-2 font-medium">文件路径</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-800/50">
                                {filteredLogs.length === 0 ? (
                                    <tr>
                                        <td colSpan="4" className="px-4 py-8 text-center text-gray-500">
                                            没有找到日志
                                        </td>
                                    </tr>
                                ) : (
                                    filteredLogs.map((log, index) => (
                                        <tr
                                            key={index}
                                            className="hover:bg-gray-800/50 transition-colors group"
                                            onClick={() => {
                                                // Optional: Seek video to timestamp if possible
                                            }}
                                        >
                                            <td className="px-4 py-2.5 font-mono text-gray-500 whitespace-nowrap">
                                                {formatTime(log.timestamp)}
                                            </td>
                                            <td className="px-4 py-2.5 text-blue-400 font-medium whitespace-nowrap overflow-hidden text-ellipsis max-w-[100px]" title={log.process_info?.process_name}>
                                                {log.process_info?.process_name || 'N/A'}
                                            </td>
                                            <td className="px-4 py-2.5 text-gray-300 whitespace-nowrap">
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${log.event_type === 'deleted' || log.event_type === 'renamed' ? 'bg-red-900/30 text-red-500' :
                                                    log.event_type === 'created' ? 'bg-green-900/30 text-green-500' :
                                                        log.event_type === 'modified' ? 'bg-yellow-900/30 text-yellow-500' :
                                                            log.event_type === 'opened' ? 'bg-blue-900/30 text-blue-400' :
                                                                'bg-gray-800 text-gray-400'
                                                    }`}>
                                                    {log.event_type || 'unknown'}
                                                </span>
                                            </td>
                                            <td className="px-4 py-2.5 text-gray-400 break-all leading-tight font-mono text-[11px]" title={log.file_path}>
                                                {log.file_path || log.file_name}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>

                    <div className="h-8 border-t border-gray-800 px-4 flex items-center justify-between text-[11px] text-gray-500 bg-[#161b22] shrink-0">
                        <span>{filteredLogs.length} events</span>
                        <span>Session Log</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SessionDetail;
