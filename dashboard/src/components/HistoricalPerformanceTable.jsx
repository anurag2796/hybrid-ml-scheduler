import React, { useState, useEffect } from 'react';

const HistoricalPerformanceTable = () => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchData = async () => {
        try {
            const response = await fetch(`http://localhost:8000/api/history/comparative?limit=50&t=${Date.now()}`);
            const result = await response.json();
            setData(result);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching history:', error);
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 2000); // Refresh every 2s
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="text-cyan-400">Loading history...</div>;

    return (
        <div className="bg-gray-900/80 backdrop-blur border border-cyan-500/30 rounded-lg p-4 shadow-[0_0_15px_rgba(6,182,212,0.1)]">
            <h3 className="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                Historical Performance Analysis
            </h3>

            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-gray-300">
                    <thead className="text-xs text-cyan-300 uppercase bg-gray-800/50 border-b border-cyan-500/30">
                        <tr>
                            <th className="px-4 py-3">Task ID</th>
                            <th className="px-4 py-3">Size</th>
                            <th className="px-4 py-3">Intensity</th>
                            <th className="px-4 py-3 text-center">Round Robin</th>
                            <th className="px-4 py-3 text-center">Random</th>
                            <th className="px-4 py-3 text-center">Hybrid ML</th>
                            <th className="px-4 py-3 text-center">RL Agent</th>
                            <th className="px-4 py-3 text-center">Oracle (Best)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((task) => {
                            const rr = task.results['round_robin']?.time || Infinity;
                            const rnd = task.results['random']?.time || Infinity;
                            const ml = task.results['hybrid_ml']?.time || Infinity;
                            const rl = task.results['rl_agent']?.time || Infinity;
                            const oracle = task.results['oracle']?.time || Infinity;

                            // Find winner (excluding Oracle)
                            const times = { 'Round Robin': rr, 'Random': rnd, 'Hybrid ML': ml, 'RL Agent': rl };
                            const winner = Object.entries(times).reduce((a, b) => a[1] < b[1] ? a : b)[0];

                            return (
                                <tr key={task.task_id} className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                                    <td className="px-4 py-2 font-mono text-cyan-200">#{task.task_id}</td>
                                    <td className="px-4 py-2">{task.size.toFixed(1)}</td>
                                    <td className="px-4 py-2">{task.intensity.toFixed(1)}</td>

                                    <td className={`px-4 py-2 text-center ${winner === 'Round Robin' ? 'text-green-400 font-bold' : ''}`}>
                                        {rr !== Infinity ? rr.toFixed(3) + 's' : '-'}
                                    </td>
                                    <td className={`px-4 py-2 text-center ${winner === 'Random' ? 'text-green-400 font-bold' : ''}`}>
                                        {rnd !== Infinity ? rnd.toFixed(3) + 's' : '-'}
                                    </td>
                                    <td className={`px-4 py-2 text-center ${winner === 'Hybrid ML' ? 'text-green-400 font-bold' : ''}`}>
                                        {ml !== Infinity ? ml.toFixed(3) + 's' : '-'}
                                    </td>
                                    <td className={`px-4 py-2 text-center ${winner === 'RL Agent' ? 'text-green-400 font-bold' : ''}`}>
                                        {rl !== Infinity ? rl.toFixed(3) + 's' : '-'}
                                    </td>
                                    <td className="px-4 py-2 text-center text-yellow-400 font-mono">
                                        {oracle !== Infinity ? oracle.toFixed(3) + 's' : '-'}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default HistoricalPerformanceTable;
