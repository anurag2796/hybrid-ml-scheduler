import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Minus, ArrowUpDown } from 'lucide-react';

const StatsTable = ({ data, schedulers }) => {
    // data: { scheduler_id: { mean, median, std, min, max, count } }
    const [sortBy, setSortBy] = useState('mean');
    const [sortAsc, setSortAsc] = useState(true);

    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
                No statistics available yet
            </div>
        );
    }

    const metrics = ['mean', 'median', 'std', 'min', 'max', 'count'];
    const metricLabels = {
        mean: 'Mean',
        median: 'Median',
        std: 'Std Dev',
        min: 'Min',
        max: 'Max',
        count: 'Tasks'
    };

    // Sort schedulers
    const sortedSchedulers = Object.keys(data).sort((a, b) => {
        const valA = data[a][sortBy] || 0;
        const valB = data[b][sortBy] || 0;
        return sortAsc ? valA - valB : valB - valA;
    });

    const toggleSort = (metric) => {
        if (sortBy === metric) {
            setSortAsc(!sortAsc);
        } else {
            setSortBy(metric);
            setSortAsc(true);
        }
    };

    // Find best/worst for highlighting
    const getBest = (metric) => {
        if (metric === 'count') return null;
        return Math.min(...Object.values(data).map(d => d[metric] || Infinity));
    };

    const getWorst = (metric) => {
        if (metric === 'count') return null;
        return Math.max(...Object.values(data).map(d => d[metric] || -Infinity));
    };

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-white/10">
                        <th className="text-left p-2 text-gray-400 font-semibold text-xs">Scheduler</th>
                        {metrics.map(metric => (
                            <th
                                key={metric}
                                className="text-right p-2 text-gray-400 font-semibold text-xs cursor-pointer hover:text-white transition-colors"
                                onClick={() => toggleSort(metric)}
                            >
                                <div className="flex items-center justify-end gap-1">
                                    {metricLabels[metric]}
                                    <ArrowUpDown size={12} className={sortBy === metric ? 'text-cyan-400' : 'text-gray-600'} />
                                </div>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sortedSchedulers.map((schedulerId, idx) => {
                        const stats = data[schedulerId];
                        const scheduler = schedulers?.find(s => s.id === schedulerId);

                        return (
                            <tr
                                key={schedulerId}
                                className={`border-b border-white/5 hover:bg-white/5 transition-colors ${idx === 0 ? 'bg-green-500/10' : idx === sortedSchedulers.length - 1 ? 'bg-red-500/10' : ''
                                    }`}
                            >
                                <td className="p-2 font-semibold flex items-center gap-2">
                                    <div
                                        className="w-2 h-2 rounded-full"
                                        style={{ backgroundColor: scheduler?.color || '#6b7280' }}
                                    />
                                    <span style={{ color: scheduler?.color || '#fff' }}>{scheduler?.name || schedulerId}</span>
                                </td>
                                {metrics.map(metric => {
                                    const value = stats[metric];
                                    const best = getBest(metric);
                                    const worst = getWorst(metric);
                                    const isBest = value === best && best !== null;
                                    const isWorst = value === worst && worst !== null;

                                    return (
                                        <td
                                            key={metric}
                                            className={`p-2 text-right font-mono ${isBest ? 'text-green-400 font-bold' :
                                                    isWorst ? 'text-red-400' :
                                                        'text-gray-300'
                                                }`}
                                        >
                                            {metric === 'count' ? value : value?.toFixed(4)}
                                            {isBest && <TrendingDown size={12} className="inline ml-1 text-green-400" />}
                                            {isWorst && <TrendingUp size={12} className="inline ml-1 text-red-400" />}
                                        </td>
                                    );
                                })}
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};

export default StatsTable;
