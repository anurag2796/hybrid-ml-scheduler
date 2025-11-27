import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PercentileChart = ({ data, schedulerId, schedulerColor }) => {
    // data: [{ window: 0, p50: 0.5, p75: 0.6, p95: 0.8, p99: 0.9 }]

    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                Insufficient data for percentiles
            </div>
        );
    }

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <defs>
                    <linearGradient id={`p50-${schedulerId}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={schedulerColor} stopOpacity={0.8} />
                        <stop offset="95%" stopColor={schedulerColor} stopOpacity={0.2} />
                    </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis
                    dataKey="window"
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Task Window', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 10 }}
                />
                <YAxis
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Execution Time (s)', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: '#18181b',
                        border: '1px solid #3f3f46',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}
                    formatter={(value) => value.toFixed(4) + 's'}
                />
                <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                    iconType="line"
                />

                <Line
                    type="monotone"
                    dataKey="p50"
                    stroke={schedulerColor}
                    strokeWidth={2}
                    dot={false}
                    name="P50 (Median)"
                />
                <Line
                    type="monotone"
                    dataKey="p75"
                    stroke={schedulerColor}
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 5"
                    name="P75"
                    opacity={0.8}
                />
                <Line
                    type="monotone"
                    dataKey="p95"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="3 3"
                    name="P95"
                />
                <Line
                    type="monotone"
                    dataKey="p99"
                    stroke="#ef4444"
                    strokeWidth={2.5}
                    dot={false}
                    name="P99"
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default PercentileChart;
