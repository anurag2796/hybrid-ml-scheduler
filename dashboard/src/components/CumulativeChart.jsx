import React from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CumulativeChart = ({ data, dataKeys, colors }) => {
    // data: [{ time: 0, scheduler_a: 10, scheduler_b: 12 }]
    // dataKeys: ['scheduler_a', 'scheduler_b']
    // colors: { scheduler_a: '#06b6d4', scheduler_b: '#8b5cf6' }

    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                No cumulative data available
            </div>
        );
    }

    return (
        <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                <defs>
                    {dataKeys.map(key => (
                        <linearGradient key={key} id={`gradient-${key}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={colors[key]} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={colors[key]} stopOpacity={0} />
                        </linearGradient>
                    ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis
                    dataKey="index"
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Task Index', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 10 }}
                />
                <YAxis
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Cumulative Time (s)', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: '#18181b',
                        border: '1px solid #3f3f46',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}
                    formatter={(value) => value.toFixed(2) + 's'}
                />
                <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                    iconType="line"
                />

                {dataKeys.map(key => (
                    <React.Fragment key={key}>
                        <Area
                            type="monotone"
                            dataKey={key}
                            fill={`url(#gradient-${key})`}
                            stroke="none"
                        />
                        <Line
                            type="monotone"
                            dataKey={key}
                            stroke={colors[key]}
                            strokeWidth={2}
                            dot={false}
                            name={key.replace(/_/g, ' ').toUpperCase()}
                        />
                    </React.Fragment>
                ))}
            </ComposedChart>
        </ResponsiveContainer>
    );
};

export default CumulativeChart;
