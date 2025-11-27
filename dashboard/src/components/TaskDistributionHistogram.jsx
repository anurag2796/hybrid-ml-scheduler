import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const TaskDistributionHistogram = ({ tasks, bins = 10 }) => {
    if (!tasks || tasks.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                No task data available
            </div>
        );
    }

    // Extract sizes
    const sizes = tasks.map(t => t.task?.size || t.size || 0);
    const minSize = Math.min(...sizes);
    const maxSize = Math.max(...sizes);
    const binWidth = (maxSize - minSize) / bins || 1;

    // Create histogram bins
    const histogram = Array(bins).fill(0).map((_, i) => ({
        range: `${(minSize + i * binWidth).toFixed(0)}-${(minSize + (i + 1) * binWidth).toFixed(0)}`,
        count: 0,
        binStart: minSize + i * binWidth
    }));

    // Fill bins
    sizes.forEach(size => {
        const binIndex = Math.min(Math.floor((size - minSize) / binWidth), bins - 1);
        histogram[binIndex].count++;
    });

    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogram} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis
                    dataKey="range"
                    stroke="#6b7280"
                    tick={{ fontSize: 9, angle: -45, textAnchor: 'end' }}
                    height={60}
                    label={{ value: 'Task Size Range', position: 'insideBottom', offset: -15, fill: '#9ca3af', fontSize: 10 }}
                />
                <YAxis
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }}
                />
                <Tooltip
                    contentStyle={{
                        backgroundColor: '#18181b',
                        border: '1px solid #3f3f46',
                        borderRadius: '8px'
                    }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {histogram.map((entry, index) => (
                        <Cell
                            key={`cell-${index}`}
                            fill={`hsl(${180 + (index / bins) * 100}, 70%, 50%)`}
                        />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};

export default TaskDistributionHistogram;
