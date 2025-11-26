import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const MetricsChart = ({ data }) => {
    return (
        <div style={{ width: '100%', height: '200px' }}>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis yAxisId="left" stroke="#38bdf8" fontSize={12} />
                    <YAxis yAxisId="right" orientation="right" stroke="#22c55e" fontSize={12} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                    />
                    <Line yAxisId="left" type="monotone" dataKey="avg_time" stroke="#38bdf8" dot={false} strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="reward" stroke="#22c55e" dot={false} strokeWidth={2} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default MetricsChart;
