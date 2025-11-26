import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ResourceMonitor = ({ resources }) => {
    // Transform resources object to array
    const data = Object.keys(resources)
        .filter(k => k.startsWith('gpu_'))
        .map(k => ({
            name: k.toUpperCase().replace('_', ' '),
            load: resources[k].utilization * 100, // Convert to %
            tasks: resources[k].tasks_running,
            memory: resources[k].available_memory
        }));

    return (
        <div style={{ width: '100%', height: '300px' }}>
            {data.length === 0 ? (
                <div className="flex-center" style={{ height: '100%', color: '#64748b' }}>
                    Waiting for resource data...
                </div>
            ) : (
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
                        <XAxis type="number" domain={[0, 100]} hide />
                        <YAxis type="category" dataKey="name" width={60} tick={{ fill: '#94a3b8' }} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        />
                        <Bar dataKey="load" radius={[0, 4, 4, 0]} barSize={30}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.load > 80 ? '#ef4444' : entry.load > 50 ? '#eab308' : '#3b82f6'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            )}
        </div>
    );
};

export default ResourceMonitor;
