import React from 'react';

const LiveLog = ({ tasks }) => {
    return (
        <div style={{ flex: 1, overflowY: 'auto', minHeight: '0' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead style={{ position: 'sticky', top: 0, backgroundColor: '#1e293b', zIndex: 10 }}>
                    <tr style={{ textAlign: 'left', color: '#94a3b8' }}>
                        <th style={{ padding: '0.75rem' }}>Task ID</th>
                        <th style={{ padding: '0.75rem' }}>Placement</th>
                        <th style={{ padding: '0.75rem' }}>Est. Time</th>
                        <th style={{ padding: '0.75rem' }}>Reward</th>
                        <th style={{ padding: '0.75rem' }}>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {tasks.map((task, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #334155' }}>
                            <td style={{ padding: '0.75rem', fontFamily: 'monospace' }}>#{task.task_id}</td>
                            <td style={{ padding: '0.75rem' }}>
                                {task.gpu_fraction > 0.5 ? (
                                    <span style={{ color: '#38bdf8' }}>GPU {task.gpu_id}</span>
                                ) : (
                                    <span style={{ color: '#a8a29e' }}>CPU</span>
                                )}
                            </td>
                            <td style={{ padding: '0.75rem' }}>{task.estimated_time.toFixed(2)}s</td>
                            <td style={{ padding: '0.75rem', color: task.reward > -1 ? '#22c55e' : '#ef4444' }}>
                                {task.reward ? task.reward.toFixed(2) : '-'}
                            </td>
                            <td style={{ padding: '0.75rem' }}>
                                <span style={{
                                    padding: '0.25rem 0.5rem',
                                    borderRadius: '9999px',
                                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                                    color: '#22c55e',
                                    fontSize: '0.75rem'
                                }}>
                                    Scheduled
                                </span>
                            </td>
                        </tr>
                    ))}
                    {tasks.length === 0 && (
                        <tr>
                            <td colSpan="5" style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                                Waiting for tasks...
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
};

export default LiveLog;
