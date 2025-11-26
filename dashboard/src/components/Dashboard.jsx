import React, { useEffect, useState, useRef } from 'react';
import { Activity, Cpu, Server, Zap } from 'lucide-react';
import ResourceMonitor from './ResourceMonitor';
import LiveLog from './LiveLog';
import MetricsChart from './MetricsChart';

const Dashboard = () => {
    const [connected, setConnected] = useState(false);
    const [resources, setResources] = useState({});
    const [tasks, setTasks] = useState([]);
    const [metrics, setMetrics] = useState([]);
    const ws = useRef(null);

    useEffect(() => {
        const connect = () => {
            ws.current = new WebSocket('ws://localhost:8000/ws');

            ws.current.onopen = () => {
                console.log('Connected to WebSocket');
                setConnected(true);
            };

            ws.current.onclose = () => {
                console.log('Disconnected');
                setConnected(false);
                setTimeout(connect, 3000); // Reconnect
            };

            ws.current.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
        };

        connect();

        return () => {
            if (ws.current) ws.current.close();
        };
    }, []);

    const handleMessage = (msg) => {
        if (msg.type === 'resources') {
            setResources(msg.data);
        } else if (msg.type === 'decision') {
            // Update tasks
            setTasks(prev => [msg.data, ...prev].slice(0, 50));

            // Update resources
            if (msg.utilization) {
                setResources(msg.utilization);
            }

            // Update metrics
            if (msg.rl_metrics) {
                setMetrics(prev => [...prev, {
                    time: new Date().toLocaleTimeString(),
                    reward: msg.rl_metrics.reward,
                    epsilon: msg.rl_metrics.epsilon,
                    avg_time: msg.data.estimated_time
                }].slice(-50)); // Keep last 50 points
            }
        }
    };

    return (
        <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <header style={{
                padding: '1rem 2rem',
                borderBottom: '1px solid #334155',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: '#0f172a'
            }}>
                <div className="flex-center" style={{ gap: '1rem' }}>
                    <Activity className="text-accent" size={28} />
                    <div>
                        <h1 style={{ fontSize: '1.25rem' }}>Hybrid ML Scheduler</h1>
                        <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>Live Dashboard</span>
                    </div>
                </div>
                <div className="flex-center" style={{ gap: '0.5rem' }}>
                    <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: connected ? '#22c55e' : '#ef4444'
                    }} />
                    <span style={{ fontSize: '0.875rem', color: connected ? '#22c55e' : '#ef4444' }}>
                        {connected ? 'System Online' : 'Disconnected'}
                    </span>
                </div>
            </header>

            {/* Main Grid */}
            <main className="grid-layout">
                {/* Resource Monitor */}
                <div className="card col-span-8">
                    <div className="flex-between" style={{ marginBottom: '1rem' }}>
                        <h2 className="flex-center" style={{ gap: '0.5rem' }}>
                            <Server size={20} /> Cluster Resources
                        </h2>
                        <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                            Avg Load: {(resources.average_utilization * 100 || 0).toFixed(1)}%
                        </span>
                    </div>
                    <ResourceMonitor resources={resources} />
                </div>

                {/* Key Metrics */}
                <div className="card col-span-4">
                    <div className="flex-between" style={{ marginBottom: '1rem' }}>
                        <h2 className="flex-center" style={{ gap: '0.5rem' }}>
                            <Zap size={20} /> Performance
                        </h2>
                    </div>
                    <MetricsChart data={metrics} />
                </div>

                {/* Live Log */}
                <div className="card col-span-12" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <div className="flex-between" style={{ marginBottom: '1rem' }}>
                        <h2 className="flex-center" style={{ gap: '0.5rem' }}>
                            <Cpu size={20} /> Recent Scheduling Decisions
                        </h2>
                        <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                            {tasks.length} tasks processed
                        </span>
                    </div>
                    <LiveLog tasks={tasks} />
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
