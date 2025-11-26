import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ScatterChart, Scatter, PieChart, Pie, Legend
} from 'recharts';
import { Play, Pause, Activity, Cpu, Zap, Layers, Terminal, Wifi, BarChart2, Database, PieChart as PieIcon, Target } from 'lucide-react';

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [history, setHistory] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [currentTask, setCurrentTask] = useState(null);
  const [latestResults, setLatestResults] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [notification, setNotification] = useState(null);

  const wsRef = useRef(null);

  // WebSocket Connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'notification') {
          setNotification(message.message);
          setTimeout(() => setNotification(null), 5000);
          return;
        }

        if (message.type === 'simulation_update') {
          const newData = transformData(message);
          setData(prev => [...prev, newData].slice(-50));
          setHistory(prev => [message, ...prev].slice(0, 50));
          setComparisonData(message.comparison);
          setCurrentTask(message.task);
          setLatestResults(message.latest_results);
        }
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const togglePause = async () => {
    const newState = !isPaused;
    setIsPaused(newState);
    try {
      await fetch(`http://localhost:8000/api/${newState ? 'pause' : 'resume'}`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to toggle simulation", e);
    }
  };

  const transformData = (message) => {
    const util = message.utilization;
    return {
      time: new Date().toLocaleTimeString(),
      avgUtil: (util.average_utilization * 100).toFixed(1),
      gpu0: util.gpu_0.utilization * 100,
      gpu1: util.gpu_1.utilization * 100,
      gpu2: util.gpu_2.utilization * 100,
      gpu3: util.gpu_3.utilization * 100,
      raw: message
    };
  };

  // Prepare Radar Data (Time, Energy, Cost normalized)
  const radarData = latestResults ? [
    { subject: 'Time', A: latestResults.hybrid_ml.time, B: latestResults.oracle.time, fullMark: 1 },
    { subject: 'Energy', A: latestResults.hybrid_ml.energy / 100, B: latestResults.oracle.energy / 100, fullMark: 1 },
    { subject: 'Cost', A: latestResults.hybrid_ml.cost * 1000, B: latestResults.oracle.cost * 1000, fullMark: 1 },
  ] : [];

  // Prepare Pie Data (Compute vs Memory)
  const pieData = history.slice(0, 50).reduce((acc, item) => {
    if (item.task.intensity > 0.5) acc[0].value++;
    else acc[1].value++;
    return acc;
  }, [{ name: 'Compute Bound', value: 0, fill: '#8b5cf6' }, { name: 'Memory Bound', value: 0, fill: '#06b6d4' }]);

  // Prepare Scatter Data (Size vs Duration)
  const scatterData = history.map(h => ({
    x: h.task.size,
    y: h.latest_results.hybrid_ml.time,
    z: h.task.intensity
  }));

  return (
    <div className="min-h-screen grid-bg p-4 flex flex-col gap-4">
      {/* Notification Toast */}
      {notification && (
        <div className="fixed top-20 right-6 z-50 glass-panel p-4 border-l-4 border-l-green-500 animate-bounce">
          <p className="text-green-400 font-bold">SYSTEM UPDATE</p>
          <p className="text-sm">{notification}</p>
        </div>
      )}

      {/* Header */}
      <header className="glass-panel p-3 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
            <Cpu className="text-cyan-400" size={20} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white glow-text">HYBRID ML SCHEDULER</h1>
            <p className="text-[10px] text-cyan-400/70 font-mono tracking-wider">MISSION CONTROL // V2.0</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-black/40 border border-white/5">
            <Wifi size={12} className={isConnected ? "text-green-400" : "text-red-400"} />
            <span className="text-[10px] font-medium text-gray-300">{isConnected ? "ONLINE" : "OFFLINE"}</span>
          </div>
          <button
            onClick={togglePause}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors border border-transparent hover:border-white/10"
          >
            {isPaused ? <Play size={18} className="text-green-400" /> : <Pause size={18} className="text-yellow-400" />}
          </button>
        </div>
      </header>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-12 grid-rows-12 gap-4 flex-1 min-h-[800px]">

        {/* 1. Scheduler Race (Top Left, 4x4) */}
        <div className="col-span-12 md:col-span-4 row-span-4 glass-panel p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-2">
            <BarChart2 size={14} /> PERFORMANCE RACE
          </h2>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
              <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 9 }} />
              <YAxis dataKey="name" type="category" stroke="#9ca3af" width={70} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46' }} />
              <Bar dataKey="avg_time" radius={[0, 4, 4, 0]} barSize={15}>
                {comparisonData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={
                    entry.name === 'oracle' ? '#10b981' :
                      entry.name === 'hybrid_ml' ? '#06b6d4' :
                        entry.name === 'rl_agent' ? '#8b5cf6' : '#4b5563'
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 2. Radar Chart (Top Center, 4x4) */}
        <div className="col-span-12 md:col-span-4 row-span-4 glass-panel p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-2">
            <Target size={14} /> HYBRID VS ORACLE
          </h2>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} axisLine={false} />
              <Radar name="Hybrid ML" dataKey="A" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.3} />
              <Radar name="Oracle" dataKey="B" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* 3. Live Feed (Right Column, 4x12) */}
        <div className="col-span-12 md:col-span-4 row-span-12 glass-panel flex flex-col overflow-hidden">
          <div className="p-3 border-b border-white/5 bg-black/20">
            <h2 className="text-xs font-semibold text-gray-400 flex items-center gap-2">
              <Terminal size={14} /> LIVE DECISION LOG
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2 font-mono text-[10px]">
            {history.map((item, idx) => (
              <div key={idx} className="p-2 rounded bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                <div className="flex justify-between text-gray-500 mb-1">
                  <span>{new Date().toLocaleTimeString()}</span>
                  <span className="text-cyan-400">TASK-{item.task?.id}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-500 block">Hybrid</span>
                    <span className="text-green-400">{item.latest_results?.hybrid_ml?.time.toFixed(3)}s</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">Energy</span>
                    <span className="text-yellow-400">{item.latest_results?.hybrid_ml?.energy.toFixed(1)}J</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 4. Real-time Utilization (Middle Left, 8x4) */}
        <div className="col-span-12 md:col-span-8 row-span-4 glass-panel p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-2">
            <Activity size={14} /> CLUSTER UTILIZATION
          </h2>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorUtil" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
              <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 9 }} interval={10} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 9 }} domain={[0, 100]} />
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46' }} />
              <Area type="monotone" dataKey="avgUtil" stroke="#8b5cf6" strokeWidth={2} fill="url(#colorUtil)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* 5. Scatter Plot (Bottom Left, 4x4) */}
        <div className="col-span-12 md:col-span-4 row-span-4 glass-panel p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-2">
            <Database size={14} /> WORKLOAD DISTRIBUTION
          </h2>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" dataKey="x" name="Size" stroke="#6b7280" tick={{ fontSize: 9 }} />
              <YAxis type="number" dataKey="y" name="Time" stroke="#6b7280" tick={{ fontSize: 9 }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46' }} />
              <Scatter name="Tasks" data={scatterData} fill="#f472b6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* 6. Pie Chart (Bottom Center, 4x4) */}
        <div className="col-span-12 md:col-span-4 row-span-4 glass-panel p-4 flex flex-col">
          <h2 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-2">
            <PieIcon size={14} /> TASK INTENSITY
          </h2>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={5}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46' }} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
