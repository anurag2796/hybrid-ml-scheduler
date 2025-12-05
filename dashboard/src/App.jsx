
import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ScatterChart, Scatter, PieChart, Pie, Legend, LineChart, Line
} from 'recharts';
import { Play, Pause, Activity, Cpu, Zap, Layers, Terminal, Wifi, BarChart2, Database, PieChart as PieIcon, Target, History, Trash2, LayoutDashboard, Server, Brain, Shuffle, MousePointer2, GripHorizontal, TrendingUp } from 'lucide-react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import EnhancedVisualizationsDemo from './components/EnhancedVisualizationsDemo';
import HistoricalPerformanceTable from './components/HistoricalPerformanceTable';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [history, setHistory] = useState([]);
  const [fullHistory, setFullHistory] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [currentTask, setCurrentTask] = useState(null);
  const [latestResults, setLatestResults] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [activeView, setActiveView] = useState('global'); // 'global', 'history', 'enhanced', or scheduler_id
  const [notification, setNotification] = useState(null);

  const wsRef = useRef(null);

  const SCHEDULERS = [
    { id: 'hybrid_ml', name: 'Hybrid ML', icon: Brain, color: '#06b6d4' },
    { id: 'rl_agent', name: 'RL Agent', icon: Zap, color: '#8b5cf6' },
    { id: 'oracle', name: 'Oracle', icon: Target, color: '#10b981' },
    { id: 'round_robin', name: 'Round Robin', icon: Layers, color: '#f59e0b' },
    { id: 'random', name: 'Random', icon: Shuffle, color: '#ef4444' },
    { id: 'greedy', name: 'Greedy', icon: MousePointer2, color: '#ec4899' },
  ];

  // Default Layouts for Scheduler View
  const defaultLayouts = {
    lg: [
      { i: 'metrics', x: 0, y: 0, w: 12, h: 2, static: true },
      { i: 'cluster_load', x: 0, y: 2, w: 8, h: 8 },
      { i: 'oracle_vs', x: 8, y: 2, w: 4, h: 6 },
      { i: 'resource_split', x: 8, y: 8, w: 4, h: 6 },
      { i: 'logs', x: 0, y: 10, w: 8, h: 6 }
    ]
  };

  const [layouts, setLayouts] = useState(defaultLayouts);

  // WebSocket Connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
      };
      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setIsConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log('📨 WebSocket message received:', message.type);

        if (message.type === 'notification') {
          setNotification(message.message);
          setTimeout(() => setNotification(null), 5000);
          return;
        }

        if (message.type === 'simulation_update') {
          console.log('✅ Simulation update - latestResults:', message.latest_results ? 'YES' : 'NO');
          const newData = transformData(message);
          if (newData) {
            setData(prev => {
              const updated = [...prev, newData].slice(-50);
              console.log('📊 Data array length:', updated.length);
              return updated;
            });
          }
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

  // Fetch Full History when view is history
  useEffect(() => {
    if (activeView === 'history' || activeView === 'enhanced') {
      fetch('http://localhost:8000/api/full_history')
        .then(res => res.json())
        .then(data => {
          console.log('Fetched full history, length:', data.length);
          setFullHistory(data);
        })
        .catch(err => console.error("Failed to fetch full history", err));
    }
  }, [activeView]);

  const togglePause = async () => {
    const newState = !isPaused;
    setIsPaused(newState);
    try {
      await fetch(`http://localhost:8000/api/${newState ? 'pause' : 'resume'}`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to toggle simulation", e);
    }
  };

  const clearHistory = async () => {
    if (confirm("Are you sure you want to clear all historical data?")) {
      await fetch('http://localhost:8000/api/history', { method: 'DELETE' });
      setFullHistory([]);
      setNotification("History Cleared");
    }
  };

  const transformData = (message) => {
    try {
      const util = message.utilization;
      if (!util) {
        console.error('❌ No utilization data in message');
        return null;
      }
      const transformed = {
        time: new Date().toLocaleTimeString(),
        avgUtil: (util.average_utilization * 100).toFixed(1),
        gpu0: (util.gpu_0?.utilization || 0) * 100,
        gpu1: (util.gpu_1?.utilization || 0) * 100,
        gpu2: (util.gpu_2?.utilization || 0) * 100,
        gpu3: (util.gpu_3?.utilization || 0) * 100,
        raw: message
      };
      console.log('✅ Transformed data:', transformed);
      return transformed;
    } catch (error) {
      console.error('❌ Error in transformData:', error);
      return null;
    }
  };

  // --- Derived Metrics for Selected Scheduler ---
  const getSelectedMetrics = () => {
    if (!latestResults || !latestResults[activeView]) return null;
    const res = latestResults[activeView];
    return {
      time: res.time,
      energy: res.energy,
      cost: res.cost,
      efficiency: (1.0 / (res.cost + 0.00001)).toFixed(0) // Tasks per $
    };
  };

  const selectedMetrics = getSelectedMetrics();

  // Filter history for selected scheduler log
  const selectedHistory = history.map(h => ({
    time: new Date().toLocaleTimeString(),
    task_id: h.task.id,
    result: h.latest_results[activeView]
  }));

  // Radar Data: Selected vs Oracle
  const radarData = latestResults ? [
    { subject: 'Time', A: latestResults[activeView]?.time || 0, B: latestResults.oracle?.time || 0, fullMark: 10 },
    // Energy is usually 100-300J. Divide by 200 to map to ~0.5-1.5 range
    { subject: 'Energy', A: (latestResults[activeView]?.energy || 0) / 200, B: (latestResults.oracle?.energy || 0) / 200, fullMark: 10 },
    // Cost is tiny (~5e-6). Multiply by 200,000 to map to ~1.0 range
    { subject: 'Cost', A: (latestResults[activeView]?.cost || 0) * 200000, B: (latestResults.oracle?.cost || 0) * 200000, fullMark: 10 },
  ] : [];

  // Pie Data: GPU vs CPU Split (Mocked based on scheduler logic)
  const getSplitData = () => {
    if (activeView === 'rl_agent') return [{ name: 'GPU', value: 90, fill: '#8b5cf6' }, { name: 'CPU', value: 10, fill: '#4b5563' }];
    if (activeView === 'round_robin') return [{ name: 'GPU', value: 50, fill: '#f59e0b' }, { name: 'CPU', value: 50, fill: '#4b5563' }];
    if (activeView === 'random') return [{ name: 'GPU', value: 40, fill: '#ef4444' }, { name: 'CPU', value: 60, fill: '#4b5563' }];
    const gpuVal = currentTask?.intensity > 0.5 ? 80 : 20;
    return [{ name: 'GPU', value: gpuVal, fill: '#06b6d4' }, { name: 'CPU', value: 100 - gpuVal, fill: '#4b5563' }];
  };

  // Historical Trend Data
  const trendData = fullHistory.map((item, idx) => ({
    id: idx,
    optimal_time: item.optimal_time,
    size: item.size
  }));

  // Render Content based on Active View
  const renderContent = () => {
    if (activeView === 'global') {
      return (
        <div className="grid grid-cols-12 gap-6 p-6">
          <div className="col-span-12 md:col-span-6 glass-panel p-6 relative overflow-hidden group" style={{ height: 450 }}>
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2 relative z-10">
              <BarChart2 size={16} className="text-cyan-400" /> PERFORMANCE RACE
            </h2>
            <div className="w-full relative z-10" style={{ height: 380 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} layout="vertical" margin={{ left: 40, right: 30, top: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                  <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" stroke="#9ca3af" width={80} tick={{ fontSize: 11, fontWeight: 500 }} />
                  <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px' }} />
                  <Bar dataKey="avg_time" radius={[0, 4, 4, 0]} barSize={24}>
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
          </div>
          <div className="col-span-12 md:col-span-6 glass-panel p-6 relative overflow-hidden group" style={{ height: 450 }}>
            <div className="absolute inset-0 bg-gradient-to-bl from-pink-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2 relative z-10">
              <Database size={16} className="text-pink-400" /> WORKLOAD DISTRIBUTION
            </h2>
            <div className="w-full relative z-10" style={{ height: 380 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" dataKey="x" name="Size" stroke="#6b7280" tick={{ fontSize: 10 }} label={{ value: 'Task Size', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }} />
                  <YAxis type="number" dataKey="y" name="Time" stroke="#6b7280" tick={{ fontSize: 10 }} label={{ value: 'Execution Time (s)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px' }} />
                  <Scatter name="Tasks" data={history.map(h => ({ x: h.task?.size || 0, y: h.latest_results?.hybrid_ml?.time || 0 }))} fill="#f472b6" shape="circle" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Historical Data Section */}
          <div className="col-span-12 glass-panel p-6 relative overflow-hidden group">
            <HistoricalPerformanceTable />
          </div>
        </div>
      );
    }

    if (activeView === 'history') {
      return (
        <div className="p-6 h-full flex flex-col">
          <div className="glass-panel p-8 text-center flex-1 flex flex-col relative overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900/20 via-black to-black -z-10" />
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <History className="text-purple-400" /> Historical Analysis Mode
              </h2>
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-400 font-mono">
                  DATA POINTS: <span className="text-white font-bold">{fullHistory.length}</span>
                </span>
                <button onClick={clearHistory} className="text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors flex items-center gap-2 text-xs px-4 py-2 border border-red-500/30 rounded-md">
                  <Trash2 size={14} /> Clear Data
                </button>
              </div>
            </div>

            <div className="flex-1 min-h-[400px] w-full bg-black/20 rounded-xl border border-white/5 p-4">
              {fullHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                    <XAxis dataKey="id" stroke="#6b7280" tick={{ fontSize: 10 }} label={{ value: 'Task ID', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }} />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} label={{ value: 'Optimal Time (s)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="optimal_time" name="Optimal Execution Time" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                  <Database size={48} className="mb-4 opacity-20" />
                  <p className="text-lg font-semibold">No historical data available yet</p>
                  <p className="text-sm mt-2 text-gray-600">The simulation is collecting data points...</p>
                  <p className="text-xs mt-4 text-gray-700">Current data points: {fullHistory.length}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    if (activeView === 'enhanced') {
      return (
        <EnhancedVisualizationsDemo
          history={history}
          fullHistory={fullHistory}
          comparisonData={comparisonData}
          latestResults={latestResults}
        />
      );
    }

    // Scheduler View (Grid Layout)
    return (
      <ResponsiveGridLayout
        className="layout p-6"
        layouts={layouts}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
        rowHeight={30}
        onLayoutChange={(layout, layouts) => setLayouts(layouts)}
        draggableHandle=".drag-handle"
        margin={[24, 24]}
      >
        <div key="metrics" className="glass-panel p-0 flex justify-between items-center overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-transparent to-transparent pointer-events-none" />
          <div className="flex gap-0 w-full h-full relative z-10 divide-x divide-white/5">
            <div className="flex-1 p-4 flex flex-col justify-center">
              <div className="text-[10px] font-bold text-gray-500 tracking-wider mb-1">AVG LATENCY</div>
              <div className="text-2xl font-bold text-neon-cyan font-mono">{selectedMetrics?.time.toFixed(4) || '-'}s</div>
            </div>
            <div className="flex-1 p-4 flex flex-col justify-center">
              <div className="text-[10px] font-bold text-gray-500 tracking-wider mb-1">ENERGY / TASK</div>
              <div className="text-2xl font-bold text-neon-purple font-mono">{selectedMetrics?.energy.toFixed(2) || '-'}J</div>
            </div>
            <div className="flex-1 p-4 flex flex-col justify-center">
              <div className="text-[10px] font-bold text-gray-500 tracking-wider mb-1">COST EFFICIENCY</div>
              <div className="text-2xl font-bold text-white font-mono">{selectedMetrics?.efficiency || '-'} T/$</div>
            </div>
          </div>
          <div className="h-full w-8 bg-white/5 flex items-center justify-center cursor-move drag-handle hover:bg-white/10 transition-colors">
            <GripHorizontal className="text-gray-500" size={16} />
          </div>
        </div>

        <div key="cluster_load" className="glass-panel p-0 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-white/5 bg-black/20 flex justify-between items-center cursor-move drag-handle">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2">
              <Activity size={14} className="text-cyan-400" /> VIRTUAL CLUSTER LOAD
            </h3>
            <GripHorizontal className="text-gray-600" size={14} />
          </div>
          <div className="flex-1 min-h-0 w-full p-4">
            {data.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                  <defs>
                    <linearGradient id="colorUtil" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={SCHEDULERS.find(s => s.id === activeView)?.color} stopOpacity={0.4} />
                      <stop offset="95%" stopColor={SCHEDULERS.find(s => s.id === activeView)?.color} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 9 }} interval={10} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 9 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px' }} />
                  <Area type="monotone" dataKey="avgUtil" stroke={SCHEDULERS.find(s => s.id === activeView)?.color} strokeWidth={2} fill="url(#colorUtil)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-xs">Waiting for simulation data...</div>}
          </div>
        </div>


        <div key="oracle_vs" className="glass-panel p-0 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-white/5 bg-black/20 flex justify-between items-center cursor-move drag-handle">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2">
              <Target size={14} className="text-green-400" /> VS ORACLE BASELINE
            </h3>
            <GripHorizontal className="text-gray-600" size={14} />
          </div>
          <div className="flex-1 min-h-0 w-full p-2">
            {latestResults ? (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="65%" data={radarData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 10, fontWeight: 500 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} axisLine={false} />
                  <Radar name={SCHEDULERS.find(s => s.id === activeView)?.name} dataKey="A" stroke={SCHEDULERS.find(s => s.id === activeView)?.color} fill={SCHEDULERS.find(s => s.id === activeView)?.color} fillOpacity={0.4} />
                  <Radar name="Oracle" dataKey="B" stroke="#10b981" fill="#10b981" fillOpacity={0.1} />
                  <Legend wrapperStyle={{ fontSize: '10px', bottom: 0 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px' }} />
                </RadarChart>
              </ResponsiveContainer>
            ) : <div className="flex items-center justify-center h-full text-gray-500 text-xs">Waiting for simulation data...</div>}
          </div>

        </div>
        <div key="resource_split" className="glass-panel p-0 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-white/5 bg-black/20 flex justify-between items-center cursor-move drag-handle">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2">
              <Server size={14} className="text-yellow-400" /> RESOURCE SPLIT
            </h3>
            <GripHorizontal className="text-gray-600" size={14} />
          </div>
          <div className="flex-1 min-h-0 w-full p-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={getSplitData()} cx="50%" cy="50%" innerRadius={45} outerRadius={65} paddingAngle={5} dataKey="value">
                  {getSplitData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} stroke="rgba(0,0,0,0.5)" />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px' }} />
                <Legend wrapperStyle={{ fontSize: '10px', bottom: 0 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div key="logs" className="glass-panel p-0 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-white/5 bg-black/20 flex justify-between items-center cursor-move drag-handle">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2">
              <Terminal size={14} className="text-gray-400" /> {SCHEDULERS.find(s => s.id === activeView)?.name.toUpperCase()} LOGS
            </h3>
            <GripHorizontal className="text-gray-600" size={14} />
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-1 font-mono text-[10px] bg-black/10">
            {selectedHistory.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-2 rounded bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                <span className="text-gray-500 w-16">{item.time}</span>
                <span className="text-cyan-400 w-16">TASK-{item.task_id}</span>
                <div className="flex-1 flex justify-end gap-4">
                  <span className="text-gray-300">T: <span className="text-white">{item.result?.time.toFixed(3)}s</span></span>
                  <span className="text-yellow-500/80">E: <span className="text-yellow-400">{item.result?.energy.toFixed(1)}J</span></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </ResponsiveGridLayout >
    );
  };

  return (
    <div className="min-h-screen grid-bg flex">
      {/* Sidebar */}
      <div className="w-64 glass-panel border-r border-white/5 flex flex-col z-20">
        <div className="p-4 border-b border-white/5">
          <h1 className="text-lg font-bold tracking-tight text-white glow-text flex items-center gap-2">
            <LayoutDashboard size={20} className="text-neon-cyan" />
            <span className="text-neon-cyan">SCHEDULER LAB</span>
          </h1>
          <p className="text-[10px] text-cyan-400/70 font-mono tracking-wider mt-1">VIRTUAL CLUSTER MANAGER</p>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="text-xs font-semibold text-gray-500 px-2 py-2">VIEWS</div>
          <button
            onClick={() => setActiveView('global')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${activeView === 'global' ? 'bg-white/10 border border-white/10 text-white shadow-lg' : 'text-gray-400 hover:bg-white/5'}`}
          >
            <BarChart2 size={18} className="text-blue-400" />
            <span className="text-sm font-medium">Global Comparison</span>
          </button>
          <button
            onClick={() => setActiveView('history')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${activeView === 'history' ? 'bg-white/10 border border-white/10 text-white shadow-lg' : 'text-gray-400 hover:bg-white/5'}`}
          >
            <History size={18} className="text-purple-400" />
            <span className="text-sm font-medium">Historical Analysis</span>
          </button>
          <button
            onClick={() => setActiveView('enhanced')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${activeView === 'enhanced' ? 'bg-white/10 border border-white/10 text-white shadow-lg' : 'text-gray-400 hover:bg-white/5'}`}
          >
            <TrendingUp size={18} className="text-cyan-400" />
            <span className="text-sm font-medium">Enhanced Analytics</span>
          </button>

          <div className="text-xs font-semibold text-gray-500 px-2 py-2 mt-4">SCHEDULERS</div>
          <div className="space-y-1">
            {SCHEDULERS.map(s => (
              <button
                key={s.id}
                onClick={() => setActiveView(s.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${activeView === s.id ? 'bg-white/10 border border-white/10 text-white shadow-lg' : 'text-gray-400 hover:bg-white/5'}`}
              >
                <s.icon size={18} style={{ color: s.color }} />
                <span className="text-sm font-medium">{s.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="p-4 border-t border-white/5">
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-black/40 border border-white/5 mb-2">
            <Wifi size={14} className={isConnected ? "text-green-400" : "text-red-400"} />
            <span className="text-xs font-medium text-gray-300">{isConnected ? "ONLINE" : "OFFLINE"}</span>
          </div>
          <button
            onClick={togglePause}
            className={`w-full flex items-center justify-center gap-2 p-2 rounded-lg transition-colors border ${isPaused ? 'bg-green-500/20 border-green-500/50 text-green-300' : 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300'}`}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
            <span className="text-xs font-bold">{isPaused ? "RESUME SIM" : "PAUSE SIM"}</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden relative">
        {/* Notification Toast */}
        {notification && (
          <div className="absolute top-6 right-6 z-50 glass-panel p-4 border-l-4 border-l-green-500 animate-bounce">
            <p className="text-green-400 font-bold">SYSTEM UPDATE</p>
            <p className="text-sm">{notification}</p>
          </div>
        )}

        {/* Top Bar */}
        <header className="h-16 border-b border-white/5 flex justify-between items-center px-6 bg-black/20 backdrop-blur-sm">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              {activeView === 'global' ? 'Global Comparison' :
                activeView === 'history' ? 'Historical Analysis' :
                  activeView === 'enhanced' ? 'Enhanced Analytics' :
                    SCHEDULERS.find(s => s.id === activeView)?.name}
            </h2>
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="flex-1 overflow-y-auto">
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
