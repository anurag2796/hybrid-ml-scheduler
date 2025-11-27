import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Gauge, TrendingUp, Target, Database, Layers } from 'lucide-react';
import GaugeChart from './GaugeChart.jsx';
import HeatmapChart from './HeatmapChart.jsx';
import StatsTable from './StatsTable.jsx';
import WinLossMatrix from './WinLossMatrix.jsx';
import CorrelationMatrix from './CorrelationMatrix.jsx';
import PercentileChart from './PercentileChart.jsx';
import TaskDistributionHistogram from './TaskDistributionHistogram.jsx';
import CumulativeChart from './CumulativeChart.jsx';

const EnhancedVisualizationsDemo = ({ history, fullHistory, comparisonData, latestResults }) => {
    const SCHEDULERS = [
        { id: 'hybrid_ml', name: 'Hybrid ML', icon: null, color: '#06b6d4' },
        { id: 'rl_agent', name: 'RL Agent', icon: null, color: '#a855f7' },
        { id: 'oracle', name: 'Oracle', icon: null, color: '#10b981' },
        { id: 'round_robin', name: 'Round Robin', icon: null, color: '#fbbf24' },
        { id: 'random', name: 'Random', icon: null, color: '#f87171' },
        { id: 'greedy', name: 'Greedy', icon: null, color: '#f472b6' },
    ];

    // Calculate all the enhanced metrics
    const schedulerStats = useMemo(() => {
        if (!history || history.length === 0) return {};

        const stats = {};
        SCHEDULERS.forEach(scheduler => {
            const times = history
                .map(h => h.latest_results?.[scheduler.id]?.time)
                .filter(t => t !== undefined && t !== null);

            if (times.length === 0) return;

            const sorted = [...times].sort((a, b) => a - b);
            const mean = times.reduce((a, b) => a + b, 0) / times.length;
            const median = sorted[Math.floor(sorted.length / 2)];
            const variance = times.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / times.length;
            const std = Math.sqrt(variance);

            stats[scheduler.id] = {
                mean,
                median,
                std,
                min: Math.min(...times),
                max: Math.max(...times),
                count: times.length
            };
        });

        return stats;
    }, [history]);

    const winLossMatrix = useMemo(() => {
        if (!history || history.length < 2) return {};

        const matrix = {};
        SCHEDULERS.forEach(s1 => {
            matrix[s1.id] = {};
            SCHEDULERS.forEach(s2 => {
                if (s1.id === s2.id) return;

                let wins = 0;
                let total = 0;

                history.forEach(h => {
                    const time1 = h.latest_results?.[s1.id]?.time;
                    const time2 = h.latest_results?.[s2.id]?.time;

                    if (time1 !== undefined && time2 !== undefined) {
                        total++;
                        if (time1 < time2) wins++;
                    }
                });

                matrix[s1.id][s2.id] = total > 0 ? wins / total : null;
            });
        });

        return matrix;
    }, [history]);

    const correlationData = useMemo(() => {
        console.log('Computing correlationData, fullHistory length:', fullHistory?.length);
        if (!fullHistory || fullHistory.length < 5) return { correlations: {}, features: [] };

        const features = ['size', 'compute_intensity', 'memory_required', 'optimal_time'];
        const correlations = {};

        features.forEach(f1 => {
            correlations[f1] = {};
            features.forEach(f2 => {
                const values1 = fullHistory.map(h => h[f1] || 0);
                const values2 = fullHistory.map(h => h[f2] || 0);

                const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
                const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;

                let num = 0, den1 = 0, den2 = 0;
                for (let i = 0; i < values1.length; i++) {
                    const diff1 = values1[i] - mean1;
                    const diff2 = values2[i] - mean2;
                    num += diff1 * diff2;
                    den1 += diff1 * diff1;
                    den2 += diff2 * diff2;
                }

                const correlation = den1 === 0 || den2 === 0 ? 0 : num / Math.sqrt(den1 * den2);
                correlations[f1][f2] = correlation;
            });
        });

        return { correlations, features };
    }, [fullHistory]);

    const heatmapData = useMemo(() => {
        console.log('Computing heatmapData, fullHistory length:', fullHistory?.length);
        if (!fullHistory || fullHistory.length < 5) return [];

        const bins = 5;
        const sizes = fullHistory.map(h => h.size || 0);
        const intensities = fullHistory.map(h => h.compute_intensity || 0);

        const minSize = Math.min(...sizes);
        const maxSize = Math.max(...sizes);
        const minInt = Math.min(...intensities);
        const maxInt = Math.max(...intensities);

        const sizeBinWidth = (maxSize - minSize) / bins || 1;
        const intBinWidth = (maxInt - minInt) / bins || 1;

        const grid = {};

        fullHistory.forEach(item => {
            const size = item.size || 0;
            const intensity = item.compute_intensity || 0;
            const time = item.optimal_time || 0;

            const sizeBin = Math.min(Math.floor((size - minSize) / sizeBinWidth), bins - 1);
            const intBin = Math.min(Math.floor((intensity - minInt) / intBinWidth), bins - 1);

            const key = `${sizeBin},${intBin}`;
            if (!grid[key]) grid[key] = { count: 0, totalTime: 0, sizeBin, intBin };
            grid[key].count++;
            grid[key].totalTime += time;
        });

        return Object.values(grid).map(cell => ({
            x: minSize + (cell.sizeBin + 0.5) * sizeBinWidth,
            y: minInt + (cell.intBin + 0.5) * intBinWidth,
            value: cell.totalTime / cell.count
        }));
    }, [fullHistory]);

    const cumulativeData = useMemo(() => {
        if (!history || history.length === 0) return [];

        const cumulative = { index: [] };
        SCHEDULERS.forEach(s => { cumulative[s.id] = []; });

        let running = {};
        SCHEDULERS.forEach(s => running[s.id] = 0);

        history.forEach((h, idx) => {
            cumulative.index.push(idx);
            SCHEDULERS.forEach(s => {
                const time = h.latest_results?.[s.id]?.time || 0;
                running[s.id] += time;
                cumulative[s.id].push(running[s.id]);
            });
        });

        return cumulative.index.map((idx, i) => {
            const entry = { index: idx };
            SCHEDULERS.forEach(s => { entry[s.id] = cumulative[s.id][i]; });
            return entry;
        });
    }, [history]);

    const efficiencyGauges = useMemo(() => {
        if (!latestResults || !latestResults.oracle) return {};

        const gauges = {};
        const oracleTime = latestResults.oracle.time;

        SCHEDULERS.forEach(s => {
            if (s.id === 'oracle') {
                gauges[s.id] = 100;
            } else {
                const schedulerTime = latestResults[s.id]?.time || Infinity;
                const efficiency = Math.min(100, (oracleTime / schedulerTime) * 100);
                gauges[s.id] = efficiency;
            }
        });

        return gauges;
    }, [latestResults]);

    return (
        <div className="p-6 grid grid-cols-12 gap-4 overflow-y-auto" style={{ maxHeight: '90vh' }}>

            {/* Efficiency Gauges */}
            <div className="col-span-12 glass-panel p-4">
                <h2 className="text-sm font-bold text-gray-300 mb-3 flex items-center gap-2">
                    <Gauge size={16} className="text-cyan-400" /> SCHEDULER EFFICIENCY (vs Oracle)
                </h2>
                <div className="grid grid-cols-5 gap-4">
                    {SCHEDULERS.filter(s => s.id !== 'oracle').map(s => (
                        <div key={s.id} className="flex flex-col items-center p-2 rounded-lg bg-black/20 border border-white/5">
                            <GaugeChart value={efficiencyGauges[s.id] || 0} max={100} label={s.name} color={s.color} size={100} />
                        </div>
                    ))}
                </div>
            </div>

            {/* Statistical Table */}
            <div className="col-span-12 glass-panel p-6">
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <TrendingUp size={16} className="text-green-400" /> STATISTICAL COMPARISON
                </h2>
                <StatsTable data={schedulerStats} schedulers={SCHEDULERS} />
            </div>

            {/* Win/Loss Matrix */}
            <div className="col-span-12 md:col-span-6 glass-panel p-6" style={{ height: 450 }}>
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <Target size={16} className="text-yellow-400" /> WIN/LOSS MATRIX
                </h2>
                <div className="flex justify-center" style={{ height: 380 }}>
                    <WinLossMatrix matrix={winLossMatrix} schedulers={SCHEDULERS} />
                </div>
            </div>

            {/* Performance Heatmap */}
            <div className="col-span-12 md:col-span-6 glass-panel p-6" style={{ height: 450 }}>
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <Activity size={16} className="text-red-400" /> PERFORMANCE HEATMAP
                </h2>
                <div className="flex justify-center" style={{ height: 380 }}>
                    <HeatmapChart data={heatmapData} xLabel="Task Size" yLabel="Compute Intensity" />
                </div>
            </div>

            {/* Task Distribution Histogram */}
            <div className="col-span-12 md:col-span-6 glass-panel p-6" style={{ height: 400 }}>
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <Database size={16} className="text-purple-400" /> TASK SIZE DISTRIBUTION
                </h2>
                <div style={{ height: 330 }}>
                    <TaskDistributionHistogram tasks={history || []} bins={12} />
                </div>
            </div>

            {/* Correlation Matrix */}
            <div className="col-span-12 md:col-span-6 glass-panel p-6" style={{ height: 400 }}>
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <Activity size={16} className="text-blue-400" /> FEATURE CORRELATION MATRIX
                </h2>
                <div className="flex justify-center" style={{ height: 330 }}>
                    <CorrelationMatrix correlations={correlationData.correlations} features={correlationData.features} />
                </div>
            </div>

            {/* Cumulative Performance */}
            <div className="col-span-12 glass-panel p-6" style={{ height: 450 }}>
                <h2 className="text-sm font-bold text-gray-300 mb-4 flex items-center gap-2">
                    <Layers size={16} className="text-blue-400" /> CUMULATIVE PERFORMANCE
                </h2>
                <div style={{ height: 380 }}>
                    <CumulativeChart
                        data={cumulativeData}
                        dataKeys={SCHEDULERS.map(s => s.id)}
                        colors={Object.fromEntries(SCHEDULERS.map(s => [s.id, s.color]))}
                    />
                </div>
            </div>

        </div>
    );
};

export default EnhancedVisualizationsDemo;
