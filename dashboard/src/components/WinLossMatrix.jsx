import React from 'react';

const WinLossMatrix = ({ matrix, schedulers }) => {
    // matrix: { scheduler_a: { scheduler_b: 0.65 (win rate) } }

    if (!matrix || Object.keys(matrix).length === 0) {
        return (
            <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
                Insufficient data for comparison
            </div>
        );
    }

    const schedulerIds = Object.keys(matrix);
    const cellSize = 60;
    const padding = 80;
    const size = schedulerIds.length * cellSize + padding;

    const getColor = (winRate) => {
        if (winRate === null || winRate === undefined) return '#1f2937';

        // Red (0%) -> Yellow (50%) -> Green (100%)
        if (winRate < 0.5) {
            const t = winRate / 0.5;
            // Red to Yellow
            return `rgb(${Math.round(239 + (251 - 239) * t)}, ${Math.round(68 + (191 - 68) * t)}, 68)`;
        } else {
            const t = (winRate - 0.5) / 0.5;
            // Yellow to Green
            return `rgb(${Math.round(251 - (251 - 16) * t)}, ${Math.round(191 - (191 - 185) * t)}, ${Math.round(68 + (129 - 68) * t)})`;
        }
    };

    return (
        <div className="overflow-auto">
            <svg width={size} height={size}>
                {/* Grid */}
                {schedulerIds.map((rowId, ri) => (
                    schedulerIds.map((colId, ci) => {
                        const winRate = rowId === colId ? null : matrix[rowId]?.[colId];
                        const scheduler = schedulers?.find(s => s.id === rowId);

                        return (
                            <g key={`${rowId}-${colId}`}>
                                <rect
                                    x={padding + ci * cellSize}
                                    y={padding + ri * cellSize}
                                    width={cellSize - 1}
                                    height={cellSize - 1}
                                    fill={rowId === colId ? '#27272a' : getColor(winRate)}
                                    stroke="#000"
                                    strokeWidth={0.5}
                                />
                                {rowId !== colId && winRate !== null && winRate !== undefined && (
                                    <text
                                        x={padding + ci * cellSize + cellSize / 2}
                                        y={padding + ri * cellSize + cellSize / 2 + 5}
                                        textAnchor="middle"
                                        style={{ fontSize: 12, fill: '#fff', fontWeight: 700 }}
                                    >
                                        {(winRate * 100).toFixed(0)}%
                                    </text>
                                )}
                                {rowId === colId && (
                                    <text
                                        x={padding + ci * cellSize + cellSize / 2}
                                        y={padding + ri * cellSize + cellSize / 2 + 3}
                                        textAnchor="middle"
                                        style={{ fontSize: 16, fill: '#4b5563' }}
                                    >
                                        —
                                    </text>
                                )}
                            </g>
                        );
                    })
                ))}

                {/* Row labels */}
                {schedulerIds.map((id, i) => {
                    const scheduler = schedulers?.find(s => s.id === id);
                    return (
                        <text
                            key={`row-${id}`}
                            x={padding - 10}
                            y={padding + i * cellSize + cellSize / 2 + 4}
                            textAnchor="end"
                            style={{ fontSize: 11, fill: scheduler?.color || '#9ca3af', fontWeight: 600 }}
                        >
                            {scheduler?.name || id}
                        </text>
                    );
                })}

                {/* Column labels */}
                {schedulerIds.map((id, i) => {
                    const scheduler = schedulers?.find(s => s.id === id);
                    return (
                        <text
                            key={`col-${id}`}
                            x={padding + i * cellSize + cellSize / 2}
                            y={padding - 10}
                            textAnchor="middle"
                            style={{ fontSize: 11, fill: scheduler?.color || '#9ca3af', fontWeight: 600 }}
                        >
                            {scheduler?.name || id}
                        </text>
                    );
                })}

                {/* Labels */}
                <text
                    x={padding / 2}
                    y={size / 2}
                    textAnchor="middle"
                    transform={`rotate(-90, ${padding / 2}, ${size / 2})`}
                    style={{ fontSize: 12, fill: '#6b7280', fontWeight: 700 }}
                >
                    Wins Against →
                </text>

                <text
                    x={size / 2}
                    y={padding / 2}
                    textAnchor="middle"
                    style={{ fontSize: 12, fill: '#6b7280', fontWeight: 700 }}
                >
                    ← Opponent
                </text>
            </svg>
        </div>
    );
};

export default WinLossMatrix;
