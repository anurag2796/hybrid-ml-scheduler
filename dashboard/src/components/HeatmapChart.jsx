import React from 'react';

const HeatmapChart = ({ data, xLabel, yLabel, title }) => {
    // data format: [{ x: value, y: value, value: intensity }]

    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                No data available
            </div>
        );
    }

    // Get unique x and y values
    const xValues = [...new Set(data.map(d => d.x))].sort((a, b) => a - b);
    const yValues = [...new Set(data.map(d => d.y))].sort((a, b) => a - b);

    // Find min/max for color scaling
    const values = data.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);

    // Color interpolation
    const getColor = (value) => {
        const normalized = (value - minValue) / (maxValue - minValue || 1);

        // Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
        if (normalized < 0.25) {
            const t = normalized / 0.25;
            return `rgb(${Math.round(59 + (6 - 59) * t)}, ${Math.round(130 + (182 - 130) * t)}, ${Math.round(246 + (212 - 246) * t)})`;
        } else if (normalized < 0.5) {
            const t = (normalized - 0.25) / 0.25;
            return `rgb(${Math.round(6 + (16 - 6) * t)}, ${Math.round(182 + (185 - 182) * t)}, ${Math.round(212 + (129 - 212) * t)})`;
        } else if (normalized < 0.75) {
            const t = (normalized - 0.5) / 0.25;
            return `rgb(${Math.round(16 + (251 - 16) * t)}, ${Math.round(185 + (191 - 185) * t)}, ${Math.round(129 + (36 - 129) * t)})`;
        } else {
            const t = (normalized - 0.75) / 0.25;
            return `rgb(${Math.round(251 + (239 - 251) * t)}, ${Math.round(191 + (68 - 191) * t)}, ${Math.round(36 + (68 - 36) * t)})`;
        }
    };

    const cellWidth = 40;
    const cellHeight = 30;
    const paddingLeft = 60;
    const paddingTop = 20;
    const paddingBottom = 40;

    const width = paddingLeft + xValues.length * cellWidth + 20;
    const height = paddingTop + yValues.length * cellHeight + paddingBottom;

    return (
        <div className="overflow-auto">
            {title && <h3 className="text-sm font-bold text-gray-300 mb-2">{title}</h3>}
            <svg width={width} height={height}>
                {/* Heatmap cells */}
                {yValues.map((y, yi) => (
                    xValues.map((x, xi) => {
                        const cell = data.find(d => d.x === x && d.y === y);
                        const value = cell ? cell.value : 0;

                        return (
                            <g key={`${x}-${y}`}>
                                <rect
                                    x={paddingLeft + xi * cellWidth}
                                    y={paddingTop + yi * cellHeight}
                                    width={cellWidth - 1}
                                    height={cellHeight - 1}
                                    fill={cell ? getColor(value) : '#1f2937'}
                                    stroke="#000"
                                    strokeWidth={0.5}
                                />
                                <text
                                    x={paddingLeft + xi * cellWidth + cellWidth / 2}
                                    y={paddingTop + yi * cellHeight + cellHeight / 2 + 4}
                                    textAnchor="middle"
                                    style={{ fontSize: 10, fill: '#fff', fontWeight: 600 }}
                                >
                                    {cell ? value.toFixed(2) : '-'}
                                </text>
                            </g>
                        );
                    })
                ))}

                {/* X-axis labels */}
                {xValues.map((x, xi) => (
                    <text
                        key={`x-${x}`}
                        x={paddingLeft + xi * cellWidth + cellWidth / 2}
                        y={height - paddingBottom + 15}
                        textAnchor="middle"
                        style={{ fontSize: 10, fill: '#9ca3af' }}
                    >
                        {x.toFixed(1)}
                    </text>
                ))}

                {/* Y-axis labels */}
                {yValues.map((y, yi) => (
                    <text
                        key={`y-${y}`}
                        x={paddingLeft - 10}
                        y={paddingTop + yi * cellHeight + cellHeight / 2 + 4}
                        textAnchor="end"
                        style={{ fontSize: 10, fill: '#9ca3af' }}
                    >
                        {y.toFixed(1)}
                    </text>
                ))}

                {/* Axis labels */}
                <text
                    x={width / 2}
                    y={height - 5}
                    textAnchor="middle"
                    style={{ fontSize: 11, fill: '#6b7280', fontWeight: 600 }}
                >
                    {xLabel}
                </text>

                <text
                    x={15}
                    y={height / 2}
                    textAnchor="middle"
                    transform={`rotate(-90, 15, ${height / 2})`}
                    style={{ fontSize: 11, fill: '#6b7280', fontWeight: 600 }}
                >
                    {yLabel}
                </text>

                {/* Legend */}
                <g transform={`translate(${width - 60}, ${paddingTop})`}>
                    {[0, 0.25, 0.5, 0.75, 1].map((normalized, i) => {
                        const value = minValue + normalized * (maxValue - minValue);
                        return (
                            <g key={i}>
                                <rect
                                    x={0}
                                    y={i * 15}
                                    width={20}
                                    height={14}
                                    fill={getColor(value)}
                                    stroke="#000"
                                    strokeWidth={0.5}
                                />
                                <text
                                    x={25}
                                    y={i * 15 + 10}
                                    style={{ fontSize: 8, fill: '#9ca3af' }}
                                >
                                    {value.toFixed(2)}
                                </text>
                            </g>
                        );
                    })}
                </g>
            </svg>
        </div>
    );
};

export default HeatmapChart;
