import React from 'react';

const CorrelationMatrix = ({ correlations, features }) => {
    // correlations: { feature_a: { feature_b: 0.85 } }
    // features: ['size', 'intensity', 'memory', 'optimal_time']

    if (!correlations || !features || features.length === 0) {
        return (
            <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
                No correlation data available
            </div>
        );
    }

    const cellSize = 50;
    const padding = 100;
    const size = features.length * cellSize + padding;

    const getColor = (correlation) => {
        if (correlation === null || correlation === undefined) return '#1f2937';

        // Blue (negative) -> White (0) -> Red (positive)
        const abs = Math.abs(correlation);

        if (correlation < 0) {
            // Negative correlation: darker blue
            return `rgb(${Math.round(59 * (1 - abs))}, ${Math.round(130 * (1 - abs))}, ${Math.round(246)})`;
        } else {
            // Positive correlation: darker red
            return `rgb(${Math.round(239)}, ${Math.round(68 * (1 - abs))}, ${Math.round(68 * (1 - abs))})`;
        }
    };

    const featureLabels = {
        size: 'Task Size',
        intensity: 'Compute Int.',
        compute_intensity: 'Compute Int.',
        memory: 'Memory',
        memory_required: 'Memory',
        optimal_time: 'Opt. Time',
        optimal_gpu_fraction: 'GPU Frac.'
    };

    return (
        <div className="overflow-auto">
            <svg width={size} height={size}>
                {/* Grid */}
                {features.map((rowFeature, ri) => (
                    features.map((colFeature, ci) => {
                        const corr = rowFeature === colFeature ? 1 :
                            (correlations[rowFeature]?.[colFeature] ?? correlations[colFeature]?.[rowFeature] ?? 0);

                        return (
                            <g key={`${rowFeature}-${colFeature}`}>
                                <rect
                                    x={padding + ci * cellSize}
                                    y={padding + ri * cellSize}
                                    width={cellSize - 1}
                                    height={cellSize - 1}
                                    fill={getColor(corr)}
                                    stroke="#000"
                                    strokeWidth={0.5}
                                />
                                <text
                                    x={padding + ci * cellSize + cellSize / 2}
                                    y={padding + ri * cellSize + cellSize / 2 + 4}
                                    textAnchor="middle"
                                    style={{
                                        fontSize: 11,
                                        fill: Math.abs(corr) > 0.5 ? '#fff' : '#9ca3af',
                                        fontWeight: 700
                                    }}
                                >
                                    {corr.toFixed(2)}
                                </text>
                            </g>
                        );
                    })
                ))}

                {/* Row labels */}
                {features.map((feature, i) => (
                    <text
                        key={`row-${feature}`}
                        x={padding - 10}
                        y={padding + i * cellSize + cellSize / 2 + 4}
                        textAnchor="end"
                        style={{ fontSize: 10, fill: '#9ca3af', fontWeight: 600 }}
                    >
                        {featureLabels[feature] || feature}
                    </text>
                ))}

                {/* Column labels */}
                {features.map((feature, i) => (
                    <text
                        key={`col-${feature}`}
                        x={padding + i * cellSize + cellSize / 2}
                        y={padding - 10}
                        textAnchor="end"
                        transform={`rotate(-45, ${padding + i * cellSize + cellSize / 2}, ${padding - 10})`}
                        style={{ fontSize: 10, fill: '#9ca3af', fontWeight: 600 }}
                    >
                        {featureLabels[feature] || feature}
                    </text>
                ))}

                {/* Legend */}
                <g transform={`translate(${size - 80}, ${padding + 20})`}>
                    <text x={0} y={-10} style={{ fontSize: 10, fill: '#6b7280', fontWeight: 600 }}>
                        Correlation
                    </text>
                    {[-1, -0.5, 0, 0.5, 1].map((val, i) => (
                        <g key={i}>
                            <rect
                                x={0}
                                y={i * 18}
                                width={25}
                                height={16}
                                fill={getColor(val)}
                                stroke="#000"
                                strokeWidth={0.5}
                            />
                            <text
                                x={30}
                                y={i * 18 + 11}
                                style={{ fontSize: 9, fill: '#9ca3af' }}
                            >
                                {val.toFixed(1)}
                            </text>
                        </g>
                    ))}
                </g>
            </svg>
        </div>
    );
};

export default CorrelationMatrix;
