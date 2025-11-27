import React from 'react';

const GaugeChart = ({ value, max = 100, label, color = '#06b6d4', size = 120 }) => {
    const percentage = Math.min(100, (value / max) * 100);
    const angle = (percentage / 100) * 180; // 0-180 degrees for semi-circle

    // Calculate needle position
    const needleLength = size * 0.35;
    const centerX = size / 2;
    const centerY = size / 2;
    const angleRad = ((angle - 90) * Math.PI) / 180;
    const needleX = centerX + needleLength * Math.cos(angleRad);
    const needleY = centerY + needleLength * Math.sin(angleRad);

    return (
        <div className="flex flex-col items-center">
            <svg width={size} height={size * 0.6} viewBox={`0 0 ${size} ${size * 0.6}`}>
                {/* Background arc */}
                <path
                    d={`M ${size * 0.1} ${size / 2} A ${size * 0.4} ${size * 0.4} 0 0 1 ${size * 0.9} ${size / 2}`}
                    fill="none"
                    stroke="#374151"
                    strokeWidth={size * 0.08}
                    strokeLinecap="round"
                />

                {/* Progress arc */}
                <path
                    d={`M ${size * 0.1} ${size / 2} A ${size * 0.4} ${size * 0.4} 0 0 1 ${size * 0.9} ${size / 2}`}
                    fill="none"
                    stroke={color}
                    strokeWidth={size * 0.08}
                    strokeLinecap="round"
                    strokeDasharray={`${(percentage / 100) * Math.PI * size * 0.4} ${Math.PI * size * 0.4}`}
                    style={{ filter: `drop-shadow(0 0 ${size * 0.05}px ${color})` }}
                />

                {/* Center circle */}
                <circle cx={centerX} cy={centerY} r={size * 0.04} fill="#18181b" stroke={color} strokeWidth={2} />

                {/* Needle */}
                <line
                    x1={centerX}
                    y1={centerY}
                    x2={needleX}
                    y2={needleY}
                    stroke={color}
                    strokeWidth={size * 0.015}
                    strokeLinecap="round"
                    style={{ filter: `drop-shadow(0 0 ${size * 0.02}px ${color})` }}
                />

                {/* Value text */}
                <text
                    x={centerX}
                    y={centerY + size * 0.15}
                    textAnchor="middle"
                    className="font-bold"
                    style={{ fontSize: size * 0.15, fill: '#fff' }}
                >
                    {value.toFixed(1)}
                </text>

                {/* Min/Max labels */}
                <text x={size * 0.05} y={size * 0.55} style={{ fontSize: size * 0.08, fill: '#6b7280' }}>0</text>
                <text x={size * 0.88} y={size * 0.55} textAnchor="end" style={{ fontSize: size * 0.08, fill: '#6b7280' }}>{max}</text>
            </svg>

            {label && (
                <div className="text-xs text-gray-400 mt-1 font-semibold text-center">{label}</div>
            )}
        </div>
    );
};

export default GaugeChart;
