/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Cyberpunk Palette
                bg: {
                    dark: '#050505',
                    panel: 'rgba(24, 24, 27, 0.6)',
                },
                primary: {
                    DEFAULT: '#00f3ff', // Neon Cyan
                    dim: 'rgba(0, 243, 255, 0.1)',
                },
                secondary: {
                    DEFAULT: '#bc13fe', // Neon Purple
                    dim: 'rgba(188, 19, 254, 0.1)',
                },
                accent: {
                    DEFAULT: '#ff0055', // Neon Pink
                },
                success: {
                    DEFAULT: '#0aff00', // Neon Green
                }
            },
            fontFamily: {
                mono: ['"JetBrains Mono"', 'monospace'],
                sans: ['"Rajdhani"', 'sans-serif'],
            },
            boxShadow: {
                'neon-cyan': '0 0 10px rgba(0, 243, 255, 0.5), 0 0 20px rgba(0, 243, 255, 0.3)',
                'neon-purple': '0 0 10px rgba(188, 19, 254, 0.5), 0 0 20px rgba(188, 19, 254, 0.3)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'scanline': 'scanline 8s linear infinite',
            },
            keyframes: {
                scanline: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(100%)' },
                }
            }
        },
    },
    plugins: [],
}
