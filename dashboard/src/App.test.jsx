import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock Recharts to avoid rendering issues in JSDOM
vi.mock('recharts', () => {
    const OriginalModule = vi.importActual('recharts');
    return {
        ...OriginalModule,
        ResponsiveContainer: ({ children }) => <div className="recharts-responsive-container" style={{ width: 800, height: 800 }}>{children}</div>,
        AreaChart: () => <div data-testid="area-chart">AreaChart</div>,
        Area: () => null,
        XAxis: () => null,
        YAxis: () => null,
        CartesianGrid: () => null,
        Tooltip: () => null,
        BarChart: () => <div data-testid="bar-chart">BarChart</div>,
        Bar: () => null,
        Cell: () => null,
        RadarChart: () => <div data-testid="radar-chart">RadarChart</div>,
        PolarGrid: () => null,
        PolarAngleAxis: () => null,
        PolarRadiusAxis: () => null,
        Radar: () => null,
        ScatterChart: () => <div data-testid="scatter-chart">ScatterChart</div>,
        Scatter: () => null,
        PieChart: () => <div data-testid="pie-chart">PieChart</div>,
        Pie: () => null,
        Legend: () => null,
        LineChart: () => <div data-testid="line-chart">LineChart</div>,
        Line: () => null,
    };
});

// Mock react-grid-layout
vi.mock('react-grid-layout', () => ({
    Responsive: ({ children }) => <div data-testid="grid-layout">{children}</div>,
    WidthProvider: (Component) => Component,
}));

describe('Dashboard Component', () => {
    it('renders the dashboard title', () => {
        render(<App />);
        expect(screen.getByText(/SCHEDULER LAB/i)).toBeInTheDocument();
        expect(screen.getByText(/VIRTUAL CLUSTER MANAGER/i)).toBeInTheDocument();
    });

    it('renders the sidebar with navigation options', () => {
        render(<App />);
        expect(screen.getAllByText(/Global Comparison/i)[0]).toBeInTheDocument();
        expect(screen.getByText(/Historical Analysis/i)).toBeInTheDocument();
        expect(screen.getByText(/Hybrid ML/i)).toBeInTheDocument();
        expect(screen.getByText(/RL Agent/i)).toBeInTheDocument();
    });

    it('switches to Global Comparison view by default', () => {
        render(<App />);
        expect(screen.getByText(/PERFORMANCE RACE/i)).toBeInTheDocument();
        expect(screen.getByText(/WORKLOAD DISTRIBUTION/i)).toBeInTheDocument();
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
        expect(screen.getByTestId('scatter-chart')).toBeInTheDocument();
    });

    it('switches to Historical Analysis view when clicked', async () => {
        render(<App />);
        const historyButton = screen.getByText(/Historical Analysis/i);
        fireEvent.click(historyButton);

        await waitFor(() => {
            expect(screen.getByText(/Historical Analysis Mode/i)).toBeInTheDocument();
        });
        expect(screen.getByText(/Clear Data/i)).toBeInTheDocument();
    });

    it('switches to Scheduler view (Hybrid ML) when clicked', async () => {
        render(<App />);
        const hybridMLButton = screen.getByText(/Hybrid ML/i);
        fireEvent.click(hybridMLButton);

        await waitFor(() => {
            expect(screen.getByText(/VIRTUAL CLUSTER LOAD/i)).toBeInTheDocument();
        });
        expect(screen.getByText(/VS ORACLE BASELINE/i)).toBeInTheDocument();
        expect(screen.getByText(/RESOURCE SPLIT/i)).toBeInTheDocument();
        expect(screen.getByText(/HYBRID ML LOGS/i)).toBeInTheDocument();
    });

    it('toggles simulation pause/resume', () => {
        render(<App />);
        const pauseButton = screen.getByText(/PAUSE SIM/i);
        expect(pauseButton).toBeInTheDocument();

        // Mock fetch for API call
        global.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({}) }));

        fireEvent.click(pauseButton);
        expect(screen.getByText(/RESUME SIM/i)).toBeInTheDocument();
        expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/pause'), expect.anything());

        fireEvent.click(screen.getByText(/RESUME SIM/i));
        expect(screen.getByText(/PAUSE SIM/i)).toBeInTheDocument();
        expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/resume'), expect.anything());
    });
});
