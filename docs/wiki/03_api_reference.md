# 03. API Reference

This document covers the APIs exposed by the Hybrid ML Scheduler backend for the dashboard and interaction.

## REST Endpoints (`/api/*`)

*   **`GET /api/status`**
    *   **Returns**: `{"is_running": true, "tasks_processed": 500, ...}`
    *   **Description**: Used by the React dashboard as a heartbeat check.

*   **`POST /api/pause`**
    *   **Action**: Suspends the simulation loops inside the `ContinuousSimulation` class, halting Workload execution.

*   **`POST /api/resume`**
    *   **Action**: Resumes the Workload execution and task simulation loop.

*   **`GET /api/history/comparative`**
    *   **Returns**: JSON array of detailed scheduler results suitable for long-term historical charting.
    *   **Parameters**: Supports a `limit` query string to constrain the returned batch of records.

*   **`GET /api/full_history`**
    *   **Returns**: Retrieve complete historical training data.
    *   **Format**: `[{"task_id": 0, "size": 960, "compute_intensity": 0.26, ...}, ...]`

*   **`DELETE /api/history`**
    *   **Action**: Clear all historical data from PostgreSQL.

## WebSocket Protocol (`ws://localhost:8000/ws`)

The dashboard connects to this endpoint to receive real-time streams of simulated task outputs.

*   **Update Frequency**: Dispatched approximately 2x per second (Every 500ms).
*   **Payload Format**:
    ```json
    {
      "type": "simulation_update",
      "task": { 
        "id": 101, 
        "size": 960,
        "intensity": 0.9,
        "memory": 81
      },
      "comparison": [ 
        {"name": "rl_agent", "avg_time": 1.2}, 
        ... 
      ],
      "latest_results": { 
        "hybrid_ml": {"time": 0.45, "energy": 22.5, "cost": 0.0009},
        "oracle": {"time": 0.42, "energy": 21.0, "cost": 0.0008}
      },
      "utilization": {
         "average_utilization": 0.65,
         "gpu_0": {"utilization": 0.72}
      }
    }
    ```
