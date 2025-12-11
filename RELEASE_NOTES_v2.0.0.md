# Release Notes v2.0.0 - Major Update

**Date:** November 27, 2025
**Version:** 2.0.0

## 🚀 Major Release Highlights

This release marks a significant milestone in the Hybrid ML Scheduler project, transitioning from a prototype to a production-ready system. It introduces a fully refactored backend architecture, enhanced security, persistent storage, and a comprehensive observability suite.

### 🏗️ Architectural Overhaul
- **Modular Backend:** Refactored monolithic code into a layered architecture (API, Services, Repositories, Models).
- **Async Database:** Integrated PostgreSQL with `asyncpg` and SQLAlchemy for high-performance, non-blocking data persistence.
- **Redis Caching:** Implemented Redis caching layer for high-frequency data access (training data, scheduler stats).
- **Pydantic Configuration:** Centralized configuration management using Pydantic Settings with environment variable support.

### 🔒 Security & Reliability
- **Rate Limiting:** Added token-bucket rate limiting middleware backed by Redis.
- **Security Headers:** Implemented OWASP-recommended security headers (HSTS, CSP, etc.).
- **Input Validation:** Added comprehensive input validation and sanitization (SQLi, XSS prevention).
- **Robust Error Handling:** Standardized error responses and global exception handling.

### 📊 Enhanced Observability & Dashboard
- **Structured Logging:** Implemented JSON structured logging with correlation IDs for request tracing.
- **Real-time Metrics:** Exposed Prometheus metrics for monitoring system health and performance.
- **Advanced Visualizations:** Added new "Enhanced Analytics" and "Historical Analytics" views to the dashboard.
- **Interactive Charts:** Included Heatmaps, Win/Loss Matrices, and Correlation Matrices for deep performance analysis.

### 🧠 Simulation & ML Improvements
- **Continuous Simulation:** Upgraded simulation engine to run continuously with real-time task generation.
- **Online Retraining:** Implemented automated model retraining using a sliding window of historical data.
- **Data Persistence:** All simulation data and scheduler decisions are now persisted to the database for long-term analysis.

## 🛠️ Technical Details

### Dependencies Added
- `fastapi`, `uvicorn[standard]`
- `sqlalchemy[asyncio]`, `asyncpg`
- `redis`
- `pydantic-settings`
- `loguru`
- `prometheus-client`

### Database Schema
- **Tables:** `tasks`, `scheduler_results`, `training_data`, `simulation_state`
- **Indexes:** Optimized indexes for time-series queries and frequent lookups.

## 📝 Upgrade Instructions
1. **Environment Setup:**
   ```bash
   cp .env.example .env
   # Update .env with your database and redis credentials
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize Database:**
   ```bash
   python scripts/init_db.py
   ```
4. **Run Server:**
   ```bash
   python src/dashboard_server_v2.py
   ```

---

