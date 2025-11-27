# Incremental Backend Migration Plan

## ✅ Phase 1: Foundation (COMPLETED)
- [x] Database setup (PostgreSQL)
- [x] Configuration management (Pydantic Settings)
- [x] Database models (SQLAlchemy)
- [x] Pydantic schemas
- [x] Redis client setup
- [x] Database initialized with 5 tables

## 🚀 Phase 2: Data Layer Migration (CURRENT)

### Step 1: Create Repository Layer
Create repository classes to abstract database operations:
- `TaskRepository` - CRUD for tasks
- `SchedulerResultRepository` - Store scheduler results
- `TrainingDataRepository` - Manage training data
- `MetricRepository` - Aggregate metrics

### Step 2: Migrate CSV Writing to Database
Update `simulation_engine.py`:
- Replace CSV writes to database inserts
- Use batching for performance
- Keep CSV export option for compatibility

### Step 3: Test Data Flow
- Verify data is being written to database
- Check performance (should be faster than CSV)
- Ensure no data loss

## 📊 Phase 3: API Refactoring

### Step 4: Create API Routes
Split `dashboard_server.py` into modules:
- `routes/health.py` - Health check endpoints
- `routes/history.py` - Historical data endpoints  
- `routes/simulation.py` - Simulation control
- `routes/websocket.py` - WebSocket connections

### Step 5: Add Service Layer
- `SimulationService` - Business logic for simulation
- `DataService` - Data retrieval and aggregation
- `CacheService` - Redis caching logic

### Step 6: Update Dashboard Server
- Use new routes and services
- Add request validation
- Implement proper error handling

## ⚡ Phase 4: Performance Optimization

### Step 7: Add Redis Caching
- Cache recent metrics (last 100 results)
- Cache scheduler leaderboard
- Implement cache invalidation

### Step 8: Optimize Database Queries
- Add query result caching
- Use database indexes effectively  
- Batch operations

## 📈 Phase 5: Observability

### Step 9: Add Prometheus Metrics
- Request latency
- WebSocket connections
- Database query times
- Error rates

### Step 10: Add Health Checks
- `/health` endpoint
- `/ready` endpoint
- Database connectivity check
- Redis connectivity check

## 🔒 Phase 6: Security & Polish

### Step 11: Security Headers
- CORS configuration
- Rate limiting
- Input validation

### Step 12: Testing
- Integration tests
- Performance tests
- End-to-end tests

## 📝 Implementation Schedule

**Today (Phase 2 - 1 hour):**
1. Create repositories ✓
2. Migrate simulation_engine to use database ✓
3. Test data flow ✓

**Next Session (Phase 3 - 1 hour):**
4. Refactor API routes
5. Add service layer
6. Update dashboard server

**Future Sessions:**
7. Caching & optimization
8. Observability
9. Security & testing
