# 06. Current Challenges

Below are the known limitations, tech debt, and immediate TODOs for the Hybrid ML Scheduler ecosystem.

## Known Challenges & Bugs

* **Initial RL Agent Latency Penalties**: The Reinforcement Learning (DQN) Agent often makes terrible decisions for the first hundred loops (exploration phase), exploding the cluster cost metrics.
  * **Workaround**: We instituted a hacky Heuristic `pretrain()` algorithm that simulates 1000 tasks statically to seed the Experience Replay.
* **Hybrid ML Scikit-Learn Bootstrapping**: Similar to the RL Agent, the Random Forest model cannot infer answers on `Task 1`. 
  * **Workaround**: Currently injecting fake manual data into `_initial_pretrain()` as scaffolding.
* **Cost Dominance over MakeSpan**: The current mathematical formulation penalizes GPUs by 5x (cost scaling from AWS). Because CPU execution is artificially 5x cheaper, both ML models frequently default to exclusively scheduling on CPU to farm positive evaluation scores, dragging the total system Makespan.
  * **TODO**: Abstract out the Evaluation reward variables into user-adjustable levers rather than hardcoded metrics in `simulator.py` power models.

## Backlog / Technical Debt
* Modify the React dashboard WebSocket payload to shrink the massive array broadcasting (Current implementation slows down the Vite frontend heavily during long-term continuous 10H+ scaling runs).
* Migrate SQLite `data/long_term_history.csv` to full-scale distributed PostgreSQL schema logs in Azure.
* DAG Data structures: `Task`s currently run homogeneously and independently. The engine needs a DAG topological sort dependency map logic overhaul.
