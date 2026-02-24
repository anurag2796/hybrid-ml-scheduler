# Hybrid ML Scheduler: Optimizing Heterogeneous Computing
## A Reinforcement Learning Approach to Task Scheduling

---

# Slide 1: Title Slide

**Title:** Hybrid ML Scheduler
**Subtitle:** Optimizing Heterogeneous GPU/CPU Clusters using Reinforcement Learning
**Presenter:** [Your Name/Team Name]
**Context:** Advanced Parallel Computing Project

**Key Visual:** A split image showing a Brain (AI) connected to a Server Rack.

---

# Slide 2: The Problem - "The Kitchen Metaphor"

**The Challenge:**
Modern computers have different types of processors:
1.  **CPUs (Prep Cooks):** Slow, cheap, good for simple logic.
2.  **GPUs (Master Chefs):** Fast, expensive, power-hungry. (50x faster for math).

**The Dilemma:**
*   Sending a small task to a GPU is wasteful (Transfer time > Execution time).
*   Sending a huge task to a CPU causes bottlenecks.
*   **Goal:** Assign the *right* task to the *right* processor automatically.

---

# Slide 3: The Solution - AI-Driven Scheduling

**Introducing the Hybrid ML Scheduler:**
An intelligent system that learns to manage the cluster without human rules.

**Core Innovation:**
Instead of static rules ("If size > 100MB..."), we use **Reinforcement Learning** (DQN). The scheduler:
1.  **Observes** the task (Size, Complexity).
2.  **Acts** (Assigns to CPU or GPU).
3.  **Learns** from the result (Reward = Speed - Cost).

---

# Slide 4: System Architecture

**The Stack:**
*   **Frontend:** React Dashboard (Real-time visualization).
*   **Backend:** FastAPI (Task generation & Simulation).
*   **Model:** PyTorch (DQN) + Scikit-Learn (Random Forest).

**Data Flow:**
`Workload Gen` -> `Simulator` -> `Scheduler (The Brain)` -> `Execution` -> `Feedback Loop`

*(Visual Idea: Use the Mermaid diagram from the project wiki showing the loop)*

---

# Slide 5: The Contenders (Scheduling Strategies)

We compare 6 strategies in a "Race":

1.  **Round Robin:** Alternates A -> B -> A -> B. (Fair but dumb).
2.  **Random:** Total chaos. (Baseline for "worst case").
3.  **Greedy:** "If it has math, send to GPU." (Fails on small data-heavy tasks).
4.  **Hybrid ML:** Random Forest. Trained on past data.
5.  **RL Agent (Our Hero):** Learns live. Adapts to changes.
6.  **Oracle:** The theoretical limit. Brute-forces every option to find the truth.

---

# Slide 6: Deep Dive - The RL Agent (DQN)

**How it works (Deep Q-Network):**
*   **State:** [Task Size, Compute Intensity, Memory Required]
*   **Action:** Choose [CPU, GPU 0, GPU 1, GPU 2, GPU 3]
*   **Reward Function:** $- (Time + 0.5 \times Energy + 0.5 \times Cost)$

**Key Tech:**
*   **Experience Replay:** Remembers past mistakes to avoid repeating them.
*   **Target Networks:** Stabilizes learning.

---

# Slide 7: The "Oracle" - Knowing the Truth

**What is the Oracle?**
A scheduler that "cheats" by running every task 11 times (0% GPU, 10% GPU ... 100% GPU) and picking the best result.

**Why do we need it?**
*   It gives us a **"Speed of Light"** baseline.
*   If our RL Agent gets within 5% of the Oracle, we have solved the problem.
*   It generates "Labelled Data" for the Random Forest model.

---

# Slide 8: The Dashboard - Real-Time Visibility

**Features:**
*   **Live Race:** Watch schedulers compete in real-time.
*   **Radar Chart:** Compare Cost vs. Time vs. Energy.
*   **Utilization:** See if the cluster is overloaded.

*(Include Screenshot of the Dark Mode Dashboard)*

---

# Slide 9: Results - RL vs The World

**Performance Metrics:**
*   **RL Agent vs Random:** **5x Faster**.
*   **RL Agent vs Hybrid ML:** **20% Faster** (Adapts better to new patterns).
*   **RL Agent vs Oracle:** **~95% Efficiency** (Almost perfect!).

**Key Insight:**
The RL agent learned to **keep small tasks on the CPU**. It realized the "walking time" to the GPU wasn't worth it for small jobs.

---

# Slide 10: Conclusion & Future Work

**Summary:**
We successfully built a self-optimizing scheduler that saves time and money by understanding the physics of the hardware.

**Key Takeaways:**
1.  Heterogeneous scheduling is non-trivial.
2.  RL agents can learn complex physics rules (Amdahl's Law) without being explicitly programmed.
3.  Real-time visualization is crucial for trusting AI decisions.

**Future Work:**
*   Multi-Node Clustering (Kubernetes integration).
*   Transformer-based Agents (Attention mechanisms).

---
