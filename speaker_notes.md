
# 🎤 Project Presentation: Hybrid ML Scheduler
**Speaker Notes & Walkthrough Guide**

---

## 🕒 Time Estimate: ~10 Minutes
**Goal:** Explain how we optimize task scheduling using AI, in simple English.

---

## 1. Introduction (1 Minute)
**Objective:** Hook the audience.

*   "Hello everyone. Today I am presenting my project: **Hybrid ML Scheduler**."
*   "In modern computing, we have many tasks (like math calculations) and many resources (CPUs and GPUs). The big problem is: **Where should we put each task?**"
*   "If we put a small task on a big GPU, we waste energy. If we put a big task on a slow CPU, it takes forever. We need a smart way to decide."
*   "My solution uses two types of AI:
    1.  **Hybrid ML (Random Forest):** A teacher that learns from perfect examples.
    2.  **RL Agent (Deep Q-Network):** A student that learns by trying and failing."

---

## 2. How We Create Tasks (The Workload Generator) (2 Minutes)
**Objective:** Show that the simulation is realistic.
**File:** `src/workload_generator.py`

*   "First, let's look at how we simulate work. Please open `src/workload_generator.py`."
*   "Go to **Line 132**. Here we create a `Task` object. Every task has a Size and 'Intensity' (how much math it needs)."
*   "Look at **Lines 100-102**:
    ```python
    100:                 # GPU-bound peak
    101:                 compute_intensity = np.random.normal(0.8, 0.1)
    102:             compute_intensity = np.clip(compute_intensity, 0.05, 1.0)
    ```
    *   **Simple Explanation:** We create random tasks. Some are very heavy (Intensity 1.0), which should go to GPU. Some are light (Intensity 0.1), which should stay on CPU."
*   "Look at **Line 126**:
    ```python
    126:                 base_duration = (size / 1000) ** 1.5
    ```
    *   **Simple Explanation:** Bigger tasks take longer to finish. We simulate this math here."

---

## 3. The Reinforcement Learning (RL) Agent (2 Minutes)
**Objective:** Explain the "Student" AI.
**Concept:** It learns by trial and error.
**File:** `src/dqn_scheduler.py`

*   "Now, the star of the show: The RL Agent. This comes from 'Deep Q-Learning'."
*   "Open `src/dqn_scheduler.py`."
*   "Look at **Section 135-152** (The `get_action` function):
    ```python
    135:     def get_action(self, task: Task) -> Dict:
    ...
    140:         state = self._get_state_vector(task)
    143:         if random.random() < self.epsilon:
    144:             action = random.randrange(self.action_dim)
    ```
    *   **Line 140:** The agent looks at the task (its Size and Intensity). This is like looking at a math problem."
    *   **Line 143:** Sometimes it guesses randomly (Exploration). This helps it discover new tricks."
*   "If it doesn't guess, it uses its Brain (**Line 148**):
    ```python
    148:                 q_values = self.policy_net(state_tensor)
    ```
    *   **Simple Explanation:** The Neural Network predicts: 'If I give 50% of this task to the GPU, how good will the reward be?'"
*   **The Special Trick:** "We use **Fractional Actions** (**Line 152**). Instead of just 'GPU vs CPU', the agent can say 'Give 40% to GPU'. This is very precise!"

---

## 4. The Hybrid ML (The Teacher) (2 Minutes)
**Objective:** Explain the supervised model.
**File:** `src/offline_trainer.py`

*   "The RL agent is slow to start, so we have a 'Teacher' called Hybrid ML."
*   "Open `src/offline_trainer.py`."
*   "It uses a **Random Forest**, which is a classic Machine Learning model."
*   "Look at **Line 119**:
    ```python
    119:         results = self.model.fit(X, y, test_size=test_size)
    ```
    *   **Simple Explanation:** We run thousands of experiments first. We find the 'Perfect Answer' (Oracle). Then we show these answers to the Hybrid ML model. It memorizes the patterns."
*   "We actually used this Hybrid ML to **pre-train** the RL agent, so the Student starts smart!"

---

## 5. How We Calculate Efficiency (1 Minute)
**Objective:** Explain the score.
**File:** `src/simulation_engine.py`

*   "How do we know who is winning? Please go to `src/simulation_engine.py`, **Line 199**."
*   "Here is the `_calculate_metrics` function."
    ```python
    203:         time = result['actual_time']
    204:         gpu_frac = result['gpu_fraction']
    206:         # Power Model: GPU=50W, CPU=30W
    207:         power = (gpu_frac * 50.0) + ((1.0 - gpu_frac) * 30.0)
    208:         energy_joules = power * time
    ```
*   **Simple Explanation:**
    *   **Time:** How long the task took. Faster is better.
    *   **Energy:** We assume a GPU uses 50 Watts and CPU uses 30 Watts.
    *   **Calculation:** `Energy = Power * Time`.
    *   **Goal:** We want to minimize BOTH Time and Energy."

---

## 6. The Dashboard & Results (2 Minutes)
**Objective:** Show the live demo.
**Action:** Switch to the Browser / Dashboard UI.

*   "Let me show you the live system."
*   "On the screen, you see the **Live Task Stream**. Each moving bar is a task."
*   "**Colors:**
    *   **Blue bars:** CPU work.
    *   **Green bars:** GPU work."
*   "**The Charts:**
    *   You can see the **Average Task Duration** chart.
    *   **Hybrid ML (Green Line)** and **RL Agent (Purple Line)** are very low. This means they are FAST."
    *   **Round Robin (Orange Line)** is high. It is slow and dumb."
*   "**Conclusion:** My RL Agent (Student) learned to be almost as fast as the Hybrid ML (Teacher), but it is often more energy-efficient because it thinks about the long term!"

---

## Summary for Q&A
*   **Workload Generator:** create random math problems.
*   **Hybrid ML:** copies the best answers.
*   **RL Agent:** learns by doing.
*   **Efficiency:** Speed + Low Power.
