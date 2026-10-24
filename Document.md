# Project Proposal: Hierarchical Language-Conditioned Reinforcement Learning for Dynamic Obstacle Avoidance

**Prepared by:** Kevin D. Flynn  
**Program:** M.Tech CSE (Artificial Intelligence and Machine Learning)

---

## 1. Abstract
Autonomous agents operating in dynamic environments must seamlessly integrate high-level natural language instructions with low-level spatial navigation and reactive obstacle avoidance. Existing approaches often rely on static pathfinding algorithms that fail when confronted with rapidly moving obstacles, or utilize monolithic neural networks that struggle to generalize semantic goals. 

This project proposes a hierarchical reinforcement learning framework that decouples language-conditioned goal planning from reactive motor control. By framing the navigation task as a **Constrained Markov Decision Process (CMDP)**, the agent is trained using Deep Reinforcement Learning to maximize goal attainment probabilities while strictly bounding collision risks. Experimental results in simulated multi-agent continuous-space environments will evaluate whether this decoupled architecture achieves superior trajectory optimization, reduces inference latency, and exhibits robust zero-shot adaptation to novel dynamic threats.

## 2. Introduction
The deployment of autonomous agents in chaotic, real-world spaces requires bridging the gap between semantic human intent and continuous motor control. While traditional game AI and classical robotics rely heavily on hardcoded behavior trees and NavMesh algorithms, these methods degrade rapidly in unpredictably changing or adversarial environments. The next frontier in Embodied AI—**Dynamic Vision-and-Language Navigation (DynamicVLN)**—demands agents capable of understanding instructions and executing them while autonomously preserving their safety.

## 3. Problem Statement
There is a fundamental conflict between executing high-level semantic instructions (e.g., *"Retrieve the objective from the target zone"*) and surviving immediate, unpredictable, or adversarial threats. The problem lies in optimizing multi-objective decision-making: how can an agent maximize the probability of achieving a human-dictated goal while mathematically bounding the risk of collision or failure in a space where obstacles act independently or adversarially?

## 4. Proposed Architecture and Technology
The project will utilize a two-tier hybrid architecture:

* **High-Level Intent Parser:** A lightweight, quantized Large Language Model (LLM) or sentence-transformer pipeline that converts speech/text input into a continuous goal embedding.
* **Low-Level Control Policy:** A Deep Reinforcement Learning (DRL) algorithm (such as PPO or SAC) operating under a Constrained Markov Decision Process (CMDP). This policy takes the goal embedding and real-time spatial data to output continuous movement vectors.
* **Simulation Engine:** Unity3D combined with the ML-Agents toolkit to provide a high-fidelity physics environment for training and adversarial scenario generation.

## 5. Real-World Applications
While trained in simulation, the framework directly maps to critical autonomous industries:

* **Search and Rescue Robotics:** Navigating a partially collapsed building based on verbal commands while dodging falling debris and structural shifts.
* **Industrial Logistics:** Warehouse robots retrieving shipments while avoiding human workers, forklifts, and other automated systems on a chaotic floor.
* **Autonomous Vehicles:** Executing user commands (*"Pull over at the next pharmacy"*) while reacting to dynamic road hazards and erratic drivers.

## 6. Resources and Tools
The development and training of this model will require the following stack:

* **Machine Learning Frameworks:** PyTorch or TensorFlow for policy network training.
* **Simulation:** Unity Game Engine + Unity ML-Agents Toolkit.
* **NLP Stack:** HuggingFace Transformers (for local quantized model inference), OpenAI Whisper (for optional speech-to-text pipeline).
* **Hardware:** CUDA-enabled GPU for accelerated DRL training iterations and rendering.

---

## Simulation & Middleware Strategy
Testing and developing algorithms for Dynamic Obstacle Avoidance—where an autonomous system must continuously sense, predict, and maneuver around moving entities (like pedestrians, other vehicles, or drones)—requires specific simulation tools that accurately model physics, real-time sensor streams (Lidar, depth cameras), and dynamic actor trajectories.

### 1. The Core Middleware: ROS 2 & Nav2 (Essential Requirement)
Before picking a graphical simulator, you must learn the Robot Operating System 2 (ROS 2).
* **Why it’s needed:** ROS 2 provides the infrastructure that handles communication between your sensors, collision avoidance algorithms, and vehicle actuators.
* **Key Framework:** Nav2 (Navigation 2). It features the environmental modeling frameworks (like costmaps and voxel layers) required to clear out old sensor data and track dynamic trajectories using algorithms like Time-Elastic Band (TEB) or Dynamic Window Approach (DWA).

### 2. High-Fidelity 3D Simulation Software
Depending on your hardware target and development approach, you should learn one of these simulators to connect with your ROS 2 stack (e.g., Gazebo or NVIDIA Isaac Sim).

### 3. Numerical & Mathematical Prototyping
* **MATLAB & Simulink:** Before deploying complex C++ or Python code in a full 3D environment, developers often prototype theoretical avoidance math here. You can use the Robotics System Toolbox and UAV Toolbox to simulate optimal control rules (like Velocity Obstacles, Model Predictive Control, or Dijkstra variants) against geometric dynamic actors.

### Recommended Learning Roadmap
1. Start by learning ROS 2 basics (Nodes, Topics, and Actions) alongside C++ or Python.
2. Learn Nav2, specifically focusing on how the Costmap 2D architecture inserts and updates obstacles dynamically.
3. Pick Gazebo for general terrestrial/drone robotics or Isaac Lab if your obstacle avoidance solution relies heavily on deep reinforcement learning.

---

## Literature Review & Keywords
Use the "Advanced Search" function to stack Boolean operators. Enter command strings such as: `("Reinforcement Learning" OR "Deep Reinforcement Learning") AND ("Dynamic Obstacle Avoidance" OR "Collision Avoidance") AND ("Navigation" OR "Continuous Control”)`.

* **Vision-and-Language Navigation (VLN):** The formal academic term for agents following natural language instructions.
* **Safe Reinforcement Learning:** The specific sub-field that deals with Constrained Markov Decision Processes (CMDPs) and mathematical collision bounding.
* **Hierarchical Reinforcement Learning (HRL):** Used when separating high-level goals (language parsing) from low-level control (motor policies).
* **Embodied AI:** The broader category covering agents interacting within 3D physics environments.
* **Continuous Control:** Specifies that the agent uses vector-based acceleration rather than grid-based discrete steps.
* **Dynamic Obstacle Avoidance:** Standard terminology for evading moving threats rather than static walls.

---

## Implementation Details

### 1. How a Virtual AI Character "Sees" and Uses the Algorithm
In a game engine like Unity, an AI character does not see pixels as a human player does. Instead, it perceives the environment through simulated mathematical sensors:

* **Spatial Perception (Simulated LiDAR / Raycasting):** In real life, a drone uses LiDAR or stereo cameras. In Unity, the AI character casts invisible 3D rays (using `ML-Agents RayPerceptionSensor3D`) in a circular pattern around its body. Each ray returns continuous numerical values: distance to the nearest collider and the semantic tag of whatever it hits (e.g., Wall, Player, MovingThreat, PowerUp).
* **Kinematic State:** The engine reads the character's internal physics state: current velocity vector `(v_x, v_y, v_z)`, orientation, and angular momentum.
* **Semantic Goal Parsing:** The High-Level Intent Parser takes a command (e.g., *"Intercept the opponent from the flank"*) and converts it into a continuous goal embedding vector `g`.

**The Policy Inference Loop:**
1. At every physical frame (typically 50 Hz in Unity's `FixedUpdate`), the raycast distances, kinematic variables, and goal vector `g` are packed into a 1D tensor: the state vector `S_t`.
2. `S_t` passes through your PyTorch neural network (the policy `\pi_\theta`).
3. The network outputs a continuous 2D/3D action vector `a_t \in [-1.0, 1.0]`.
4. Unity's physics engine applies that vector directly as forces or torque to the character's Rigidbody:

---

##Objective

###1. The "Elevator Pitch" (For friends, recruiters, or non-technical people)

"I am building a robotic brain that separates complex human instructions from survival reflexes. Most robots crash if a moving object blocks their pre-programmed path. 'Dodger' uses one AI to understand human voice commands and map the destination, while a second, independent reinforcement learning AI acts as a reactive nervous system—instinctively dodging unpredictable moving threats in real-time to reach the target safely."

###2. The Embodied AI Objective (For AI peers or tech interviews)

"The objective is to solve the multi-objective conflict in Dynamic Vision-and-Language Navigation (DynamicVLN). 'Dodger' creates a hierarchical agent that decouples long-horizon semantic goal execution from short-horizon continuous motor control. It ensures that an agent can follow natural language instructions without relying on static NavMesh algorithms, using simulated LiDAR to reactively evade dynamic obstacles in continuous space."

###3. The Formal Academic Objective (For M.Tech thesis committee)

"To develop a two-tier hierarchical reinforcement learning architecture that translates natural language commands into continuous goal embeddings via a High-Level Intent Parser, while delegating spatial navigation to a Low-Level Motor Control Policy. By framing the navigation task as a Constrained Markov Decision Process (CMDP), the objective is to maximize semantic goal attainment while mathematically bounding the collision risk against dynamic adversaries."

---
##Objective points for paper

###1. Formulation of the Experiential Data and State Spaces
To construct and define the interactive 3D simulation environment (utilizing Unity ML-Agents) that will generate continuous experiential data. This includes defining the state space ($S$) via simulated LiDAR raycasts and kinematic sensors, and establishing the continuous action space ($A$) required for fluid motor control.

###2. Integration of the High-Level Intent Parser
To develop and integrate a lightweight language-processing layer (using quantized Sentence-Transformers or a local LLM) capable of translating unstructured natural language instructions into continuous, mathematical spatial goal embeddings ($g$) without relying on hardcoded keyword scripts.

###3. Optimization of the Constrained Control Policy
To train a Deep Reinforcement Learning algorithm (such as Proximal Policy Optimization) mathematically framed as a Constrained Markov Decision Process (CMDP). The objective is to optimize the policy to maximize semantic goal attainment (Reward $R$) while strictly bounding collision risks against dynamic threats below a predefined safety threshold (Cost $C \le \epsilon$).

###4. Algorithmic Benchmarking and Ablation Analysis
To evaluate the hierarchical architecture's performance against classical static pathfinding algorithms (e.g., NavMesh) and standard, unconstrained RL baselines. This evaluation will strictly measure inference latency, path efficiency, and the success rate of zero-shot evasion in highly adversarial, dynamic scenarios.

###5. Conclusion on Sim-to-Real Transferability
To synthesize the simulation benchmarks and conclude on the algorithmic viability of decoupling semantic intent from reactive motor control. The final objective is to validate how this continuous control architecture can be directly mapped to critical real-world Embodied AI domains, such as autonomous warehouse logistics and search-and-rescue robotics.

---
```csharp

// Unity C# translating neural action vector into physical movement
rigidbody.AddForce(new Vector3(action[0], 0f, action[1]) * moveSpeed);//



