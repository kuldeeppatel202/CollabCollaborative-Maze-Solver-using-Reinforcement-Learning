## Collaborative-Maze-Solver-using-Reinforcement-Learning

### Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Reinforcement Learning](#reinforcement-learning)
  - [Key Concepts](#key-concepts)
  - [Neural Networks in RL](#neural-networks-in-rl)
  - [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Training Process](#training-process)
### Problem Statement

This project aims to design a maze-solving robot system utilizing two Webots-based e-Puck robots. The system employs Deep Q-Learning algorithms to navigate from a designated starting point to a target destination within a maze environment. The robots are equipped with infrared distance sensors and cameras to gather real-time data, enabling them to collaboratively explore and navigate through the maze.

### Features

- Two e-Puck robots working collaboratively
- Deep Q-Learning algorithm for decision making
- Real-time data gathering using infrared sensors and cameras
- Efficient maze navigation and exploration

### Getting Started

To run this project on your local machine, follow these steps:

1. **Install Webots**:
   - Download and install Webots from the [official Webots website](https://cyberbotics.com/).
   - Follow the installation instructions for your operating system.

2. **Clone the Repository**:
   ```
   git clone https://github.com/MobileRobotics-project/Collaborative-Maze-Solver-using-Reinforcement-Learning.git
   ```

3. **Create a New Webots Project**:
   - Open Webots
   - Go to File > New > New Project Directory
   - Choose a name and location for your project

4. **Add Project Files**:
   - Copy all files from `worlds` folder in this repository to your Webots project's `worlds` directory.
   - Copy all controller files from the `controllers` directory to your Webots project's `controllers` directory.

5. **Open the World File**:
   - In Webots, go to File > Open World
   - Navigate to your project's `worlds` directory and select `maze_world.wbt`

6. **Start the Simulation**:
   - In the Webots window, click the "Play" button to start the simulation
   - You should now see the robots navigating through the maze using the untrained DQN model

### Usage

Once you have the simulation running:

- The robots will start navigating through the maze automatically.
- You can observe their behavior and how they learn to solve the maze over time.
- The terminal running the supervisor script will display information about each episode, including rewards and exploration rate (epsilon).

To stop the simulation:
- Click the "Pause" button in Webots
- Press `Ctrl+C` in the terminal running the supervisor script

  <div align="center">
  <a href="https://www.youtube.com/watch?v=YtQ5b0F2KA4"><img src="https://img.youtube.com/vi/YtQ5b0F2KA4/0.jpg" alt="IMAGE ALT TEXT"></a>
   </div>

### Reinforcement Learning

This project utilizes reinforcement learning, specifically Deep Q-Learning, to train the robots to navigate through the maze efficiently.

#### Key Concepts

1. **Agent**: In our case, the e-Puck robots that learn to navigate the maze.
2. **Environment**: The maze in which the robots operate.
3. **State**: The current position of the robot in the maze, represented by sensor readings.
4. **Action**: The movement decisions made by the robot (e.g., move forward, turn left, turn right).
5. **Reward**: A numerical value given to the agent based on its actions. In our implementation:
   - Reaching the target: High positive reward
   - Collision: High negative reward
   - Empty space: Small negative reward (to encourage efficient pathfinding)
6. **Policy**: The strategy that the agent follows to determine the next action based on the current state.

#### Neural Networks in RL

In this project, we use a neural network to approximate the Q-function, which predicts the expected cumulative reward for each action in a given state. Our network architecture consists of:

- Input layer: Corresponds to the state representation
- Hidden layers: Two fully connected layers with ReLU activation
- Output layer: Corresponds to the Q-values for each possible action

#### Deep Q-Network (DQN)

DQN is an advanced RL algorithm that combines Q-learning with deep neural networks. Key components of our DQN implementation include:

1. **Q-Network**: A neural network that approximates the Q-function, mapping states to action values.
2. **Target Network**: A separate network used to compute target Q-values, updated periodically to stabilize training.
3. **Experience Replay**: A buffer that stores and randomly samples past experiences (state, action, reward, next state) to break correlations between consecutive samples and improve learning stability.
4. **Epsilon-Greedy Exploration**: A strategy that balances exploration and exploitation by sometimes taking random actions (controlled by an epsilon value that decays over time).

#### Training Process


The training process for our maze-solving robots involves the following steps:
![flowchart](https://github.com/user-attachments/assets/1e50cf0a-944f-416a-aa90-9cacfdaa5475)


1. **Initialization**: The robots start with random policies, and the replay buffer is empty.
2. **Exploration and Data Collection**: 
   - The robots explore the maze using an epsilon-greedy strategy.
   - Experiences (state, action, reward, next state) are collected and stored in the replay buffer.
3. **Training**:
   - Batches of experiences are randomly sampled from the replay buffer.
   - The Q-network is updated using these experiences:
     - Compute the current Q-values
     - Compute the target Q-values using the target network
     - Calculate the loss (mean squared error between current and target Q-values)
     - Update the Q-network weights using backpropagation
4. **Target Network Update**: Periodically, the target network weights are updated to match the Q-network.
5. **Epsilon Decay**: The exploration rate (epsilon) is gradually reduced to shift from exploration to exploitation.
6. **Iteration**: Steps 2-5 are repeated for many episodes to refine the policy.

This implementation uses PyTorch for building and training the neural network, allowing for efficient GPU acceleration if available.
