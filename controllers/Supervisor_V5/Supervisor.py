import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from controller import Robot , Supervisor , DistanceSensor ,Compass
import math , time 

#V5
# Define the hyperparameters
replay_buffer_capacity = 2000
learning_rate = 0.01
input_size = 1
output_size = 4
episode = 0
epsilon = 1
epsilon_decay = 0.9995
min_epsilon = 0.01
batch_size = 50
gamma = 0.99  # Discount factor
target = [14]
step_limit_for_termination = 40
obstacle_value = 700

target_reward = 100
empty_space_penalty = -1
collision_penalty = -100

            
trained = False
# Define the QNetwork
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.mem_size = capacity
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.mem_cntr = 0

    def add_experience(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_cntr, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# Define DQNAgent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.update_target_interval = 50  # Adjust this value as needed
        self.target_update_counter = 0
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity,input_size)
        
        
    def update_target_network(self):
        if self.target_update_counter % self.update_target_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print("target netwrok updated")
        self.target_update_counter += 1 
        
        
    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(output_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def train(self, batch_size, gamma,epsilon):
   
        if self.replay_buffer.mem_cntr < batch_size:
            return

        batch = self.replay_buffer.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * (1 - dones) * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
    
def calculate_reward(ps_values , new_state):
    
    obstacle_detected = obstacle_value < ps_values[7]
    
    cell = new_state[0]
    
    if cell in target:
        target_reached = True
    else:
        target_reached = False
    
    if obstacle_detected:
        return collision_penalty
    elif target_reached:
        if cell == target[0]:
            print("****************TARGET REACHED****************")
        return target_reward
    else:
        return empty_space_penalty

def check_termination_condition(message, total_steps):
    obstacle_detected = obstacle_value < message[7]
    #print(message[7],obstacle_detected)
    
    if obstacle_detected:  
        return True
    elif total_steps > step_limit_for_termination:
        print("Episode terminated : step limit")
        return True   
    else:
        return False



last_10_scores = [0] * 10
average_score_limit = 200
score = 0
save_path = "q_network_model.pth" 
# Initialize DQNAgent
agent = DQNAgent(input_size, output_size)

# Training loop
robot = Supervisor()
e_puck_1_node = robot.getFromDef("E-puck_1")
e_puck_2_node = robot.getFromDef("E-puck_2")
time_step = 32

supervisor_receiver_1 = robot.getDevice('receiver_1')
supervisor_receiver_1.enable(time_step)
supervisor_receiver_2 = robot.getDevice('receiver_2')
supervisor_receiver_2.enable(time_step)

supervisor_emitter = robot.getDevice('emitter')

while not trained: 

    robot.step(33) 
    if supervisor_receiver_1.getQueueLength()>0:
        recevied_message_1 = supervisor_receiver_1.getFloats()
        supervisor_receiver_1.nextPacket() 
        state_1 = [recevied_message_1[-1]]
        sensor_1 = list(recevied_message_1[:8])
    else:
        robot.step(32)      
        continue
        
    robot.step(33)
    if supervisor_receiver_2.getQueueLength()>0:
        recevied_message_2 = supervisor_receiver_2.getFloats()
        supervisor_receiver_2.nextPacket()
        state_2 = [recevied_message_2[-1]]
        sensor_2 = list(recevied_message_2[:8])
    else:
        robot.step(32)      
        continue
    
    episode += 1
    print("-----------------------------------------------------------------------------------------------------------------------")
    print(f"Episode No: {episode}  || Previous episode score: {score} || Epsilon : {epsilon}" )
    done = False
    idle_1 = False
    idle_2 = False
    total_steps = 0
    score = 0

    while not done:
        # Select action using epsilon-greedy policy
        action_1 = agent.select_action(state_1, epsilon)
        action_2 = agent.select_action(state_2, epsilon)
        #send action to robot_1
        robot.step(33)
        if supervisor_receiver_1.getQueueLength()>0:
            supervisor_emitter.setChannel(1)
            if not idle_1:
                #print("action_1", action_1)
                supervisor_emitter.send([action_1])
            message_1 = supervisor_receiver_1.getFloats()
            supervisor_receiver_1.nextPacket()  
        
        #send action to robot_2
        robot.step(33)
        if supervisor_receiver_2.getQueueLength()>0:
            supervisor_emitter.setChannel(5)
            if not idle_2:
                #print("action_2", action_2)
                supervisor_emitter.send([action_2])
            message_2 = supervisor_receiver_2.getFloats()
            supervisor_receiver_2.nextPacket()
           
        robot.step(40000)
        message_1 = supervisor_receiver_1.getFloats()
        supervisor_receiver_1.nextPacket()
        message_2 = supervisor_receiver_2.getFloats()
        supervisor_receiver_2.nextPacket()
        robot.step(10000)
        message_1 = supervisor_receiver_1.getFloats()
        supervisor_receiver_1.nextPacket()
        message_2 = supervisor_receiver_2.getFloats()
        supervisor_receiver_2.nextPacket()
  
        if not idle_1: 
            new_state_1 = [message_1[-1]]
            #print("new_state_1" , new_state_1)
            sensor_1 = list(message_1[:8])
            reward_1 = calculate_reward(message_1,new_state_1)
            done_1 = check_termination_condition(sensor_1, total_steps)
            agent.replay_buffer.add_experience(state_1, action_1, reward_1, new_state_1, done_1)
            
            score+=reward_1
            
            if new_state_1 == target:
                translationField_1 = robot.getFromDef("E-puck_1").getField('translation')
                translationField_1.setSFVec3f([0.447971,0.453389, 0])
            if done_1:
                idle_1 = True
        
        if not idle_2: 
            new_state_2 = [message_2[-1]]
            #print("new_state_2" , new_state_2)
            reward_2 = calculate_reward(message_2,new_state_2)
            sensor_2 = list(message_2[:8])
            done_2 = check_termination_condition(sensor_2, total_steps)
            agent.replay_buffer.add_experience(state_2, action_2, reward_2, new_state_2, done_2)
            score+=reward_2
            if new_state_2 == target:
                    translationField_2 = robot.getFromDef("E-puck_2").getField('translation')
                    translationField_2.setSFVec3f([0.431144,-0.442926, 0])
            if done_2:
                    idle_2 = True
            
        if done_1 and done_2:
            done =  True
            print("Episode terminated : obstacle_detected")        
        
        # Train the agent
        agent.train(batch_size, gamma,epsilon)
            
        # Update state for the next iteration
        state_1 = new_state_1
        state_2 = new_state_2
        agent.update_target_network()
        
        robot.step(time_step)
        total_steps += 1 
    
    if agent.replay_buffer.mem_cntr > batch_size :
        epsilon = max(min_epsilon, epsilon * epsilon_decay)  
    
    robot.simulationReset() 
    e_puck_1_node.restartController() 
    e_puck_2_node.restartController()
    
    if episode%10 == 0:
        torch.save(agent.q_network.state_dict(), save_path)
       

if trained:
    agent.q_network.load_state_dict(torch.load(save_path))
    agent.q_network.eval()
    
    while trained: 
        robot.step(33) 
        if supervisor_receiver_1.getQueueLength()>0:
            recevied_message_1 = supervisor_receiver_1.getFloats()
            supervisor_receiver_1.nextPacket() 
            state_1 = [recevied_message_1[-1]]
            sensor_1 = list(recevied_message_1[:8])
        else:
            robot.step(32)      
            continue
            
        robot.step(33)
        if supervisor_receiver_2.getQueueLength()>0:
            recevied_message_2 = supervisor_receiver_2.getFloats()
            supervisor_receiver_2.nextPacket()
            state_2 = [recevied_message_2[-1]]
            sensor_2 = list(recevied_message_2[:8])
        else:
            robot.step(32)      
            continue
 
        with torch.no_grad():
            state_tensor = torch.tensor(state_1, dtype=torch.float32)
            q_values = agent.q_network(state_tensor)
            action_1 = torch.argmax(q_values).item()
        
        with torch.no_grad():
            state_tensor = torch.tensor(state_2, dtype=torch.float32)
            q_values = agent.q_network(state_tensor)
            action_2 = torch.argmax(q_values).item()
            
        if supervisor_receiver_1.getQueueLength()>0:
            supervisor_emitter.setChannel(1)
            supervisor_emitter.send([action_1])

        robot.step(33)
        if supervisor_receiver_2.getQueueLength()>0:
            supervisor_emitter.setChannel(5)
            supervisor_emitter.send([action_2])
