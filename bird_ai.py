from collections import deque
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from flappy_bird_env import FlappyBirdEnv

# --- Hyperparameters ---
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration probability
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
episodes = 2000
batch_size = 64
memory_size = 2000

# --- Environment ---
env = FlappyBirdEnv(False)
state_dim = env.STATE_COUNT  # 5
action_dim = env.ACTION_COUNT  # 2
nn_model_path = "dqn_flappy_bird.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super(DQN, self).__init__()
        # Hidden layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        # Output layer  
        self.out = nn.Linear(64, action_dim)
        
        # Optional: weight initialization for stability
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)   # Q-values for each action


def train_dqn():
    global epsilon
    if nn_model_path and os.path.exists(nn_model_path):
        print(f"Loading model from {nn_model_path}")
        nn_model = DQN(state_dim, action_dim).to(device)
        nn_model.load_state_dict(torch.load(nn_model_path, map_location=device))
        return nn_model
    nn_model = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- Replay memory ---
    memory = deque(maxlen=memory_size)


    # --- ε-greedy action selection ---
    def choose_action(state) -> int:
        global epsilon
        if random.random() < epsilon:
            return random.choice([0, 1])
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = nn_model(state_tensor)
                return int(torch.argmax(q_values).item())
            
            
    # --- Training loop ---
    print(f"Starting training for {episodes} episodes...")
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train NN if enough samples
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Current Q-values for actions taken
                q_values = nn_model(states).gather(1, actions)

                # Target Q-values
                with torch.no_grad():
                    q_next = nn_model(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards + gamma * q_next * (1 - dones)

                # Loss & backprop
                loss = loss_fn(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay exploration
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {ep + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")


    torch.save(nn_model.state_dict(), nn_model_path)
    print(f"Model saved to {nn_model_path}")
    return nn_model