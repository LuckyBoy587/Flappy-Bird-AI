from collections import deque
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from flappy_bird_env import FlappyBirdEnv
from vector_env import VectorFlappyBirdEnv

# --- Hyperparameters ---
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration probability
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
episodes = 2000
batch_size = 128  # Increased for better GPU utilization
memory_size = 5000  # Increased for more diverse experiences

# --- Multi-Bird Training Parameters ---
num_birds = 16  # Number of birds to train simultaneously
steps_per_episode = 1000  # Max steps per episode for vectorized training (reduced for speed)
train_frequency = 4  # Train every N steps (instead of every step)
num_train_iterations = 1  # Number of training iterations per training step

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


def train_dqn_vectorized(num_envs: int = 16, render_first: bool = False):
    """
    Train DQN with multiple birds learning simultaneously.
    
    This function trains a single DQN network using experiences from multiple
    birds running in parallel. All birds share the same network and replay buffer,
    but explore independently, leading to faster and more diverse learning.
    
    Args:
        num_envs: Number of birds to train simultaneously (default: 16)
        render_first: If True, render the first bird during training (slower)
    
    Returns:
        Trained DQN model
    """
    global epsilon
    
    print(f"\n{'='*60}")
    print(f"MULTI-BIRD TRAINING MODE")
    print(f"Training with {num_envs} birds simultaneously")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if nn_model_path and os.path.exists(nn_model_path):
        print(f"Loading existing model from {nn_model_path}")
        nn_model = DQN(state_dim, action_dim).to(device)
        nn_model.load_state_dict(torch.load(nn_model_path, map_location=device))
        print("Continuing training from loaded model...")
    else:
        print("Creating new model...")
        nn_model = DQN(state_dim, action_dim).to(device)
    
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Shared replay memory for all birds
    memory = deque(maxlen=memory_size)
    
    # Create vectorized environment
    vec_env = VectorFlappyBirdEnv(num_envs=num_envs, render_mode=render_first, initial_flap=True)
    
    # Training statistics
    best_avg_reward = -float('inf')
    episode_rewards_history = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Each episode runs up to {steps_per_episode} steps\n")
    
    for ep in range(episodes):
        state = vec_env.reset()  # shape (num_envs, state_dim)
        # Track rewards for each bird's current life (reset when bird dies)
        current_episode_rewards = np.zeros(num_envs, dtype=float)
        # Track all completed episode rewards for averaging
        completed_episode_rewards = []
        episode_steps = 0
        min_episodes_per_round = num_envs * 2  # Each bird should die at least twice
        
        for step in range(steps_per_episode):
            # Epsilon-greedy batched action selection
            if random.random() < epsilon:
                actions = np.random.choice([0, 1], size=num_envs)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    q_values = nn_model(state_tensor)  # shape (num_envs, action_dim)
                    actions = q_values.argmax(dim=1).cpu().numpy()
            
            # Execute actions in all environments
            next_state, rewards, dones, infos = vec_env.step(actions)
            
            # Store transitions from all birds into shared memory
            for i in range(num_envs):
                memory.append((
                    state[i],
                    int(actions[i]),
                    float(rewards[i]),
                    next_state[i],
                    bool(dones[i])
                ))
                current_episode_rewards[i] += float(rewards[i])
                
                # When a bird dies, save its episode reward and reset counter
                if dones[i]:
                    completed_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
            
            state = next_state
            episode_steps += 1
            
            # Training step - only train every N steps for speed (instead of every step)
            if len(memory) >= batch_size and step % train_frequency == 0:
                for _ in range(num_train_iterations):
                    batch = random.sample(memory, batch_size)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                    
                    states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
                    actions_b = torch.tensor(actions_b, dtype=torch.long).unsqueeze(1).to(device)
                    rewards_b = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1).to(device)
                    next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32).to(device)
                    dones_b = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1).to(device)
                    
                    # Current Q-values for actions taken
                    q_values = nn_model(states_b).gather(1, actions_b)
                    
                    # Target Q-values (Bellman equation)
                    with torch.no_grad():
                        q_next = nn_model(next_states_b).max(1)[0].unsqueeze(1)
                        q_target = rewards_b + gamma * q_next * (1 - dones_b)
                    
                    # Loss & backprop
                    loss = loss_fn(q_values, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Optional: render if enabled
            if render_first:
                vec_env.render()
            
            # Early stopping: if we have enough completed episodes, stop this round
            if len(completed_episode_rewards) >= min_episodes_per_round:
                break
        
        # Add any remaining episode rewards that didn't finish
        for i in range(num_envs):
            if current_episode_rewards[i] > 0:
                completed_episode_rewards.append(current_episode_rewards[i])
        
        # Decay exploration rate
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Calculate statistics from completed episodes
        if len(completed_episode_rewards) > 0:
            avg_reward = np.mean(completed_episode_rewards)
            max_reward = np.max(completed_episode_rewards)
            min_reward = np.min(completed_episode_rewards)
            num_episodes_completed = len(completed_episode_rewards)
        else:
            # Fallback if no episodes completed (very rare, early in training)
            avg_reward = np.mean(current_episode_rewards)
            max_reward = np.max(current_episode_rewards)
            min_reward = np.min(current_episode_rewards)
            num_episodes_completed = 0
            
        episode_rewards_history.append(avg_reward)
        
        # Track best performance
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(nn_model.state_dict(), "dqn_flappy_bird_best.pth")
        
        
        # Save checkpoint every 50 episodes
        if (ep + 1) % 50 == 0:
            # Print progress
            print(f"Episode {ep + 1:4d}/{episodes} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Max: {max_reward:7.2f} | "
                f"Min: {min_reward:7.2f} | "
                f"Completed: {num_episodes_completed:3d} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Memory: {len(memory):5d}")
            checkpoint_path = f"dqn_flappy_bird_ep{ep+1}.pth"
            torch.save(nn_model.state_dict(), checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Print rolling average
            if len(episode_rewards_history) >= 50:
                recent_avg = np.mean(episode_rewards_history[-50:])
                print(f"  → Last 50 episodes avg: {recent_avg:.2f}")
    
    # Save final model
    vec_env.close()
    torch.save(nn_model.state_dict(), nn_model_path)
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved to {nn_model_path}")
    print(f"Best model saved to dqn_flappy_bird_best.pth")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"{'='*60}\n")
    
    return nn_model