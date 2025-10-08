# Quick Start Guide - Flappy Bird AI Environment

## 🎮 What You Have

A complete **Flappy Bird environment** built with Pygame that's ready for reinforcement learning! The environment follows the OpenAI Gym interface pattern, making it easy to plug into RL algorithms.

## 📁 Files Created

1. **`flappy_bird_env.py`** - Main environment class
   - `FlappyBirdEnv` class with `reset()`, `step()`, `render()` methods
   - Complete game logic with physics, collisions, scoring
   - State observation and reward system

2. **`example.py`** - Simple demonstration
   - Rule-based AI agent example
   - Shows how to use the environment

3. **`test_env.py`** - Test suite
   - Manual play mode
   - Random agent test
   - Interface testing (no rendering)

4. **`config.py`** - Configuration settings
   - Easily adjust game parameters
   - Difficulty presets (easy, normal, hard, extreme)

5. **`requirements.txt`** - Dependencies
6. **`README.md`** - Full documentation

## 🚀 How to Run

### 1. Play Manually (Test the Game)
```bash
python flappy_bird_env.py
```
- Press SPACE or UP ARROW to flap
- Press R to reset when game over
- Press ESC to quit

### 2. Watch a Simple AI
```bash
python example.py
```
- Runs a rule-based agent that flaps when below the pipe gap center
- Good for testing the environment

### 3. Run Test Suite
```bash
python test_env.py
```
Choose from:
- Random agent (watch random play)
- Manual play
- Interface test (no rendering)

## 🤖 Using in Your RL Code

### Basic Usage

```python
from flappy_bird_env import FlappyBirdEnv

# Create environment
env = FlappyBirdEnv(render_mode=True)

# Reset to get initial state
state = env.reset()  # Returns: [bird_y, velocity, pipe_x, pipe_top, pipe_bottom]

# Game loop
done = False
while not done:
    # Your AI decides: 0 = do nothing, 1 = flap
    action = your_model.predict(state)
    
    # Take action
    state, reward, done, info = env.step(action)
    
    # Render (optional)
    env.render()

# Clean up
env.close()
```

### State Space (5 values, normalized 0-1)
- `bird_y`: Vertical position
- `bird_velocity`: Vertical velocity
- `next_pipe_x`: Distance to next pipe
- `next_pipe_top`: Top of pipe gap
- `next_pipe_bottom`: Bottom of pipe gap

### Action Space
- `0`: Do nothing (gravity)
- `1`: Flap (jump up)

### Rewards
- `+1.0`: Pass through a pipe
- `+0.1`: Stay alive (each frame)
- `-100`: Collision (game over)

## 🎯 Next Steps for AI Training

### 1. Deep Q-Network (DQN)
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 actions
        )
    
    def forward(self, x):
        return self.network(x)
```

### 2. Training Loop Structure
```python
from flappy_bird_env import FlappyBirdEnv
import torch

env = FlappyBirdEnv(render_mode=False)  # Faster without rendering
model = DQN()
optimizer = torch.optim.Adam(model.parameters())

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state))
                action = q_values.argmax().item()
        
        next_state, reward, done, info = env.step(action)
        
        # Store experience and train
        # ... your training logic here
        
        state = next_state
```

## ⚙️ Adjusting Difficulty

Edit `config.py` or modify parameters in `flappy_bird_env.py`:

```python
# Make it easier
env.GRAVITY = 0.4
env.PIPE_GAP = 150
env.PIPE_VELOCITY = 2

# Make it harder
env.GRAVITY = 0.7
env.PIPE_GAP = 90
env.PIPE_VELOCITY = 5
```

## 🐛 Troubleshooting

**Sprites not loading?**
- Make sure `sprites/` folder is in the same directory

**Game too fast/slow?**
- Adjust `FPS` in the environment (default: 60)

**Want to train without rendering?**
- Set `render_mode=False` when creating environment
- Much faster for training!

## 📊 Measuring Performance

The environment returns useful info:
```python
state, reward, done, info = env.step(action)
print(info['score'])  # Number of pipes passed
print(info['frame_count'])  # Frames survived
```

## 🎨 Customization Ideas

1. **Different bird colors**: Change sprites to use blue or red bird
2. **Night mode**: Use `background-night.png`
3. **Different rewards**: Adjust reward values for your training
4. **State space**: Add more features (e.g., next 2 pipes, distance to ground)

## 📚 Resources for RL

Popular algorithms to try:
- **DQN** (Deep Q-Network) - Good starting point
- **PPO** (Proximal Policy Optimization) - Very stable
- **A2C/A3C** (Advantage Actor-Critic) - Good for this task
- **NEAT** (NeuroEvolution) - No backprop needed!

Libraries to help:
- Stable-Baselines3 (easiest)
- PyTorch / TensorFlow (more control)
- RLlib (scalable)

## 🎓 Learning Path

1. ✅ **You are here!** - Environment is ready
2. **Study RL basics** - Q-learning, policy gradients
3. **Implement simple agent** - Start with DQN
4. **Train and tune** - Hyperparameters matter!
5. **Experiment** - Try different algorithms and architectures

## 💡 Tips

- Start training **without rendering** (much faster)
- Use **render_mode=True** only to watch trained agents
- Save model checkpoints regularly
- Track scores over episodes to see improvement
- Start with **easy difficulty** for faster initial learning

---

**Ready to train your AI?** Start with `example.py` to understand the interface, then build your RL agent! 🚀
