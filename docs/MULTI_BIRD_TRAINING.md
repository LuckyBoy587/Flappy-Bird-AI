# Multi-Bird Training Guide 🐦🐦🐦

This guide explains the new **population-based training** feature that allows multiple Flappy Birds to learn simultaneously using a shared DQN network.

## 🎯 What is Multi-Bird Training?

Instead of training a single bird for 2000 episodes, you can now train multiple birds (e.g., 16) at the same time. All birds:
- **Share the same DQN network** (one brain for all)
- **Contribute to the same replay buffer** (shared memory)
- **Explore independently** (different experiences)
- **Learn from each other's successes and failures**

## 🚀 Benefits

### 1. **Faster Learning**
- 16 birds collect 16x more experience per episode
- Replay buffer fills up faster with diverse experiences
- Network sees more varied situations quickly

### 2. **Better Exploration**
- Each bird takes different actions (epsilon-greedy per bird)
- More diverse state coverage
- Reduces overfitting to specific scenarios

### 3. **More Stable Training**
- Averaged performance across population
- Less sensitive to individual lucky/unlucky runs
- Smoother learning curves

### 4. **Efficient Use of GPU**
- Batch processing of multiple birds
- Better GPU utilization with batched forward passes
- Modern GPUs handle parallel computation efficiently

## 📁 New Files

### `vector_env.py`
Vectorized environment wrapper that manages multiple Flappy Bird instances:
```python
from vector_env import VectorFlappyBirdEnv

# Create 16 parallel environments
vec_env = VectorFlappyBirdEnv(num_envs=16, render_mode=False)

# Reset all birds
states = vec_env.reset()  # shape: (16, 5)

# Take actions for all birds
next_states, rewards, dones, infos = vec_env.step(actions)
```

### `main.py`
Interactive script to choose training mode:
```bash
python main.py
```

## 🎮 How to Use

### Option 1: Using main.py (Recommended)

```bash
python main.py
```

You'll see a menu:
```
Select training mode:
  1. Single Bird Training (Original - 2000 episodes)
  2. Multi-Bird Training (Population-based - 16 birds)
  3. Test Trained Model
  4. Exit
```

Choose **2** for multi-bird training, then:
- Enter number of birds (default: 16)
- Choose whether to render (recommend: no for faster training)

### Option 2: Direct Function Call

```python
from bird_ai import train_dqn_vectorized

# Train with 16 birds
model = train_dqn_vectorized(num_envs=16, render_first=False)

# Train with 32 birds
model = train_dqn_vectorized(num_envs=32, render_first=False)
```

### Option 3: Modify and Run bird_ai.py

Add at the bottom of `bird_ai.py`:
```python
if __name__ == "__main__":
    # Multi-bird training
    trained_model = train_dqn_vectorized(num_envs=16, render_first=False)
```

Then run:
```bash
python bird_ai.py
```

## ⚙️ Configuration

### Number of Birds
Adjust based on your hardware:
- **8 birds**: Good for laptops/slower machines
- **16 birds**: Recommended for most desktops
- **32 birds**: High-end machines with good GPU
- **64+ birds**: Powerful GPUs (may hit memory limits)

### Training Parameters

In `bird_ai.py`, adjust these settings:

```python
# Multi-Bird Training Parameters
num_birds = 16  # Number of simultaneous birds
steps_per_episode = 5000  # Max steps per episode

# Standard DQN parameters (same as before)
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
episodes = 2000
batch_size = 64
memory_size = 2000
```

### Rendering
- **Training**: Set `render_first=False` for speed (recommended)
- **Watching**: Set `render_first=True` to see the first bird play
- **Testing**: Always render when testing with `test_trained_model()`

## 📊 Training Output

You'll see detailed progress:

```
Episode    1/2000 | Avg Reward:  -95.23 | Max:   12.50 | Min: -100.00 | Epsilon: 0.995 | Steps: 1250 | Memory:  1024
Episode    2/2000 | Avg Reward:  -78.45 | Max:   45.20 | Min: -100.00 | Epsilon: 0.990 | Steps: 1876 | Memory:  2000
...
Episode   50/2000 | Avg Reward:   23.45 | Max:  156.30 | Min:  -34.20 | Epsilon: 0.778 | Steps: 4523 | Memory:  2000
  → Checkpoint saved: dqn_flappy_bird_ep50.pth
  → Last 50 episodes avg: 12.34
```

### Metrics Explained
- **Avg Reward**: Average across all birds (main metric)
- **Max**: Best performing bird this episode
- **Min**: Worst performing bird this episode
- **Steps**: Total steps taken this episode
- **Memory**: Current replay buffer size

## 💾 Saved Models

Multi-bird training saves multiple checkpoints:

1. **`dqn_flappy_bird.pth`**: Final model after all episodes
2. **`dqn_flappy_bird_best.pth`**: Best model during training (highest avg reward)
3. **`dqn_flappy_bird_ep50.pth`**: Checkpoint every 50 episodes
4. **`dqn_flappy_bird_ep100.pth`**, etc.

## 🧪 Testing Your Model

After training, test performance:

```python
from main import test_trained_model

# Test the best model
test_trained_model(model_path="dqn_flappy_bird_best.pth", num_episodes=10)
```

Or use the interactive menu:
```bash
python main.py
# Choose option 3: Test Trained Model
```

## 🔄 Comparison: Single vs Multi-Bird

| Aspect | Single Bird | Multi-Bird (16 birds) |
|--------|-------------|----------------------|
| Experience per episode | ~500 steps | ~8000 steps (16×) |
| Training speed | Slower | 10-15x faster |
| Exploration | Limited | Diverse |
| GPU usage | Low | High (efficient) |
| Memory usage | Low | Moderate |
| Stability | Can be noisy | More stable |
| Wall-clock time | Hours | Minutes |

## 🛠️ Troubleshooting

### Out of Memory (GPU)
- Reduce `num_birds` (try 8 or 4)
- Reduce `batch_size` (try 32)
- Use CPU instead: Set `device = "cpu"` in bird_ai.py

### Training Too Slow
- Disable rendering: `render_first=False`
- Reduce `steps_per_episode`
- Ensure GPU is being used (check the "Using device" message)

### Poor Performance
- Try training longer (more episodes)
- Increase `memory_size` to 5000 or 10000
- Adjust reward structure in `config.py`
- Try different `epsilon_decay` values

### Environment Issues
Make sure all dependencies are installed:
```bash
pip install pygame numpy torch
```

## 📈 Advanced: Monitoring Training

Create a simple training monitor:

```python
import matplotlib.pyplot as plt
from bird_ai import train_dqn_vectorized

# Train and collect history
model = train_dqn_vectorized(num_envs=16)

# Plot will be added in future update
```

## 🎓 How It Works

### Architecture
```
┌─────────────────────────────────────────────┐
│         Shared DQN Network (Brain)          │
│  Input: 5 states → Hidden Layers → 2 Q-vals│
└─────────────────────────────────────────────┘
           ↑                    ↓
    Observations          Actions
           │                    │
    ┌──────┴────────────────────┴──────┐
    │   Vectorized Environment         │
    │  ┌─────┐ ┌─────┐      ┌─────┐  │
    │  │Bird1│ │Bird2│ .... │Bird16│  │
    │  └─────┘ └─────┘      └─────┘  │
    └───────────────────────────────────┘
                  ↓
         Shared Replay Buffer
    [s, a, r, s', done] × 2000
```

### Training Loop
1. All birds observe their states
2. DQN predicts Q-values for all birds (batched)
3. Each bird takes epsilon-greedy action
4. All birds step in their environments
5. Store all transitions in shared replay buffer
6. Sample random batch from buffer
7. Update DQN weights (backprop)
8. Repeat

### Key Differences from Single Bird
- **Batched inference**: Process all birds in one forward pass
- **Parallel exploration**: Each bird explores independently
- **Shared learning**: One network learns from all experiences
- **Auto-reset**: Dead birds reset automatically

## 🔮 Future Enhancements

Potential improvements:
- [ ] Add Double DQN (DDQN) support
- [ ] Implement prioritized experience replay
- [ ] Add TensorBoard logging
- [ ] Multi-processing for true parallelism
- [ ] Tournament selection between birds
- [ ] Curiosity-driven exploration

## 📚 References

- **DQN Paper**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **Vectorized Envs**: Common in modern RL libraries (Stable-Baselines3, RLlib)
- **Population-based Training**: [DeepMind PBT](https://deepmind.com/blog/article/population-based-training-neural-networks)

## 🤝 Contributing

Feel free to experiment and improve! Some ideas:
- Tune hyperparameters for better performance
- Add visualization of multiple birds
- Implement evolutionary strategies
- Create competition mode between populations

---

Happy Training! 🚀🐦

For questions or issues, check the main README.md or create an issue.
