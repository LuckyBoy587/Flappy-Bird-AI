# 🚀 Quick Start: Multi-Bird Training

Get started with population-based Flappy Bird AI training in 5 minutes!

## ⚡ Quick Start (3 Steps)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pygame` - For game rendering
- `numpy` - For numerical operations
- `torch` - For deep learning (PyTorch)

### 2️⃣ Run Training

**Option A: Interactive Menu (Recommended)**
```bash
python main.py
```
Then select option **2** for multi-bird training.

**Option B: Direct Demo**
```bash
python demo_multi_bird.py
```
Choose option **1** for a quick training demo.

**Option C: Python Code**
```python
from bird_ai import train_dqn_vectorized

# Train with 16 birds
model = train_dqn_vectorized(num_envs=16, render_first=False)
```

### 3️⃣ Test Your Model

```bash
python main.py
```
Then select option **3** to test the trained model.

---

## 📋 What You'll See

### During Training
```
Episode    1/2000 | Avg Reward:  -85.23 | Max:   12.50 | Min: -100.00 | Epsilon: 0.995
Episode    2/2000 | Avg Reward:  -72.45 | Max:   45.20 | Min: -100.00 | Epsilon: 0.990
Episode    3/2000 | Avg Reward:  -58.12 | Max:   67.80 | Min:  -89.40 | Epsilon: 0.985
...
Episode   50/2000 | Avg Reward:   34.56 | Max:  178.90 | Min:   -5.20 | Epsilon: 0.778
  → Checkpoint saved: dqn_flappy_bird_ep50.pth
  → Last 50 episodes avg: 23.45
```

### After Training
You'll have these saved models:
- `dqn_flappy_bird.pth` - Final trained model
- `dqn_flappy_bird_best.pth` - Best performing model
- `dqn_flappy_bird_ep50.pth`, `ep100.pth`, etc. - Checkpoints

---

## 🎮 Usage Examples

### Example 1: Train with Different Population Sizes

```python
from bird_ai import train_dqn_vectorized

# Small population (fast, good for testing)
model = train_dqn_vectorized(num_envs=8, render_first=False)

# Medium population (recommended)
model = train_dqn_vectorized(num_envs=16, render_first=False)

# Large population (powerful GPU needed)
model = train_dqn_vectorized(num_envs=32, render_first=False)
```

### Example 2: Train and Watch

```python
# Train with visual feedback (slower, but fun to watch!)
model = train_dqn_vectorized(num_envs=16, render_first=True)
```

### Example 3: Test Trained Model

```python
from main import test_trained_model

# Test the latest model
test_trained_model(model_path="dqn_flappy_bird.pth", num_episodes=10)

# Test the best model
test_trained_model(model_path="dqn_flappy_bird_best.pth", num_episodes=10)
```

### Example 4: Continue Training

```python
# If dqn_flappy_bird.pth exists, training will automatically continue
model = train_dqn_vectorized(num_envs=16, render_first=False)
```

---

## 🎯 Recommended Settings

### For Beginners
```python
num_birds = 8          # Easier on your hardware
episodes = 1000        # Shorter training time
render_first = False   # Faster training
```

### For Best Results
```python
num_birds = 16         # Good balance
episodes = 2000        # Full training
render_first = False   # Maximum speed
```

### For High-End Machines
```python
num_birds = 32         # More parallel learning
episodes = 3000        # Extended training
render_first = False   # Still recommend no rendering
```

---

## 🔧 Configuration

All training parameters are in `bird_ai.py`:

```python
# Training hyperparameters
gamma = 0.99              # Discount factor
epsilon = 1.0             # Initial exploration rate
epsilon_min = 0.01        # Minimum exploration rate
epsilon_decay = 0.995     # Exploration decay per episode
lr = 0.001                # Learning rate
episodes = 2000           # Total episodes
batch_size = 64           # Minibatch size
memory_size = 2000        # Replay buffer size

# Multi-bird settings
num_birds = 16            # Population size
steps_per_episode = 5000  # Max steps per episode
```

Game difficulty can be adjusted in `config.py`:

```python
# Make game easier
PIPE_GAP = 150           # Larger gap (default: 120)
PIPE_VELOCITY = 3.0      # Slower pipes (default: 3.5)
GRAVITY = 0.4            # Less gravity (default: 0.5)

# Make game harder
PIPE_GAP = 100           # Smaller gap
PIPE_VELOCITY = 4.0      # Faster pipes
GRAVITY = 0.6            # More gravity
```

---

## 📊 Understanding the Output

### Metrics Explained

- **Avg Reward**: Average performance across all birds (main metric to watch)
- **Max Reward**: Best bird's score (shows peak performance)
- **Min Reward**: Worst bird's score (shows consistency)
- **Epsilon**: Current exploration rate (decreases over time)
- **Steps**: Total steps taken in this episode
- **Memory**: Current size of replay buffer (max: 2000)

### What to Expect

| Episode Range | Avg Reward | What's Happening |
|--------------|------------|------------------|
| 1-50         | -80 to -20 | Learning basics, dying quickly |
| 50-200       | -20 to +30 | Getting through first few pipes |
| 200-500      | +30 to +80 | Consistently passing pipes |
| 500-1000     | +80 to +150| Strong performance, optimizing |
| 1000-2000    | +150 to +250+| Expert level, fine-tuning |

> **Note**: These are approximate ranges. Your results may vary based on configuration and luck!

---

## 🐛 Troubleshooting

### Problem: "Out of memory" error

**Solution:**
```python
# Reduce number of birds
model = train_dqn_vectorized(num_envs=4, render_first=False)

# Or reduce batch size in bird_ai.py
batch_size = 32  # Instead of 64
```

### Problem: Training is very slow

**Solution:**
- Ensure rendering is OFF: `render_first=False`
- Check if GPU is being used (you'll see "Using device: cuda")
- Close other GPU-intensive applications
- Reduce number of birds

### Problem: Birds not improving

**Solutions:**
1. Train longer (increase `episodes`)
2. Check that epsilon is decreasing (shown in output)
3. Adjust reward structure in `config.py`
4. Try different learning rate: `lr = 0.0005` or `lr = 0.002`

### Problem: "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install individually
pip install pygame numpy torch
```

### Problem: Can't find saved models

**Solution:**
Models are saved in the same directory as your scripts. Check:
```bash
# Windows PowerShell
ls *.pth

# Or in Python
import os
print(os.listdir('.'))
```

---

## 💡 Tips for Success

### Training Tips
1. **Start small**: Test with 8 birds and 100 episodes first
2. **Monitor progress**: Watch the "Avg Reward" metric
3. **Save checkpoints**: They're saved every 50 episodes automatically
4. **Be patient**: Good performance takes 500-1000 episodes
5. **Don't render during training**: It's much faster without graphics

### Testing Tips
1. **Test periodically**: Every 100-200 episodes
2. **Compare checkpoints**: Test different saved models
3. **Watch for consistency**: Good models score well every time
4. **Analyze failures**: Watch where birds fail to adjust config

### Optimization Tips
1. **GPU is faster**: Make sure PyTorch is using CUDA if available
2. **More birds = faster learning**: Up to GPU memory limits
3. **Balance memory size**: 2000-5000 is usually good
4. **Experiment with rewards**: Adjust in `config.py`

---

## 📚 Next Steps

After your first successful training:

1. **Read the full guide**: See `MULTI_BIRD_TRAINING.md`
2. **Tune hyperparameters**: Experiment with different settings
3. **Adjust difficulty**: Modify `config.py`
4. **Compare models**: Test single-bird vs multi-bird
5. **Track progress**: Add your own logging/visualization

---

## 🎓 Learning Resources

### Understanding DQN
- The training uses Deep Q-Network (DQN) algorithm
- Birds learn by trial and error
- Network predicts "Q-values" = expected future reward
- Actions chosen to maximize Q-values

### Multi-Bird Advantage
- All birds share one "brain" (neural network)
- Each bird explores different strategies
- Network learns from all birds' experiences
- Faster convergence than training one bird

### Code Structure
```
main.py                     # Interactive training menu
├── bird_ai.py              # DQN training logic
│   ├── train_dqn()         # Single-bird training
│   └── train_dqn_vectorized()  # Multi-bird training
├── vector_env.py           # Vectorized environment wrapper
└── flappy_bird_env.py      # Core game environment
```

---

## 🤝 Getting Help

If you run into issues:

1. Check this quick start guide
2. Read `MULTI_BIRD_TRAINING.md` for details
3. Review `README.md` for environment setup
4. Try the demo: `python demo_multi_bird.py`

---

## ✅ Checklist for First Training

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Can run `python main.py` without errors
- [ ] Choose multi-bird training (option 2)
- [ ] Set reasonable number of birds (8-16)
- [ ] Disable rendering for speed
- [ ] Monitor "Avg Reward" increasing
- [ ] Wait for at least 500 episodes
- [ ] Test the trained model
- [ ] Celebrate your AI success! 🎉

---

**Ready to train? Run `python main.py` and choose option 2!** 🚀🐦

For detailed information, see `MULTI_BIRD_TRAINING.md`.
