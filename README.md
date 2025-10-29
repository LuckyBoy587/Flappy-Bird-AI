# Flappy Bird AI Environment

A Pygame-based Flappy Bird environment compatible with OpenAI Gym interface, designed for reinforcement learning experiments.

🌐 **[View Project Website](https://luckyboy587.github.io/Flappy-Bird-AI/)**

## Features

- **Gym-style Interface**: `reset()` and `step()` methods for easy integration with RL algorithms
- **Full Pygame Rendering**: Visual game display with sprites and animations
- **Physics Simulation**: Realistic gravity, velocity, and collision detection
- **State Observation**: Normalized state vector containing bird position, velocity, and pipe locations
- **Reward System**: 
  - +1 for passing through each pipe
  - +0.1 for each frame survived
  - -100 for collisions
- **Configurable**: Easy to adjust game parameters like gravity, pipe speed, gap size, etc.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the `sprites/` folder is in the same directory as the Python files.

## Project Structure

```
Flappy Bird AI/
│
├── sprites/                    # Game sprite images
│   ├── background-day.png
│   ├── yellowbird-*.png        # Bird animation frames
│   ├── pipe-green.png
│   ├── base.png
│   └── ...
│
├── flappy_bird_env.py         # Main environment class
├── test_env.py                # Test and demo scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Quick Start - Manual Play

Run the main environment file to play manually:

```bash
python flappy_bird_env.py
```

Controls:
- `SPACE` or `UP ARROW`: Flap
- `R`: Reset (when game over)
- `ESC` or `Q`: Quit

### Using the Environment in Your Code

```python
from flappy_bird_env import FlappyBirdEnv

# Create environment
env = FlappyBirdEnv(render_mode=True)

# Reset to get initial state
state = env.reset()

# Game loop
done = False
total_reward = 0

while not done:
    # Choose action (0 = do nothing, 1 = flap)
    action = 1  # or use your AI model to decide
    
    # Execute action
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render the game
    env.render()
    
    if done:
        print(f"Game Over! Score: {info['score']}")

# Clean up
env.close()
```

### Test Scripts

Run the test suite:

```bash
python test_env.py
```

Options:
1. **Random Agent**: Watch a random agent play
2. **Manual Play**: Play with keyboard controls
3. **Interface Test**: Test the API without rendering

## Environment Details

### State Space

The state is a numpy array with 5 normalized values:
- `bird_y`: Vertical position of the bird (0-1)
- `bird_velocity`: Vertical velocity (-1 to 1)
- `next_pipe_x`: Horizontal distance to next pipe (0-1)
- `next_pipe_top`: Top of the pipe gap (0-1)
- `next_pipe_bottom`: Bottom of the pipe gap (0-1)

### Action Space

- `0`: Do nothing (gravity pulls the bird down)
- `1`: Flap (bird jumps up)

### Rewards

- `+1`: Successfully passed through a pipe
- `+0.1`: Survived another frame
- `-100`: Collision with pipe, ground, or ceiling

### Game Parameters

Key parameters that can be adjusted in the `FlappyBirdEnv` class:

```python
# Window
WINDOW_WIDTH = 288
WINDOW_HEIGHT = 512
FPS = 60

# Physics
GRAVITY = 0.5
FLAP_STRENGTH = -9
MAX_VELOCITY = 10

# Pipes
PIPE_GAP = 120          # Vertical gap size
PIPE_VELOCITY = 3       # Horizontal speed
PIPE_SPACING = 200      # Distance between pipes
```

## Methods

### `reset() -> np.ndarray`
Resets the environment to initial state and returns the starting observation.

### `step(action: int) -> Tuple[np.ndarray, float, bool, Dict]`
Executes one timestep with the given action.

**Returns:**
- `state`: Current observation
- `reward`: Reward for this step
- `done`: Whether the episode has ended
- `info`: Additional information (score, frame count)

### `render()`
Renders the game window using Pygame.

### `close()`
Cleans up Pygame resources.

### `get_state_size() -> int`
Returns the size of the state space (5).

### `get_action_size() -> int`
Returns the size of the action space (2).

## Next Steps: Reinforcement Learning

This environment is ready for RL training. You can integrate it with:

- **Deep Q-Network (DQN)**
- **Policy Gradient methods (REINFORCE, A2C, PPO)**
- **Genetic Algorithms**
- **NEAT (NeuroEvolution of Augmenting Topologies)**

Example structure for RL agent:

```python
import torch
import torch.nn as nn

class FlappyBirdAgent(nn.Module):
    def __init__(self, state_size=5, action_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, state):
        return self.network(state)
```

## Troubleshooting

**Issue**: Sprites not loading
- **Solution**: Make sure the `sprites/` folder is in the same directory as the Python files

**Issue**: Pygame window not opening
- **Solution**: Check that pygame is installed correctly: `pip install pygame`

**Issue**: Game runs too fast/slow
- **Solution**: Adjust the `FPS` constant in `FlappyBirdEnv.__init__()`

## License

This is an educational project. Sprite assets are from the original Flappy Bird game.

## Credits

- Original Flappy Bird game by Dong Nguyen
- Environment implementation for AI/ML educational purposes
