"""
Configuration file for Flappy Bird Environment.
Modify these parameters to adjust game difficulty and behavior.
"""

# Window settings
WINDOW_WIDTH = 288
WINDOW_HEIGHT = 512
FPS = 60  # Frames per second

# Bird physics
BIRD_X_POSITION = 50  # Fixed horizontal position of the bird
GRAVITY = 0.5  # Acceleration due to gravity (increase for harder game)
FLAP_STRENGTH = -9  # Upward velocity when flapping (more negative = higher jump)
MAX_VELOCITY = 10  # Terminal velocity (max falling speed)

# Pipe settings
PIPE_WIDTH = 52
PIPE_GAP = 120  # Vertical gap between pipes (smaller = harder)
PIPE_VELOCITY = 3.5  # Horizontal speed of pipes (higher = harder)
PIPE_SPACING = 250  # Distance between pipes (smaller = more obstacles)

# Minimum and maximum height for pipe gap center
MIN_PIPE_GAP_Y = 100
MAX_PIPE_GAP_Y_OFFSET = 100  # From ground level minus gap and offset

# Ground settings
GROUND_HEIGHT = 112
BASE_VELOCITY = 3  # Scrolling speed of the ground

# Reward settings
REWARD_ALIVE = 0.1  # Reward per frame for staying alive
REWARD_PIPE = 10.0  # Reward for passing through a pipe
REWARD_DEATH = -100  # Penalty for collision

# Bird animation
BIRD_ANIMATION_SPEED = 5  # Frames per animation cycle (lower = faster)
BIRD_COLOR = "yellow"  # Options: "yellow", "blue", "red"

# Rendering
BACKGROUND_TYPE = "day"  # Options: "day", "night"
PIPE_COLOR = "green"  # Options: "green", "red"

# Collision detection
COLLISION_THRESHOLD = 0  # Pixel buffer for collision (increase for easier game)

# Game behavior
ENABLE_BIRD_ROTATION = True  # Whether bird rotates based on velocity
MAX_ROTATION_UP = -25  # Maximum upward rotation angle
MAX_ROTATION_DOWN = 90  # Maximum downward rotation angle
ROTATION_VELOCITY_FACTOR = -3  # How much velocity affects rotation

# State normalization (for RL)
NORMALIZE_STATE = True  # Whether to normalize state values to [0, 1]


# Difficulty presets
DIFFICULTY_PRESETS = {
    "easy": {
        "GRAVITY": 0.4,
        "FLAP_STRENGTH": -10,
        "PIPE_GAP": 150,
        "PIPE_VELOCITY": 2,
        "REWARD_DEATH": -50,
    },
    "normal": {
        "GRAVITY": 0.5,
        "FLAP_STRENGTH": -9,
        "PIPE_GAP": 120,
        "PIPE_VELOCITY": 3,
        "REWARD_DEATH": -100,
    },
    "hard": {
        "GRAVITY": 0.6,
        "FLAP_STRENGTH": -8,
        "PIPE_GAP": 100,
        "PIPE_VELOCITY": 4,
        "REWARD_DEATH": -150,
    },
    "extreme": {
        "GRAVITY": 0.7,
        "FLAP_STRENGTH": -7,
        "PIPE_GAP": 90,
        "PIPE_VELOCITY": 5,
        "REWARD_DEATH": -200,
    }
}


def apply_difficulty_preset(difficulty: str) -> dict:
    """
    Get configuration settings for a specific difficulty level.
    
    Args:
        difficulty: One of "easy", "normal", "hard", "extreme"
    
    Returns:
        Dictionary of configuration parameters
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from {list(DIFFICULTY_PRESETS.keys())}")
    
    return DIFFICULTY_PRESETS[difficulty]


if __name__ == "__main__":
    print("Flappy Bird Environment Configuration")
    print("=" * 50)
    print("\nCurrent settings:")
    print(f"  Window: {WINDOW_WIDTH}x{WINDOW_HEIGHT} @ {FPS} FPS")
    print(f"  Gravity: {GRAVITY}")
    print(f"  Flap Strength: {FLAP_STRENGTH}")
    print(f"  Pipe Gap: {PIPE_GAP}")
    print(f"  Pipe Velocity: {PIPE_VELOCITY}")
    print(f"  Pipe Spacing: {PIPE_SPACING}")
    print(f"\nRewards:")
    print(f"  Alive: +{REWARD_ALIVE}")
    print(f"  Pipe Passed: +{REWARD_PIPE}")
    print(f"  Death: {REWARD_DEATH}")
    print("\nDifficulty Presets:")
    for name, preset in DIFFICULTY_PRESETS.items():
        print(f"  {name.capitalize()}:")
        for key, value in preset.items():
            print(f"    {key}: {value}")
