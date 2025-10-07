import pygame
import random
import numpy as np
from typing import Tuple, Dict, Any

# Try to import user configuration. If missing, we'll fall back to defaults
try:
    import config as cfg
except Exception:
    cfg = None

class FlappyBirdEnv:
    """
    Flappy Bird Environment compatible with OpenAI Gym style interface.
    
    Action Space:
        0: Do nothing (let gravity pull the bird down)
        1: Flap (make the bird jump up)
    
    State Space:
        - bird_y: Vertical position of the bird
        - bird_velocity: Vertical velocity of the bird
        - next_pipe_x: Horizontal distance to the next pipe
        - next_pipe_top: Height of the top pipe opening
        - next_pipe_bottom: Height of the bottom pipe opening
    
    Reward:
        +1: Successfully passed through a pipe
        -100: Collision with pipe or ground
        +0.1: Still alive (small reward for survival)
    """
    
    def __init__(self, render_mode: bool = True, initial_flap: bool = True):
        # Game constants (read from config.py when available)
        self.WINDOW_WIDTH = getattr(cfg, 'WINDOW_WIDTH', 288)
        self.WINDOW_HEIGHT = getattr(cfg, 'WINDOW_HEIGHT', 512)
        self.FPS = getattr(cfg, 'FPS', 60)
        self.STATE_COUNT = 5  # Number of state variables
        self.ACTION_COUNT = 2  # Number of possible actions

        # Bird physics
        # try a few common names for BIRD_X
        self.BIRD_X = getattr(cfg, 'BIRD_X_POSITION', getattr(cfg, 'BIRD_X', 50))
        self.GRAVITY = getattr(cfg, 'GRAVITY', 0.5)
        self.FLAP_STRENGTH = getattr(cfg, 'FLAP_STRENGTH', -9)
        self.MAX_VELOCITY = getattr(cfg, 'MAX_VELOCITY', 10)

        # Pipe settings
        self.PIPE_WIDTH = getattr(cfg, 'PIPE_WIDTH', 52)
        self.PIPE_GAP = getattr(cfg, 'PIPE_GAP', 120)
        self.PIPE_VELOCITY = getattr(cfg, 'PIPE_VELOCITY', 3)
        self.PIPE_SPACING = getattr(cfg, 'PIPE_SPACING', 200)

        # Ground settings
        self.GROUND_HEIGHT = getattr(cfg, 'GROUND_HEIGHT', 112)
        self.GROUND_Y = self.WINDOW_HEIGHT - self.GROUND_HEIGHT
        self.BASE_VELOCITY = getattr(cfg, 'BASE_VELOCITY', 3)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # If True, the bird will do one flap/jump automatically after reset
        # (useful so players have a moment to react to the first pipe).
        self.initial_flap = initial_flap

        # Rewards (from config when available)
        self.REWARD_ALIVE = getattr(cfg, 'REWARD_ALIVE', 0.1)
        self.REWARD_PIPE = getattr(cfg, 'REWARD_PIPE', 1.0)
        self.REWARD_DEATH = getattr(cfg, 'REWARD_DEATH', -100)

        # Animation & rendering options
        self.BIRD_ANIMATION_SPEED = getattr(cfg, 'BIRD_ANIMATION_SPEED', 5)
        self.BIRD_COLOR = getattr(cfg, 'BIRD_COLOR', 'yellow')
        self.BACKGROUND_TYPE = getattr(cfg, 'BACKGROUND_TYPE', 'day')
        self.PIPE_COLOR = getattr(cfg, 'PIPE_COLOR', 'green')

        # Rotation behavior
        self.ENABLE_BIRD_ROTATION = getattr(cfg, 'ENABLE_BIRD_ROTATION', True)
        self.MAX_ROTATION_UP = getattr(cfg, 'MAX_ROTATION_UP', -25)
        self.MAX_ROTATION_DOWN = getattr(cfg, 'MAX_ROTATION_DOWN', 90)
        self.ROTATION_VELOCITY_FACTOR = getattr(cfg, 'ROTATION_VELOCITY_FACTOR', -3)

        # State normalization
        self.NORMALIZE_STATE = getattr(cfg, 'NORMALIZE_STATE', True)

        # If rendering is enabled, initialize pygame and create the window
        # early so that event handling (pygame.event.get()) works before
        # render() is first called.
        if self.render_mode:
            pygame.init()
            try:
                self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                pygame.display.set_caption("Flappy Bird AI Environment")
            except Exception:
                # In some environments creating a display may fail; leave screen
                # as None and let render() attempt to create it later.
                self.screen = None
            self.clock = pygame.time.Clock()

        # Load sprites
        self._load_sprites()

        # Game state
        self.reset()
    
    def _load_sprites(self):
        """Load all game sprites."""
        sprite_path = "sprites/"
        
        # Background
        self.background_sprite = pygame.image.load(f"{sprite_path}background-day.png")
        
        # Bird sprites for animation
        self.bird_sprites = [
            pygame.image.load(f"{sprite_path}yellowbird-downflap.png"),
            pygame.image.load(f"{sprite_path}yellowbird-midflap.png"),
            pygame.image.load(f"{sprite_path}yellowbird-upflap.png"),
        ]
        self.bird_index = 0
        self.bird_animation_counter = 0
        
        # Pipes
        self.pipe_sprite = pygame.image.load(f"{sprite_path}pipe-green.png")
        self.pipe_sprite_flipped = pygame.transform.flip(self.pipe_sprite, False, True)
        
        # Base (ground)
        self.base_sprite = pygame.image.load(f"{sprite_path}base.png")
        self.base_width = self.base_sprite.get_width()
        
        # Game over sprite
        self.gameover_sprite = pygame.image.load(f"{sprite_path}gameover.png")
        
        # Get bird dimensions
        self.bird_width = self.bird_sprites[0].get_width()
        self.bird_height = self.bird_sprites[0].get_height()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state as numpy array
        """
        # Bird state
        self.bird_y = self.WINDOW_HEIGHT // 2
        # Apply an initial flap on reset if configured. This gives the player
        # a small upward movement so they have time to react to the next pipe.
        if self.initial_flap:
            self.bird_velocity = self.FLAP_STRENGTH
        else:
            self.bird_velocity = 0
        
        # Pipes
        self.pipes = []
        self._generate_pipe(self.WINDOW_WIDTH)
        self._generate_pipe(self.WINDOW_WIDTH + self.PIPE_SPACING)
        
        # Ground scrolling
        self.base_x = 0
        
        # Game state
        self.score = 0
        self.done = False
        self.frame_count = 0
        
        # Track which pipes have been scored
        self.scored_pipes = set()
        
        return self._get_state()
    
    def _generate_pipe(self, x_position: int):
        """
        Generate a new pipe at the specified x position.
        
        Args:
            x_position: Horizontal position for the pipe
        """
        # Random gap position (top of the gap)
        gap_y = random.randint(100, self.GROUND_Y - self.PIPE_GAP - 100)
        
        pipe = {
            'x': x_position,
            'gap_y': gap_y,  # Top of the gap
            'scored': False
        }
        self.pipes.append(pipe)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: 0 (do nothing) or 1 (flap)
        
        Returns:
            state: Current state observation
            reward: Reward for this step
            done: Whether the episode has ended
            info: Additional information
        """
        if self.done:
            return self._get_state(), 0, True, {'score': self.score}
        
        self.frame_count += 1
        reward = 0.1  # Small reward for staying alive
        
        # Apply action
        if action == 1:  # Flap
            self.bird_velocity = self.FLAP_STRENGTH
        
        # Apply gravity
        self.bird_velocity += self.GRAVITY
        self.bird_velocity = min(self.bird_velocity, self.MAX_VELOCITY)
        self.bird_y += self.bird_velocity
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= self.PIPE_VELOCITY
        
        # Remove off-screen pipes and generate new ones
        if self.pipes and self.pipes[0]['x'] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
        
        if self.pipes and self.pipes[-1]['x'] < self.WINDOW_WIDTH - self.PIPE_SPACING:
            self._generate_pipe(self.WINDOW_WIDTH)
        
        # Update ground position
        self.base_x -= self.BASE_VELOCITY
        if self.base_x <= -self.base_width:
            self.base_x = 0
        
        # Check for scoring (passing through pipes)
        for pipe in self.pipes:
            if pipe['x'] + self.PIPE_WIDTH < self.BIRD_X and id(pipe) not in self.scored_pipes:
                self.score += 1
                reward += 1
                self.scored_pipes.add(id(pipe))
        
        # Check collisions
        if self._check_collision():
            self.done = True
            reward = -100
        
        info = {
            'score': self.score,
            'frame_count': self.frame_count
        }
        
        return self._get_state(), reward, self.done, info
    
    def _check_collision(self) -> bool:
        """
        Check if the bird has collided with pipes or ground.
        
        Returns:
            True if collision detected, False otherwise
        """
        # Check ground collision
        if self.bird_y + self.bird_height >= self.GROUND_Y:
            return True
        
        # Check ceiling collision
        if self.bird_y <= 0:
            return True
        
        # Check pipe collision
        bird_rect = pygame.Rect(
            self.BIRD_X,
            self.bird_y,
            self.bird_width,
            self.bird_height
        )
        
        for pipe in self.pipes:
            # Check if bird is horizontally aligned with pipe
            if (pipe['x'] < self.BIRD_X + self.bird_width and
                pipe['x'] + self.PIPE_WIDTH > self.BIRD_X):
                
                # Check collision with top pipe
                if self.bird_y < pipe['gap_y']:
                    return True
                
                # Check collision with bottom pipe
                if self.bird_y + self.bird_height > pipe['gap_y'] + self.PIPE_GAP:
                    return True
        
        return False
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state of the environment.
        
        Returns:
            State as numpy array with normalized values
        """
        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.PIPE_WIDTH > self.BIRD_X:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            # No pipes ahead (shouldn't happen in normal gameplay)
            next_pipe_x = self.WINDOW_WIDTH
            next_pipe_top = 0
            next_pipe_bottom = self.WINDOW_HEIGHT
        else:
            next_pipe_x = next_pipe['x']
            next_pipe_top = next_pipe['gap_y']
            next_pipe_bottom = next_pipe['gap_y'] + self.PIPE_GAP
        
        # Create state vector
        state = np.array([
            self.bird_y / self.WINDOW_HEIGHT,  # Normalized bird y position
            self.bird_velocity / self.MAX_VELOCITY,  # Normalized velocity
            (next_pipe_x - self.BIRD_X) / self.WINDOW_WIDTH,  # Normalized horizontal distance to pipe
            next_pipe_top / self.WINDOW_HEIGHT,  # Normalized top pipe position
            next_pipe_bottom / self.WINDOW_HEIGHT,  # Normalized bottom pipe position
        ], dtype=np.float32)
        
        return state
    
    def render(self):
        """Render the game using Pygame."""
        if not self.render_mode:
            return
        
        # Initialize Pygame if not already done
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Flappy Bird AI Environment")
            self.clock = pygame.time.Clock()
        
        # Draw background
        self.screen.blit(self.background_sprite, (0, 0))
        
        # Draw pipes
        for pipe in self.pipes:
            # Top pipe
            top_pipe_height = pipe['gap_y']
            self.screen.blit(
                self.pipe_sprite_flipped,
                (pipe['x'], top_pipe_height - self.pipe_sprite.get_height())
            )
            
            # Bottom pipe
            bottom_pipe_y = pipe['gap_y'] + self.PIPE_GAP
            self.screen.blit(
                self.pipe_sprite,
                (pipe['x'], bottom_pipe_y)
            )
        
        # Draw ground (two copies for seamless scrolling)
        self.screen.blit(self.base_sprite, (self.base_x, self.GROUND_Y))
        self.screen.blit(self.base_sprite, (self.base_x + self.base_width, self.GROUND_Y))
        
        # Animate bird
        self.bird_animation_counter += 1
        if self.bird_animation_counter % 8 == 0:
            self.bird_index = (self.bird_index + 1) % len(self.bird_sprites)
        
        # Rotate bird based on velocity
        bird_surface = self.bird_sprites[self.bird_index]
        rotation = max(-25, min(self.bird_velocity * -3, 90))
        rotated_bird = pygame.transform.rotate(bird_surface, rotation)
        
        # Draw bird
        bird_rect = rotated_bird.get_rect(center=(self.BIRD_X + self.bird_width // 2, 
                                                    self.bird_y + self.bird_height // 2))
        self.screen.blit(rotated_bird, bird_rect)
        
        # Draw score
        self._draw_score()
        
        # Draw game over if done
        if self.done:
            gameover_rect = self.gameover_sprite.get_rect(
                center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 3)
            )
            self.screen.blit(self.gameover_sprite, gameover_rect)
        
        # Update display
        pygame.display.update()
        self.clock.tick(self.FPS)
    
    def _draw_score(self):
        """Draw the current score on the screen."""
        font = pygame.font.Font(None, 50)
        score_surface = font.render(str(self.score), True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(self.WINDOW_WIDTH // 2, 50))
        
        # Draw shadow for better visibility
        shadow_surface = font.render(str(self.score), True, (0, 0, 0))
        shadow_rect = shadow_surface.get_rect(center=(self.WINDOW_WIDTH // 2 + 2, 52))
        
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(score_surface, score_rect)
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
    
    def get_state_size(self) -> int:
        """Return the size of the state space."""
        return 5
    
    def get_action_size(self) -> int:
        """Return the size of the action space."""
        return 2


if __name__ == "__main__":
    # Demo: Play the game manually
    env = FlappyBirdEnv(render_mode=True)
    state = env.reset()
    
    print("Controls:")
    print("  SPACE or UP ARROW: Flap")
    print("  ESC or Q: Quit")
    print("\nStarting game...")
    
    running = True
    while running:
        # Handle events
        action = 0  # Default: do nothing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    action = 1  # Flap
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r and env.done:
                    # Reset on 'R' key when game over
                    state = env.reset()
                    print("\nGame reset!")
        
        # Step the environment
        state, reward, done, info = env.step(action)
        print(state)
        # Render
        env.render()
        
        # Print info when scoring or game over
        if reward > 1:
            print(f"Score: {info['score']}")
        
        if done and not env.done:
            print(f"\nGame Over! Final Score: {info['score']}")
            print("Press R to restart or ESC to quit")
    
    env.close()
    print("\nGame closed.")
