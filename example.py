"""
Simple example demonstrating the Flappy Bird environment.
This shows the basic usage pattern for integrating with RL algorithms.
"""

from flappy_bird_env import FlappyBirdEnv
import numpy as np
import torch
from bird_ai import DQN
import os

# Load the trained model
def load_trained_model():
    """Load the trained DQN model."""
    env = FlappyBirdEnv(False)
    state_dim = env.STATE_COUNT
    action_dim = env.ACTION_COUNT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)

    model_path = "dqn_flappy_bird.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded trained model from {model_path}")
        return model, device
    else:
        print(f"Model file {model_path} not found!")
        return None, None

# Global model variables
trained_model, model_device = load_trained_model()


def simple_ai_agent(state):
    """
    AI agent that uses the trained DQN model to predict actions.

    Args:
        state: Current state observation

    Returns:
        action: 0 (do nothing) or 1 (flap)
    """
    if trained_model is None:
        # Fallback to simple rule-based logic if model not loaded
        bird_y = state[0]  # Normalized bird y position
        pipe_top = state[3]  # Normalized top of pipe gap
        pipe_bottom = state[4]  # Normalized bottom of pipe gap

        # Calculate the center of the gap
        gap_center = (pipe_top + pipe_bottom) / 2

        # Flap if bird is below the gap center
        if bird_y > gap_center:
            return 1  # Flap
        else:
            return 0  # Do nothing

    # Use trained model for prediction
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model_device)
        q_values = trained_model(state_tensor)
        action = int(torch.argmax(q_values).item())

    return action


def run_simple_ai():
    """Run the game with the trained DQN AI."""
    env = FlappyBirdEnv(render_mode=True)

    if trained_model is not None:
        print("Running trained DQN AI agent...")
        print("The agent uses a deep neural network trained with reinforcement learning.")
    else:
        print("Running simple rule-based AI agent...")
        print("The agent flaps when the bird is below the center of the pipe gap.")
    print("Press ESC to quit\n")
    
    episodes = 10
    best_score = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Check for quit
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env.close()
                    return
            
            # Simple AI decides action
            action = simple_ai_agent(state)
            
            # Step environment
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render
            env.render()
        
        # Episode finished
        score = info['score']
        best_score = max(best_score, score)
        
        print(f"Episode {episode + 1}/{episodes} - Score: {score}, Total Reward: {total_reward:.1f}")
    
    print(f"\nBest Score: {best_score}")
    env.close()


def demonstrate_environment():
    """Demonstrate the environment without rendering (for quick testing)."""
    print("Demonstrating environment interface...\n")
    
    env = FlappyBirdEnv(render_mode=False)
    
    print(f"State space size: {env.get_state_size()}")
    print(f"Action space size: {env.get_action_size()}")
    print()
    
    # Run one episode
    state = env.reset()
    print(f"Initial state: {state}")
    print()
    
    step_count = 0
    total_reward = 0
    
    while step_count < 100:  # Limit to 100 steps for demo
        # Random action
        action = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% chance to flap
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if done:
            print(f"Episode ended at step {step_count}")
            print(f"Final score: {info['score']}")
            print(f"Total reward: {total_reward:.2f}")
            break
    
    env.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    # Run the simple AI agent
    run_simple_ai()
    
    # Uncomment below to run the interface demo instead
    # demonstrate_environment()
