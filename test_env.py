"""
Test script for Flappy Bird Environment.
This demonstrates the environment interface and can be used for testing.
"""

import pygame
from flappy_bird_env import FlappyBirdEnv
import time

def test_random_agent():
    """Test the environment with a random agent."""
    import random
    
    print("Testing environment with random agent...")
    env = FlappyBirdEnv(render_mode=True)
    
    episodes = 5
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        print(f"Initial state shape: {state.shape}")
        print(f"Initial state: {state}")
        
        done = False
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            # Random action (20% chance to flap)
            action = 1 if random.random() < 0.2 else 0
            
            # Step environment
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            if done:
                print(f"Episode finished!")
                print(f"  Final score: {info['score']}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Steps survived: {steps}")
                time.sleep(2)  # Pause to see game over screen
    
    env.close()
    print("\nTest completed!")


def test_manual_play():
    """Test the environment with manual keyboard control."""
    print("Testing environment with manual control...")
    print("\nControls:")
    print("  SPACE or UP ARROW: Flap")
    print("  R: Reset (when game over)")
    print("  ESC or Q: Quit")
    
    env = FlappyBirdEnv(render_mode=True)
    state = env.reset()
    
    running = True
    while running:
        action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset()
                    print("\nGame reset!")
        
        state, reward, done, info = env.step(action)
        env.render()
        
        if done:
            print(f"\nGame Over! Score: {info['score']}")
            print("Press R to restart or ESC to quit")
    
    env.close()


def test_environment_interface():
    """Test the environment interface without rendering."""
    print("Testing environment interface (no rendering)...")
    
    env = FlappyBirdEnv(render_mode=False)
    
    print(f"State size: {env.get_state_size()}")
    print(f"Action size: {env.get_action_size()}")
    
    # Test reset
    state = env.reset()
    print(f"\nInitial state: {state}")
    print(f"State shape: {state.shape}")
    
    # Test a few steps
    print("\nTesting steps...")
    for i in range(10):
        action = 1 if i % 3 == 0 else 0  # Flap every 3rd step
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}, score={info['score']}")
        
        if done:
            print("Episode ended!")
            break
    
    env.close()
    print("\nInterface test completed!")


if __name__ == "__main__":
    import sys
    
    print("Flappy Bird Environment Test Suite")
    print("=" * 50)
    print("\nSelect test mode:")
    print("1. Random agent (watch AI play randomly)")
    print("2. Manual play (play with keyboard)")
    print("3. Interface test (no rendering, just API)")
    print()
    
    choice = input("Enter choice (1-3) or press Enter for manual play: ").strip()
    
    if choice == "1":
        test_random_agent()
    elif choice == "3":
        test_environment_interface()
    else:
        test_manual_play()
