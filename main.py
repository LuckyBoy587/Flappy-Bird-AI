"""
Main script for training and testing Flappy Bird AI.
Choose between single-bird training (original) or multi-bird training (population-based).
"""

from bird_ai import train_dqn, train_dqn_vectorized, DQN, device, state_dim, action_dim
import torch
import numpy as np
from flappy_bird_env import FlappyBirdEnv
import sys


def test_trained_model(model_path: str = "dqn_flappy_bird.pth", num_episodes: int = 5, render: bool = True):
    """
    Test a trained DQN model by running it in the environment.
    
    Args:
        model_path: Path to the saved model weights
        num_episodes: Number of test episodes to run
        render: Whether to render the game visually
    """
    print(f"\n{'='*60}")
    print(f"TESTING TRAINED MODEL: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = DQN(state_dim, action_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Create environment
    env = FlappyBirdEnv(render_mode=render, initial_flap=True)
    
    all_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Use model to select action (no exploration)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        all_rewards.append(total_reward)
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Average Reward: {np.mean(all_rewards):.2f}")
    print(f"  Max Reward: {np.max(all_rewards):.2f}")
    print(f"  Min Reward: {np.min(all_rewards):.2f}")
    print(f"{'='*60}\n")


def main():
    """Main function with training mode selection."""
    print("\n" + "="*60)
    print("FLAPPY BIRD AI - DQN TRAINING")
    print("="*60)
    print("\nSelect training mode:")
    print("  1. Single Bird Training (Original - 2000 episodes)")
    print("  2. Multi-Bird Training (Population-based - 16 birds)")
    print("  3. Test Trained Model")
    print("  4. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🐦 Starting SINGLE BIRD training...")
        model = train_dqn()
        print("\n✓ Training complete!")
        
        test_choice = input("\nWould you like to test the trained model? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_trained_model()
    
    elif choice == "2":
        print("\n🐦🐦🐦 Starting MULTI-BIRD training...")
        
        # Ask for number of birds
        try:
            num_birds = input("Enter number of birds (default 16): ").strip()
            num_birds = int(num_birds) if num_birds else 16
            if num_birds < 1:
                num_birds = 16
        except ValueError:
            num_birds = 16
        
        # Ask about rendering
        render_choice = input("Render first bird during training? (y/n, default n): ").strip().lower()
        render_first = (render_choice == 'y')
        
        model = train_dqn_vectorized(num_envs=num_birds, render_first=render_first)
        print("\n✓ Training complete!")
        
        test_choice = input("\nWould you like to test the trained model? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_trained_model()
    
    elif choice == "3":
        print("\n🧪 Testing trained model...")
        
        # Ask which model to test
        print("\nAvailable models:")
        print("  1. dqn_flappy_bird.pth (latest)")
        print("  2. dqn_flappy_bird_best.pth (best during training)")
        print("  3. Custom path")
        
        model_choice = input("\nSelect model (1-3): ").strip()
        
        if model_choice == "1":
            model_path = "dqn_flappy_bird.pth"
        elif model_choice == "2":
            model_path = "dqn_flappy_bird_best.pth"
        elif model_choice == "3":
            model_path = input("Enter model path: ").strip()
        else:
            model_path = "dqn_flappy_bird.pth"
        
        num_episodes = input("Number of test episodes (default 5): ").strip()
        num_episodes = int(num_episodes) if num_episodes else 5
        
        test_trained_model(model_path=model_path, num_episodes=num_episodes)
    
    elif choice == "4":
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("Progress has been saved in checkpoint files.")
        sys.exit(0)
