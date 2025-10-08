"""
Quick example demonstrating multi-bird training.
Run this to see the population-based training in action!
"""

from bird_ai import train_dqn_vectorized, DQN, device, state_dim, action_dim
from vector_env import VectorFlappyBirdEnv
import torch
import numpy as np


def quick_demo():
    """
    Quick demonstration of multi-bird training with 8 birds for 100 episodes.
    This is a shortened version to show how it works.
    """
    print("\n" + "="*70)
    print("MULTI-BIRD TRAINING DEMO")
    print("="*70)
    print("\nThis demo will train 8 birds simultaneously for 100 episodes.")
    print("It should take just a few minutes to complete.\n")
    
    input("Press Enter to start the demo...")
    
    # Override episodes temporarily for demo
    import bird_ai
    original_episodes = bird_ai.episodes
    bird_ai.episodes = 100  # Short demo
    
    try:
        # Train with 8 birds
        print("\n🚀 Starting demo training...")
        model = train_dqn_vectorized(num_envs=8, render_first=False)
        
        print("\n✅ Demo complete!")
        print("\nThe model has been saved and you can now:")
        print("  1. Continue training with more episodes")
        print("  2. Test the trained model")
        print("  3. Run full training with main.py")
        
    finally:
        # Restore original episodes
        bird_ai.episodes = original_episodes


def compare_single_vs_multi():
    """
    Visual comparison showing the difference between single and multi-bird training.
    """
    print("\n" + "="*70)
    print("SINGLE BIRD vs MULTI-BIRD COMPARISON")
    print("="*70)
    
    print("\n📊 Training Efficiency Comparison:")
    print("-" * 70)
    
    episodes = 2000
    avg_steps_per_episode = 500
    
    # Single bird
    single_bird_steps = episodes * avg_steps_per_episode
    single_bird_time = single_bird_steps / 60  # Assuming 60 FPS
    
    print("\n🐦 SINGLE BIRD TRAINING:")
    print(f"  Episodes: {episodes}")
    print(f"  Average steps per episode: {avg_steps_per_episode}")
    print(f"  Total training steps: {single_bird_steps:,}")
    print(f"  Estimated time: ~{single_bird_time/60:.1f} minutes")
    
    # Multi-bird (16 birds)
    num_birds = 16
    multi_bird_steps = episodes * avg_steps_per_episode * num_birds
    multi_bird_time = single_bird_time  # Same wall-clock episodes
    
    print(f"\n🐦🐦🐦 MULTI-BIRD TRAINING ({num_birds} birds):")
    print(f"  Episodes: {episodes}")
    print(f"  Birds per episode: {num_birds}")
    print(f"  Average steps per episode per bird: {avg_steps_per_episode}")
    print(f"  Total training steps: {multi_bird_steps:,}")
    print(f"  Estimated time: ~{multi_bird_time/60:.1f} minutes")
    
    print("\n✨ BENEFITS:")
    speedup = multi_bird_steps / single_bird_steps
    print(f"  🚀 {speedup}x more experience in the same time!")
    print(f"  🎯 More diverse exploration")
    print(f"  📈 Faster convergence")
    print(f"  💪 Better final performance")
    
    print("\n" + "="*70)


def visualize_vectorized_env():
    """
    Show how the vectorized environment works with a simple example.
    """
    print("\n" + "="*70)
    print("VECTORIZED ENVIRONMENT DEMONSTRATION")
    print("="*70)
    
    print("\nCreating a vectorized environment with 4 birds...")
    vec_env = VectorFlappyBirdEnv(num_envs=4, render_mode=False)
    
    print(f"\n✓ Created {vec_env.num_envs} parallel environments")
    print(f"  State dimension: {vec_env.state_dim}")
    print(f"  Action dimension: {vec_env.action_dim}")
    
    print("\nResetting all environments...")
    states = vec_env.reset()
    print(f"  Initial states shape: {states.shape}")
    print(f"  States for each bird:\n{states}")
    
    print("\nTaking random actions for all birds...")
    actions = np.random.choice([0, 1], size=4)
    print(f"  Actions: {actions} (0=nothing, 1=flap)")
    
    next_states, rewards, dones, infos = vec_env.step(actions)
    
    print(f"\n  Next states shape: {next_states.shape}")
    print(f"  Rewards: {rewards}")
    print(f"  Dones: {dones}")
    
    print("\n✓ All birds can act simultaneously!")
    print("  This is the core of multi-bird training.")
    
    vec_env.close()
    print("\n" + "="*70)


def show_memory_comparison():
    """
    Show how replay memory fills up faster with multiple birds.
    """
    print("\n" + "="*70)
    print("REPLAY MEMORY FILLING SPEED")
    print("="*70)
    
    memory_size = 2000
    batch_size = 64
    steps_per_episode = 500
    
    print(f"\nReplay Memory Settings:")
    print(f"  Maximum size: {memory_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Required before training: {batch_size}")
    
    # Single bird
    episodes_to_fill_single = batch_size / steps_per_episode
    print(f"\n🐦 Single Bird:")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Episodes to start training: {episodes_to_fill_single:.2f}")
    print(f"  Episodes to fill memory: {memory_size / steps_per_episode:.2f}")
    
    # Multi-bird
    num_birds = 16
    steps_per_episode_multi = steps_per_episode * num_birds
    episodes_to_fill_multi = batch_size / steps_per_episode_multi
    
    print(f"\n🐦🐦🐦 Multi-Bird ({num_birds} birds):")
    print(f"  Steps per episode: {steps_per_episode_multi}")
    print(f"  Episodes to start training: {episodes_to_fill_multi:.2f}")
    print(f"  Episodes to fill memory: {memory_size / steps_per_episode_multi:.2f}")
    
    speedup = episodes_to_fill_single / episodes_to_fill_multi
    print(f"\n⚡ Memory fills {speedup:.1f}x faster with {num_birds} birds!")
    
    print("\n" + "="*70)


def main_menu():
    """Interactive menu for demonstrations."""
    while True:
        print("\n" + "="*70)
        print("MULTI-BIRD TRAINING EXAMPLES & DEMONSTRATIONS")
        print("="*70)
        print("\nChoose a demonstration:")
        print("  1. Quick Training Demo (8 birds, 100 episodes)")
        print("  2. Single vs Multi-Bird Comparison")
        print("  3. Vectorized Environment Demo")
        print("  4. Replay Memory Analysis")
        print("  5. Run Full Training (main.py)")
        print("  6. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            quick_demo()
        elif choice == "2":
            compare_single_vs_multi()
        elif choice == "3":
            visualize_vectorized_env()
        elif choice == "4":
            show_memory_comparison()
        elif choice == "5":
            print("\nLaunching main training script...")
            import main
            main.main()
            break
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
        print("Goodbye!")
