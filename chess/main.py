import sys
from env import ChessEnv
# If your environment class is in a file named chess_env.py, change the import below accordingly:
# from chess_env import ChessEnv

# For the sake of a self-contained example, we instantiate ChessEnv directly.
# Make sure you have your ChessEnv class available above this line or imported!

def run_test_game():
    # 1. Initialize the environment
    env = ChessEnv()
    
    print("=" * 50)
    print("        CHESS BENCHMARK ENVIRONMENT TESTER       ")
    print("=" * 50)
    
    # 2. Reset to get the initial observation
    obs = env.reset()
    
    terminated = False
    turn_count = 1
    
    while not terminated:
        # Print the current visual state returned by the environment's observation
        print(f"\n--- Turn {turn_count} ({obs['turn']}) ---")
        print(obs["visual_board"])
        
        if obs["is_check"]:
            print("⚠️  King is in check!")
            
        print(f"Available legal moves:\n{', '.join(obs['legal_moves'])}")
        print("-" * 50)
        
        # 3. Get text input for the current player's action
        action = input(f"Enter move for {obs['turn']} (or 'quit'): ").strip()
        
        if action.lower() == 'quit':
            print("\nTesting aborted by user.")
            sys.exit(0)
            
        # 4. Feed the action into a single environment step
        result = env.step(action)
        
        # Unpack the StepResult components
        obs = result.observation
        reward = result.reward
        terminated = result.terminated
        info = result.info
        
        # 5. Diagnostic Output to verify the environment's step behavior
        print(f"\n[Step Diagnostics]")
        print(f"  • Move Accepted: {info['legal_move']}")
        if "error" in info:
            print(f"  • Environment Error: {info['error']}")
        print(f"  • Immediate Step Reward: {reward}")
        print(f"  • Terminated: {terminated}")
        
        # Only increment turn counter if a legal move actually advanced the board state
        if info['legal_move']:
            turn_count += 1
            
    # Game ended loop termination
    print("\n" + "=" * 50)
    print("                GAME OVER RESULTS                ")
    print("=" * 50)
    print(obs["visual_board"])
    print(f"Final Game Result Code: {info.get('game_result', 'N/A')}")
    print(f"Final Step Reward Value: {reward}")
    print("=" * 50)

if __name__ == "__main__":
    run_test_game()