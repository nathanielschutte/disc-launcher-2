import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import sys

def plot_results(results):
    """Plot the evaluation results"""
    plt.figure(figsize=(12, 8))
    
    # Win rate plot
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    win_rates = [results[name]["win_rate"] for name in names]
    
    bars = plt.bar(names, win_rates)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7)
    
    plt.title("DQN Win Rate vs Rule-Based AIs")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    
    # Annotate bars with values
    for bar, rate in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center')
    
    # Average score plot
    plt.subplot(1, 2, 2)
    
    dqn_scores = [results[name]["dqn_avg_score"] for name in names]
    rule_scores = [results[name]["rule_avg_score"] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, dqn_scores, width, label='DQN AI')
    plt.bar(x + width/2, rule_scores, width, label='Rule-Based AI')
    
    plt.axhline(y=2000, color='g', linestyle='--', alpha=0.7, label='Goal')
    plt.xticks(x, names)
    plt.title("Average Scores")
    plt.ylabel("Score")
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('dice_ai_evaluation_results.png')
    print("Results plot saved as 'dice_ai_evaluation_results.png'")
    
    # Show figure (if in interactive environment)
    plt.show()

from games.dice_lib.logic import DieLogic
from games.dice_lib.dice_ai_claude import DiceAI as RuleBasedAI
from games.dice_lib.dice_ai_dqn import DiceAI as DQN_AI, DiceGameEnvironment, train_dice_ai

# Define a simple progress indicator to replace tqdm if not available
def tqdm(iterable, **kwargs):
    total = len(iterable)
    for i, item in enumerate(iterable):
        if i % 10 == 0 or i == total - 1:
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")
        yield item
    print()  # New line at the end

class DiceGameSimulator:
    """Simulates games between AI agents"""
    
    def __init__(self, die_logic, goal=2000):
        self.die_logic = die_logic
        self.goal = goal
    
    def play_game(self, ai1, ai2, verbose=False):
        """Play a complete game between two AI agents"""
        scores = {0: 0, 1: 0}  # Player ID: Score
        current_player = 0
        
        while scores[0] < self.goal and scores[1] < self.goal:
            turn_score = self._play_turn(
                ai1 if current_player == 0 else ai2,
                scores[current_player],
                scores[1 - current_player],
                verbose
            )
            
            scores[current_player] += turn_score
            if verbose:
                print(f"Player {current_player} scored {turn_score} this turn. Total: {scores[current_player]}")
            
            current_player = 1 - current_player
        
        winner = 0 if scores[0] >= self.goal else 1
        if verbose:
            print(f"Player {winner} wins with a score of {scores[winner]}!")
        
        return winner, scores
    
    def _play_turn(self, ai, player_score, opponent_score, verbose=False):
        """Play a single turn for the given AI"""
        dice_remaining = 6
        turn_score = 0
        roll_count = 0
        
        while True:
            # Roll dice
            current_roll = [np.random.randint(1, 7) for _ in range(dice_remaining)]
            roll_count += 1
            
            if verbose:
                print(f"Roll {roll_count}: {current_roll}")
            
            # Check for bust
            if self.die_logic.check_bust(current_roll):
                if verbose:
                    print("BUST! Turn ends with 0 points.")
                return 0
            
            # Get AI decision
            continue_decision, selected_dice = ai.make_turn_decision(
                current_roll, player_score, opponent_score, turn_score, roll_count, 
                self.goal, dice_remaining
            )
            
            # Calculate score for selection
            selected_values = [current_roll[i] for i in selected_dice]
            selection_score = self.die_logic.score(selected_values)
            
            # Invalid selection
            if selection_score == 0:
                if verbose:
                    print("Invalid selection! Turn ends with 0 points.")
                return 0
            
            turn_score += selection_score
            dice_remaining -= len(selected_dice)
            
            if verbose:
                print(f"Selected dice: {selected_dice}, Score: {selection_score}, Turn total: {turn_score}")
            
            # Reset dice if all used
            if dice_remaining == 0:
                dice_remaining = 6
                if verbose:
                    print("All dice used! Rolling 6 new dice.")
            
            # Check if AI wants to continue or pass
            if not continue_decision:
                if verbose:
                    print(f"AI passes with {turn_score} points.")
                return turn_score
            
            if verbose:
                print("AI continues rolling.")

def train_and_evaluate(train_new_model=True, evaluation_games=100):
    """Train the DQN model and evaluate against rule-based AI
    
    Args:
        train_new_model: Whether to train a new model or use existing
        evaluation_games: Number of games to play for evaluation
        
    Returns:
        Dictionary of results comparing DQN to rule-based AIs
    """
    
    die_logic = DieLogic()
    
    # 1. Train the DQN model or load existing
    if train_new_model:
        print("Training DQN model...")
        start_time = time.time()
        
        # For production, use more episodes (10000+)
        dqn_agent = train_dice_ai(die_logic, episodes=10000, batch_size=64, save_interval=500)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    else:
        print("Skipping training, using pre-trained model if available.")
    
    # 2. Initialize AIs for evaluation
    dqn_ai = DQN_AI(risk_level=0.7)
    dqn_ai.set_logic(die_logic)
    
    # Initialize rule-based AIs with different risk levels
    conservative_ai = RuleBasedAI(risk_level=0.3)
    balanced_ai = RuleBasedAI(risk_level=0.5)
    aggressive_ai = RuleBasedAI(risk_level=0.7)
    very_aggressive_ai = RuleBasedAI(risk_level=0.9)
    
    # Set logic for all AIs
    for ai in [conservative_ai, balanced_ai, aggressive_ai, very_aggressive_ai]:
        ai.set_logic(die_logic)
        
    # 3. Evaluate DQN against rule-based AIs
    print("\nEvaluating DQN against rule-based AIs...")
    simulator = DiceGameSimulator(die_logic, goal=2000)
    
    results = {}
    opponents = {
        "Conservative": conservative_ai,
        "Balanced": balanced_ai, 
        "Aggressive": aggressive_ai,
        "Very Aggressive": very_aggressive_ai
    }
    
    # Run evaluation games
    for opponent_name, opponent_ai in opponents.items():
        dqn_wins = 0
        rule_wins = 0
        dqn_avg_score = 0
        rule_avg_score = 0
        
        print(f"\nEvaluating against {opponent_name} AI ({evaluation_games} games):")
        for i in tqdm(range(evaluation_games)):
            # DQN goes first
            winner, scores = simulator.play_game(dqn_ai, opponent_ai)
            if winner == 0:
                dqn_wins += 1
            else:
                rule_wins += 1
                
            dqn_avg_score += scores[0]
            rule_avg_score += scores[1]
            
            # Rule-based AI goes first
            winner, scores = simulator.play_game(opponent_ai, dqn_ai)
            if winner == 1:
                dqn_wins += 1
            else:
                rule_wins += 1
                
            dqn_avg_score += scores[1]
            rule_avg_score += scores[0]
        
        # Calculate stats
        dqn_win_rate = dqn_wins / (evaluation_games * 2) * 100
        dqn_avg_score /= (evaluation_games * 2)
        rule_avg_score /= (evaluation_games * 2)
        
        results[opponent_name] = {
            "win_rate": dqn_win_rate,
            "dqn_avg_score": dqn_avg_score,
            "rule_avg_score": rule_avg_score
        }
        
        print(f"DQN win rate vs {opponent_name}: {dqn_win_rate:.1f}%")
        print(f"Average scores - DQN: {dqn_avg_score:.1f}, {opponent_name}: {rule_avg_score:.1f}")
    
    # 4. Plot the results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    # 5. Save results
    with open('dice_ai_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    train_new = True
    num_games = 50
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['false', '0', 'no']:
            train_new = False
    
    if len(sys.argv) > 2:
        try:
            num_games = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of games: {sys.argv[2]}, using default: {num_games}")
    
    print(f"Training new model: {train_new}")
    print(f"Evaluation games per opponent: {num_games}")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Run training and evaluation
    results = train_and_evaluate(train_new_model=train_new, evaluation_games=num_games)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for opponent, stats in results.items():
        print(f"DQN vs {opponent}: {stats['win_rate']:.1f}% win rate, " 
              f"Avg scores: DQN={stats['dqn_avg_score']:.1f}, {opponent}={stats['rule_avg_score']:.1f}")
    