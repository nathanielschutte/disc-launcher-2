"""
AI player for the Dice game
"""

import itertools
from typing import List, Tuple, Dict, Set, Optional
import random

class DiceAI:
    def __init__(self, risk_level: float = 0.7):
        """
        Initialize AI player with a risk level
        
        Args:
            risk_level: How risky the AI's strategy is (0.0 to 1.0)
                        Lower values = more conservative play
                        Higher values = more aggressive play
        """
        self.risk_level = max(0.0, min(1.0, risk_level))  # Clamp between 0 and 1
        self.die_logic = None  # Will be set when imported from main game
    
    def set_logic(self, die_logic):
        """Set the die logic for scoring calculations"""
        self.die_logic = die_logic
    
    def _get_all_possible_selections(self, dice: List[int]) -> List[List[int]]:
        """Get all possible dice selections"""
        selections = []
        for i in range(1, len(dice) + 1):
            # Get all combinations of the given length
            for combo in itertools.combinations(range(len(dice)), i):
                # Create the selection
                selections.append(list(combo))
        return selections
    
    def _score_selections(self, dice: List[int], selections: List[List[int]]) -> Dict[Tuple, int]:
        """
        Score all possible selections and return as {selection_indices: score}
        """
        scores = {}
        for sel in selections:
            selected_dice = [dice[i] for i in sel]
            score = self.die_logic.score(selected_dice)
            if score > 0:  # Only include valid scoring selections
                scores[tuple(sel)] = score
        return scores
    
    def _calculate_expected_value(self, 
                               score_so_far: int, 
                               dice_remaining: int, 
                               goal: int,
                               opponent_score: int) -> float:
        """
        Calculate expected value of continuing vs stopping
        This is a simplified model based on average expected points per die
        
        Returns:
            float: Expected points from continuing
        """
        # Base expected values per die for different numbers of dice
        # These values are approximate and could be refined with more detailed simulation
        expected_per_die = {
            6: 80,  # With 6 dice, expect about 80 points per die
            5: 75,
            4: 70,
            3: 65,
            2: 60,
            1: 50   # With 1 die, you're mostly looking at 50 or 100 points
        }
        
        # Default to lowest if we go below 1
        expected = expected_per_die.get(dice_remaining, 50)
        
        # Calculate bust probability - increases with fewer dice
        bust_probability = max(0.1, 0.3 - (dice_remaining * 0.04))
        
        # Expected value if we continue
        continue_ev = score_so_far + (expected * dice_remaining * (1 - bust_probability))
        
        # If we're close to the goal, adjust strategy
        points_needed = goal - score_so_far
        if points_needed <= 200:
            # More conservative near the goal - prefer to secure the win
            continue_ev *= 0.9
        
        # If opponent is close to winning, we need to be more aggressive
        if goal - opponent_score < 500:
            continue_ev *= 1.1
        
        return continue_ev
    
    def choose_dice(self, 
                current_roll: List[int], 
                player_score: int, 
                opponent_score: int,
                turn_score: int,
                goal: int, 
                dice_remaining: int) -> List[int]:
        """
        Choose which dice to select based on current game state
        
        Args:
            current_roll: Current dice values
            player_score: AI's current score
            opponent_score: Opponent's current score
            turn_score: Points accumulated in the current turn
            goal: Score needed to win
            dice_remaining: Number of dice remaining after this selection
        
        Returns:
            List of dice indices to select (0-based)
        """
        if not self.die_logic:
            raise ValueError("Die logic not set. Call set_logic() first.")
        
        # Get all possible selections
        possible_selections = self._get_all_possible_selections(current_roll)
        
        # Score each selection
        selection_scores = self._score_selections(current_roll, possible_selections)
        
        # If no valid selections, return empty (this should never happen in the game logic)
        if not selection_scores:
            return []
        
        # Find selection with highest score
        best_selection = max(selection_scores.items(), key=lambda x: x[1])
        best_indices = list(best_selection[0])  # Convert tuple to list
        best_score = best_selection[1]
        
        # Check if any selection uses all dice - this is a special case we might want to prioritize
        all_dice_selections = [(sel, score) for sel, score in selection_scores.items() if len(sel) == len(current_roll)]
        if all_dice_selections:
            # Find the highest scoring selection that uses all dice
            best_all_dice = max(all_dice_selections, key=lambda x: x[1])
            # If it's a reasonable score (at least 50% of the best score), use it
            if best_all_dice[1] >= best_score * 0.5:
                return list(best_all_dice[0])
        
        # If we already have enough to win, make the safest choice
        if player_score + turn_score + best_score >= goal:
            return best_indices
        
        # If we have a very high turn score already, prioritize safety
        if turn_score > 1000:
            return best_indices
        
        # Strategy variations by risk level
        if self.risk_level < 0.3:
            # Conservative strategy - prefer to get points safely
            # Find any selection that gives at least 200 points
            for sel, score in sorted(selection_scores.items(), key=lambda x: len(x[0])):
                if score >= 200:
                    return list(sel)
        
        elif self.risk_level > 0.6:  # More aggressive strategy
            # First, check if we can get a good score while preserving dice
            efficient_selections = sorted(
                selection_scores.items(), 
                key=lambda x: (x[1]/len(x[0]), -len(x[0])), 
                reverse=True
            )
            
            # Try to find selections that use fewer dice but still give good scores
            for sel, score in efficient_selections:
                if score >= best_score * 0.6 and len(sel) < len(best_indices):
                    remaining = dice_remaining + (len(best_indices) - len(sel))
                    if remaining >= 2:
                        return list(sel)
            
            # See if we can optimize further by checking for unused 1s or 5s
            # Find all 1s and 5s in the current roll
            ones_and_fives = [(i, val) for i, val in enumerate(current_roll) if val == 1 or val == 5]
            
            # If our best selection doesn't use all 1s and 5s, check if we can add them
            best_sel_indices = set(best_indices)
            unused_ones_fives = [i for i, val in ones_and_fives if i not in best_sel_indices]
            
            if unused_ones_fives:
                # Check score with added 1s and 5s
                extended_indices = best_indices + unused_ones_fives
                remaining_dice = len(current_roll) - len(extended_indices)
                extended_dice = [current_roll[i] for i in extended_indices]
                extended_score = self.die_logic.score(extended_dice)
                
                # Special case: Using all dice is advantageous for a complete re-roll
                if remaining_dice == 0 and extended_score > 0:
                    return extended_indices
                
                # Otherwise, only use this selection if it improves score and leaves at least 1 die
                elif remaining_dice >= 1 and extended_score > best_score:
                    return extended_indices
        
        # Balanced strategy or fallback
        # If we're far from the goal, optimize for efficiency
        points_needed = goal - player_score - turn_score
        
        if points_needed > 800:
            # Look for the most efficient selection (highest points per die)
            selections_by_efficiency = sorted(
                selection_scores.items(), 
                key=lambda x: (x[1]/len(x[0]), -len(x[0])), 
                reverse=True
            )
            
            # Take the most efficient selection if it's good enough
            if selections_by_efficiency[0][1] >= best_score * 0.75:
                efficient_sel = list(selections_by_efficiency[0][0])
                
                # Extra check: If we're using the efficient selection, check if we've missed any 1s or 5s
                efficient_indices = set(efficient_sel)
                missed_ones_fives = [i for i, val in enumerate(current_roll) 
                                    if (val == 1 or val == 5) and i not in efficient_indices]
                
                if missed_ones_fives:
                    # Add these to our selection
                    extended_indices = efficient_sel + missed_ones_fives
                    remaining_dice = len(current_roll) - len(extended_indices)
                    extended_dice = [current_roll[i] for i in extended_indices]
                    extended_score = self.die_logic.score(extended_dice)
                    
                    # If using all dice, strongly prefer this option if it gives a valid score
                    if remaining_dice == 0 and extended_score > 0:
                        return extended_indices
                    # Otherwise only choose if it improves score and leaves at least one die
                    elif remaining_dice >= 1 and extended_score > selections_by_efficiency[0][1]:
                        return extended_indices
                
                return efficient_sel
        
        # If we're behind, prioritize preserving dice over maximum score
        if opponent_score > player_score + 500 and dice_remaining < 6:
            # Look for selections that leave at least half the dice
            for sel, score in sorted(selection_scores.items(), 
                                key=lambda x: (-x[1], len(x[0]))):
                if len(sel) <= len(current_roll) // 2 and score >= best_score * 0.6:
                    return list(sel)
        
        # Default to the highest scoring selection
        # But check if we can add any 1s or 5s that aren't part of the best selection
        best_sel_indices = set(best_indices)
        ones_and_fives = [(i, val) for i, val in enumerate(current_roll) if val == 1 or val == 5]
        unused_ones_fives = [i for i, val in ones_and_fives if i not in best_sel_indices]
        
        # If adding 1s and 5s would use all dice, that's a great option (complete re-roll)
        # Or if it would still leave at least 1 die, try it
        if unused_ones_fives:
            extended_indices = best_indices + unused_ones_fives
            remaining_dice = len(current_roll) - len(extended_indices)
            extended_dice = [current_roll[i] for i in extended_indices]
            extended_score = self.die_logic.score(extended_dice)
            
            # If using all dice, strongly prefer this option if it gives a valid score
            if remaining_dice == 0 and extended_score > 0:
                return extended_indices
            # Otherwise, only choose if it improves the score and leaves at least 1 die
            elif remaining_dice >= 1 and extended_score > best_score:
                return extended_indices
        
        return best_indices
    
    def decide_continue_or_pass(self, 
                               current_score: int, 
                               turn_score: int, 
                               opponent_score: int,
                               goal: int,
                               dice_remaining: int) -> bool:
        """
        Decide whether to continue rolling or pass
        
        Args:
            current_score: AI's current total score
            turn_score: Points accumulated in this turn so far
            opponent_score: Opponent's current score
            goal: Score needed to win
            dice_remaining: Number of dice remaining
            
        Returns:
            True to continue, False to pass
        """
        # If we can win by passing, always pass
        if current_score + turn_score >= goal:
            return False
        
        # Calculate points needed to win
        points_needed = goal - current_score - turn_score
        
        # Calculate expected value of continuing
        expected_value = self._calculate_expected_value(
            turn_score, dice_remaining, goal, opponent_score
        )
        
        # Even lower pass threshold to strongly encourage continuing
        pass_threshold = 180 - (self.risk_level * 200)
        
        # Adjust threshold based on game state
        
        # If we're behind, be extremely aggressive
        if opponent_score > current_score:
            deficit = opponent_score - current_score
            pass_threshold -= min(deficit / 2, 200)  # Even stronger deficit impact
        
        # If opponent is close to winning, be very aggressive
        if goal - opponent_score < 500:
            pass_threshold -= 200
        
        # If we've accumulated a lot this turn, be a bit conservative,
        # but with higher threshold to still encourage continuing
        if turn_score > 700:  # Increased from 500
            pass_threshold += turn_score / 20  # Reduced further from /15
        
        # Dice remaining strongly influences decision
        if dice_remaining <= 2:
            pass_threshold += 60  # Reduced further to encourage continuing with few dice
        elif dice_remaining >= 4:
            # Strong bonus for having lots of dice
            pass_threshold -= 100  # Doubled bonus for having many dice
        elif dice_remaining == 3:
            # Even 3 dice is good enough to continue
            pass_threshold -= 50
        
        # Special case: if we have started accumulating points but aren't at huge risk
        if 200 < turn_score < 600 and dice_remaining >= 3:
            pass_threshold -= 100  # Encourage building on an okay start
        
        # Special case: if we have a hot streak (high turn score)
        # but still far from the goal, be more aggressive
        if turn_score > 300 and points_needed > 800:
            pass_threshold -= 150  # Increased from 100
        
        # Add randomness for unpredictability (weighted by risk level)
        random_factor = random.randint(-80, 120) * self.risk_level  # Skewed toward continuing
        pass_threshold += random_factor
        
        # Make the decision - Only pass if we have zero dice or exceed threshold
        # Significantly reduced chance of passing even when over threshold
        if dice_remaining == 0 or (turn_score > pass_threshold and random.random() > self.risk_level * 0.9):
            return False  # Pass
        else:
            return True   # Continue
            
    def make_turn_decision(self, 
                         current_roll: List[int], 
                         player_score: int, 
                         opponent_score: int,
                         turn_score: int,
                         goal: int,
                         dice_remaining: int) -> Tuple[str, List[int]]:
        """
        Make a complete turn decision - what dice to select and whether to continue
        
        Returns:
            Tuple of (action, dice_indices) where:
                action: 'select', 'continue', or 'pass'
                dice_indices: List of dice indices to select (only for 'select')
        """
        # First, choose which dice to select
        selected_indices = self.choose_dice(
            current_roll, player_score, opponent_score, turn_score, goal, dice_remaining
        )
        
        # Calculate new turn score after this selection
        selected_dice = [current_roll[i] for i in selected_indices]
        selection_score = self.die_logic.score(selected_dice)
        new_turn_score = turn_score + selection_score
        new_dice_remaining = dice_remaining - len(selected_indices)
        
        # If this is our first selection in a turn, return it
        if turn_score == 0:
            return ('select', selected_indices)
        
        # Otherwise, decide whether to continue or pass
        continue_decision = self.decide_continue_or_pass(
            player_score, new_turn_score, opponent_score, goal, new_dice_remaining
        )
        
        if continue_decision:
            return ('continue', selected_indices)
        else:
            return ('pass', selected_indices)


# Example usage:
if __name__ == "__main__":
    # This would be imported from the game in actual implementation
    from games.dice_lib.logic import DieLogic
    
    ai = DiceAI(risk_level=0.6)
    ai.set_logic(DieLogic())
    
    # Example scenario
    roll = [1, 1, 3, 4, 5, 6]
    player_score = 500
    opponent_score = 750
    turn_score = 300
    goal = 2000
    dice_remaining = 6
    
    decision, dice = ai.make_turn_decision(
        roll, player_score, opponent_score, turn_score, goal, dice_remaining
    )
    
    print(f"Roll: {roll}")
    print(f"Decision: {decision}")
    print(f"Selected dice indices: {dice}")
    print(f"Selected dice values: {[roll[i] for i in dice]}")
