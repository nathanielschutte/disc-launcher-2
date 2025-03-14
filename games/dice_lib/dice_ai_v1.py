"""
AI player for the Dice game
"""

import itertools
from typing import List, Tuple, Dict
import random


class DiceAI:
    def __init__(self, risk_level: float = 0.5):
        """
        Initialize AI player with a risk level
        
        Args:
            risk_level: How risky the AI's strategy is (0.0 to 1.0)
                        Lower values = more conservative play
                        Higher values = more aggressive play
        """

        self.risk_level = max(0.0, min(1.0, risk_level))  # Clamp between 0 and 1
        self.die_logic = None
    

    def set_logic(self, die_logic):
        """Set the die logic for scoring calculations"""

        self.die_logic = die_logic
    
    
    def _get_all_possible_selections(self, dice: List[int]) -> List[List[int]]:
        """Get all possible dice selections"""

        selections = []
        for i in range(1, len(dice) + 1):
            for combo in itertools.combinations(range(len(dice)), i):
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
            if score > 0:
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
        
        expected = expected_per_die.get(dice_remaining, 50)
        
        bust_probability = max(0.1, 0.3 - (dice_remaining * 0.04))
        
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
        
        # Now find the expected value of continuing vs stopping
        # This helps the AI decide whether to select a suboptimal combo to preserve more dice
        
        # If we already have enough to win, make the safest choice
        if player_score + turn_score + best_score >= goal:
            return best_indices
        
        # Strategy variations by risk level
        if self.risk_level < 0.3:
            # Conservative strategy - prefer to get points safely
            # Find any selection that gives at least 200 points
            for sel, score in sorted(selection_scores.items(), key=lambda x: len(x[0])):
                if score >= 200:
                    return list(sel)
            
        elif self.risk_level > 0.7:
            # Aggressive strategy - try to get the most points possible
            # Look for selections that use fewer dice but still give good scores
            for sel, score in sorted(selection_scores.items(), key=lambda x: (x[1]/len(x[0]), -len(x[0])), reverse=True):
                # If this selection gives at least 70% of the best score but uses fewer dice
                if score >= best_score * 0.7 and len(sel) < len(best_indices):
                    remaining = dice_remaining + (len(best_indices) - len(sel))
                    if remaining >= 3:  # Only if we'll have enough dice left to be worthwhile
                        return list(sel)
        
        # Balanced strategy (or fallback)
        # If we're far from the goal, take more risks
        points_needed = goal - player_score - turn_score
        
        if points_needed > 1000:
            # Look for efficient selections (high points per die)
            selections_by_efficiency = sorted(
                selection_scores.items(), 
                key=lambda x: (x[1]/len(x[0]), -len(x[0])), 
                reverse=True
            )
            
            # Take the most efficient selection if it's at least 80% as good as the best
            if selections_by_efficiency[0][1] >= best_score * 0.8:
                return list(selections_by_efficiency[0][0])
        
        # Default to the highest scoring selection
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
        
        # Base pass threshold on risk level and current situation
        pass_threshold = 300 - (self.risk_level * 200)
        
        # Adjust threshold based on game state
        
        # If we're behind, be more aggressive
        if opponent_score > current_score:
            deficit = opponent_score - current_score
            pass_threshold -= min(deficit / 5, 100)
        
        # If opponent is close to winning, be more aggressive
        if goal - opponent_score < 500:
            pass_threshold -= 100
        
        # If we've accumulated a lot this turn, be more conservative
        if turn_score > 500:
            pass_threshold += turn_score / 10
        
        # If we have few dice left, be more cautious
        if dice_remaining <= 2:
            pass_threshold += 100
        
        # Make the decision
        if turn_score > pass_threshold or dice_remaining == 0:
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
