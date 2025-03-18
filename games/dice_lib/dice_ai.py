import itertools
import random


class DiceAI:
    def __init__(self, risk_level: float = 0.7, logic = None):

        self.risk_level = max(0, min(risk_level, 1))
        self.die_logic = None

        if logic is not None:
            self.set_logic(logic)


    def set_logic(self, die_logic):
        """Set the die logic for scoring calculations"""

        self.die_logic = die_logic


    def _get_all_possible_selections(self, dice: list[int]) -> list[list[int]]:
        """Get all possible dice selections"""

        selections = []

        for i in range(1, len(dice) + 1):
            for combo in itertools.combinations(range(len(dice)), i):
                selections.append(list(combo))

        return selections
    

    def _score_selections(self, dice: list[int], selections: list[list[int]]) -> dict[list, int]:
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
    
    
    def make_turn_decision(
            self, 
            current_roll: list[int], 
            player_score: int,
            opponent_scores: list[int],
            turn_score: int, # current turn score
            roll_count: int, # number of rolls taken this turn
            goal: int
    ) -> tuple[bool, list[int]]:
        """
        Make decision based on roll, score, etc.

        Care about:
        1. Ours vs. opponent scores RELATIVE to the goal
        2. How far in to our turn we are compared to money made RELATIVE to the goal
        3. Risk level
        4. Current roll and potential for future rolls
        5. Selection made in sync with continue vs. pass decision (i.e. always collect max value if passing, but think about next turns if continuing)
        
        Scoring a hand:
        1. Logical dice score
        2. # rolls in + score so far
        3. Strategy: pass vs continue, dice left
        """
        
        continue_turn = False
        selected_dice = []


        return continue_turn, selected_dice
    