
class DieLogic:
    def __init__(self):
        pass


    def score(self, selected: list, allow_extra=False) -> int:
        """
        Score selection of dice according to the rules:
        1. Each individual 1 worth 100 points
        2. Each individual 5 worth 50 points
        3. Partial straight (1-5, or 2-6) worth 500 and 750 points respectively
        4. Full straight (1-6) worth 1500
        5. Three 1s worth 1000 points
        6. All other three of a kind worth 100 * the dice number
        7. Four or more of a kind double the 3 of a kind value for each additional die
        
        Dice can only contribute to one scoring pattern.
        """

        if len(selected) == 0:
            return 0
        
        dice = selected.copy()
        dice.sort()
        
        score = 0
        used_dice = set()
        
        if len(dice) >= 6 and set(dice) == {1, 2, 3, 4, 5, 6}:
            used_dice = set(range(len(dice)))
            return 1500

        if len(dice) >= 5:
            if set(dice).issuperset({1, 2, 3, 4, 5}) and len(set(dice) & {1, 2, 3, 4, 5}) == 5:
                for i, die in enumerate(dice):
                    if die in {1, 2, 3, 4, 5}:
                        used_dice.add(i)
                score += 500
            
            if set(dice).issuperset({2, 3, 4, 5, 6}) and len(set(dice) & {2, 3, 4, 5, 6}) == 5:
                if score == 0:
                    for i, die in enumerate(dice):
                        if die in {2, 3, 4, 5, 6}:
                            used_dice.add(i)
                    score += 750
        
        counts = {}
        for die in set(dice):
            counts[die] = dice.count(die)
        
        for die, count in counts.items():
            if count >= 3:
                positions = [i for i, d in enumerate(dice) if d == die and i not in used_dice]
                # ilter out already used dice
                available_positions = [pos for pos in positions if pos not in used_dice]
                
                if len(available_positions) < 3:
                    continue
                
                base_score = 1000 if die == 1 else die * 100
                
                for pos in available_positions[:3]:
                    used_dice.add(pos)
                
                extra_count = min(len(available_positions) - 3, count - 3)
                multiplier = 2 ** extra_count
                
                for pos in available_positions[3:3+extra_count]:
                    used_dice.add(pos)
                
                score += base_score * multiplier
        
        for i, die in enumerate(dice):
            if i not in used_dice:
                if die == 1:
                    score += 100
                    used_dice.add(i)
                elif die == 5:
                    score += 50
                    used_dice.add(i)
        
        if allow_extra or len(used_dice) == len(dice):
            return score
        else:
            return 0

    def check_bust(self, rolls: list) -> bool:
        """Check if the player has busted (no scoring combinations available)"""
        
        return self.score(rolls, allow_extra=True) == 0
