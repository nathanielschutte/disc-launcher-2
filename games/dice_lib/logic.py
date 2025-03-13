class DieLogic:
    def __init__(self, sides: int = 6):
        self.sides = sides


    def score(self, selected: list, allow_extra=False) -> int:
        """Score selection of roll"""

        score = 0

        if len(selected) == 0:
            return 0
        
        selected.sort()

        # check for triples
        for die in set(selected):
            if selected.count(die) >= 3:
                score += die * 100
        
        # check for 1s and 5s
        for die in selected:
            if die == 1:
                score += 100
            elif die == 5:
                score += 50

        return score


    def check_bust(self, rolls: list) -> bool:
        """Check if the player has busted"""
        
        if self.score(rolls, allow_extra=True) == 0:
            return True
        
        return False
