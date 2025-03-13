class DieLogic:
    def __init__(self, sides: int = 6):
        self.sides = sides


    def score(self, rolls: list, selections: list, allow_extra=False) -> int:
        """Score selection of roll"""

        score = 0

        selected = []
        for i in selections:
            selected.append(rolls[i])

        selected.sort()

        if len(selected) == 0:
            return 0
        
        # check for 1s and 5s
        for die in selected:
            if die == 1:
                score += 100
            elif die == 5:
                score += 50

        # check for triples
        for die in set(selected):
            if selected.count(die) >= 3:
                score += die * 100
                if die == 1:
                    score += 1000
                elif die == 5:
                    score += 500

        return score


    def check_bust(self, rolls: list) -> bool:
        """Check if the player has busted"""
        
        if self.score(rolls, range(len(rolls)), allow_extra=True) == 0:
            return True
        
        return False

    