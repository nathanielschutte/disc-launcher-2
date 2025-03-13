import random
import json
import os

with open(os.path.join(os.path.dirname(__file__), 'die.json'), 'r') as f:
    die_data = json.load(f)

class Die:
    def __init__(self, die_type='standard'):
        self.sides = 6
        self.die_type = die_type
        self.load = self._generate_load(die_type)


    def _generate_load(self, die_type):
        if die_type in die_data:
            return die_data[die_type]['load']
        else:
            raise ValueError("Unknown die type")


    def roll(self):
        return random.choices(range(1, self.sides + 1), weights=self.load)[0]
        

    def __str__(self) -> str:
        return self.die_type
    

_DIE = {}


def get_die(die_type):
    if die_type not in _DIE:
        _DIE[die_type] = Die(die_type)
    return _DIE[die_type]


if __name__ == "__main__":
    die = Die('standard')
    die.roll()
    print(die)
