import asyncio
import json
from abc import ABC, abstractmethod
from games.display.screen import Screen


class GameStartupException(Exception):
    """Exception raised when a game fails to start"""

    pass


class BaseGame(ABC):
    """Base class for all games"""
    
    def __init__(self, manager, players, *args):
        self.manager = manager
        self.requested_players = players
        self.players = []
        self.current_player_index = 0
        self.current_player = None
        self.is_active = False
        self.is_waiting_for_input = False
        self.required_players = 1
        self.max_players = 1
        self.currency_manager = None
        # discord Message with screen
        self.message = None
        self.state = {}
        self.screen = Screen(80, 24)
        self.uses_currency = False

        self.ai_players = []
        self.ai_thinking = False
        self.ai_action_delay = 1.5


    @property
    def current_player(self):
        """Get the current player"""

        return self._current_player
    

    @current_player.setter
    def current_player(self, player):
        """Set the current player"""

        self._current_player = player


    def add_ai_player(self, name="CPU", avatar_url=None):
        """Add an AI player to the game"""
        
        ai_player = type('AIPlayer', (), {
            'id': -1000 - len(self.ai_players),
            'name': name,
            'display_name': name,
            'mention': f"@{name}",
            'avatar_url': avatar_url,
            'bot': False,
            'is_ai': True
        })
        
        self.ai_players.append(ai_player)

        return ai_player
    

    async def handle_ai_turn(self):
        """Default implementation for AI turn handling"""

        if not hasattr(self.current_player, 'is_ai') or not self.current_player.is_ai:
            return False
            
        await asyncio.sleep(self.ai_action_delay)
        self.end_turn()
        return True


    async def add_player(self, player):
        """Add a player to the game"""

        if not self.is_joinable():
            return False

        if player not in self.players:
            self.players.append(player)

        return True
    

    def next_turn(self):
        """Advance to next player's turn"""

        if not self.players:
            return
        
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.current_player = self.players[self.current_player_index]
    

    async def start_game(self, ctx):
        """Start the game"""

        self.is_active = True
        await self.start_turn(ctx)
    

    async def start_turn(self, ctx):
        """Start a player's turn"""

        self.is_waiting_for_input = True
        await ctx.send(f"{self.current_player.mention} It's your turn!")
    

    async def end_turn(self, ctx):
        """End the current turn"""

        self.is_waiting_for_input = False
        self.next_turn()
        await self.start_turn(ctx)
    

    async def update_display(self):
        """Update the game display"""

        if self.message:
            display = self.render_display()
            await self.message.edit(content=f"```\n{display}\n```")


    async def render_and_send_display(self, ctx):
        display = self.render_display()
        return await ctx.send(f"```\n{display}\n```")


    def serialize_state(self):
        """Serialize the game state"""
        
        return json.dumps(self.state, indent=2)
            

    def prepare_screen(self):
        """Clear and prepare the screen for rendering"""

        self.screen.clear()
        return self.screen
    

    def is_joinable(self):
        """Check if the game can be joined"""
        
        return self.is_active and len(self.players) < self.max_players
    

    @abstractmethod
    def render_display(self):
        """Render the game display as a string (to be shown in monospace)"""

        pass
    

    @abstractmethod
    async def process_command(self, message):
        """Process a command from the current player"""

        pass


    @abstractmethod
    async def process_reaction(self, message):
        """Process a reaction from the current player"""
        
        pass
