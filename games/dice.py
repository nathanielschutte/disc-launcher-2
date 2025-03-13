import random
import asyncio
from games.game_framework import BaseGame, GameStartupException
from games.display.components import Dice, Box, Button
from games.display.screen_utils import replace_emojis


DICE_MODES = [
    'beggars',
    'wagoners',
    'lords',
    'kings'
]


class Game(BaseGame):
    """Dice game implementation based on Kingdom Come: Deliverance II"""
    
    def __init__(self, players, *args):
        super().__init__(players)

        game_mode = 'beggars'

        self.state = {
            'dice_mode': game_mode,
            'bet': 10,
            'goal': 100,
            'scores': {},
            'dice': [],
            'selected_dice': [],
            'current_combo': None,
            'combo_value': 0,
            'phase': 'setup', # setup, play, end
            'winner': None,
            'turn_message': "",
            'player_balances': {},
            'waiting_for_players': True,
            'dice_style': "fancy"
        }

        if len(args) >= 1:
            game_mode = args[0].lower()
            if game_mode in DICE_MODES:
                self.state['dice_mode'] = game_mode
            else:
                raise GameStartupException(f"Invalid dice mode: {game_mode}")

        if game_mode == 'beggars':
            self.state['bet'] = 3
            self.state['goal'] = 1250
        elif game_mode == 'wagoners':
            self.state['bet'] = 10
            self.state['goal'] = 2000
        elif game_mode == 'lords':
            self.state['bet'] = 30
            self.state['goal'] = 3000
        elif game_mode == 'kings':
            self.state['bet'] = 100
            self.state['goal'] = 4000

        self.required_players = 1
        self.max_players = 2
        self.uses_currency = True
        self.currency_manager = None
        self.message_ids_to_delete = []
        self.last_activity = 0
    

    async def start_game(self, ctx):
        """Start the Dice game"""

        self.state['waiting_for_players'] = len(self.players) < self.required_players
        
        if self.message:
            await self.message.add_reaction("🎲")
            await self.message.add_reaction("💰")
            
        for player in self.players:
            self.state['scores'][player.id] = 0
            
            if self.currency_manager:
                balance = await self.currency_manager.get_balance(player.id)
                self.state['player_balances'][player.id] = balance
        
        if self.state['waiting_for_players']:
            self.is_active = True
            self.state['turn_message'] = f'Waiting for another player to join "{self.state["dice_mode"]}". React with 🎲 to join!'
            await self.update_display()
            return
        
        self.state['phase'] = 'setup'
        self.state['turn_message'] = f"{self.current_player.mention} Please enter bet amount and goal in format 'bet,goal' (e.g. '10,100')"
        self.state['turn_message'] = f"{self.current_player.mention} Please have another player join the game!"

        # # just skip to play stage, don't need a betting stage
        # self.state['phase'] = 'play'
        # #start_message = await message.channel.send(f"Game started! Initial bet: {bet}, goal: {goal}")
        # self.state['turn_message'] = f"Game started! Initial bet: {self.state['bet']}, goal: {self.state['goal']}"
        
        # await super().start_game(ctx)
    

    def is_joinable(self):
        """Check if the game can be joined"""

        return self.is_active and len(self.players) < self.required_players and (self.state['phase'] == 'setup' or self.state['waiting_for_players'])
    

    async def add_player_via_reaction(self, player):
        """Add a player who joined via reaction"""

        bet = self.state['bet']

        if not self.is_joinable() or player in self.players:
            return False
        
        if self.currency_manager:
            balance = await self.currency_manager.get_balance(player.id)
            error_msg = None
            
            if balance < bet:
                error_msg = await self.message.channel.send(
                    f"{player.display_name} doesn't have enough funds for \"{self.state['dice_mode']}\"! (has {self.currency_manager.amount_string(balance)}, needs {self.currency_manager.amount_string(bet)})"
                )
            
            if not error_msg:
                success, new_balance = await self.currency_manager.remove_funds(
                    player.id, 
                    bet, 
                    "dice",
                    f"Bet placed in Dice game"
                )
                    
                if success:
                    self.state['player_balances'][player.id] = new_balance
                else:
                    error_msg = await self.message.channel.send(
                        f"Failed to place bet for {player.display_name}: {self.currency_manager.amount_string(bet)} ({self.currency_manager.amount_string(new_balance)})"
                    )
        
            if error_msg:
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass

            self.state['player_balances'][player.id] = balance
            self.state['scores'][player.id] = 0

        self.add_player(player)
        
        if len(self.players) >= self.required_players and self.state['waiting_for_players']:
            self.state['waiting_for_players'] = False
            self.state['phase'] = 'play'
            # self.state['turn_message'] = f"{self.current_player.mention} Please enter bet amount and goal in format 'bet,goal' (e.g. '10,100')"
            self.state['turn_message'] = "Ready to play!"

            self._roll_dice()

            await self.update_display()
            await self.start_turn(self.message.channel)
            
        return True
    

    def render_display(self):
        """Render the Dice game board using Screen class and components"""

        screen = self.prepare_screen()
        screen.use_emoji_replacement = True
        
        Box.draw_box(screen, 1, 1, 78, 22, style="double", title="DICE GAME")
        
        screen.draw_text(4, 3, "PLAYERS:")
        for i, player in enumerate(self.players[:2]):
            player_text = f"Player {i+1}: {player.display_name}"
            
            if player.id in self.state['scores']:
                player_text += f" - Game: ${self.state['scores'][player.id]}"
            
            if player.id in self.state['player_balances']:
                player_text += f" - Wallet: ${self.state['player_balances'][player.id]}"
            
            if player == self.current_player and self.state['phase'] == 'play':
                player_text = "-> " + player_text
            else:
                player_text = "   " + player_text
                
            screen.draw_text(4, 5 + i, player_text)
        
        screen.draw_text(4, 8, f"Bet: ${self.state['bet']}  Goal: ${self.state['goal']}")
        
        if self.state['waiting_for_players']:
            Box.draw_box(screen, 20, 10, 40, 5, style="single", title="Waiting for Players")
            screen.draw_text(22, 12, "React with 🎲 to join the game!")
            screen.draw_text(3, 22, " 🎲 Join Game | 💰 Show Balance | !end to quit ")
            
        elif self.state['phase'] == 'setup':
            Box.draw_box(screen, 20, 10, 40, 5, style="rounded", title="Setup Phase")
            screen.draw_text(22, 12, self.state['turn_message'])
            screen.draw_text(3, 22, " 🎲 Join Game | 💰 Show Balance | !end to quit ")
            
        elif self.state['phase'] == 'play':
            player_name = self.current_player.display_name if self.current_player else "Unknown"
            Box.draw_box(screen, 30, 3, 45, 3, style="rounded", title=f"{player_name}'s Turn")
            
            if self.state['dice']:
                Box.draw_box(screen, 30, 6, 45, 8, style="rounded", title="Current Dice")
                
                for i, die_value in enumerate(self.state['dice']):
                    die_x = 33 + (i * 8)
                    die_y = 8
                    
                    is_selected = (i in self.state['selected_dice'])
                    
                    Dice.draw_die(
                        screen, 
                        die_x, 
                        die_y, 
                        die_value, 
                        style=self.state['dice_style'], 
                        selected=is_selected
                    )
                    
                    screen.draw_text(die_x + 3, die_y + 6, str(i + 1))
            
            if self.state['current_combo']:
                combo_box_y = 15
                Box.draw_box(screen, 30, combo_box_y, 45, 4, style="rounded", title="Current Combo")
                screen.draw_text(33, combo_box_y + 1, f"Combo: {self.state['current_combo']}")
                screen.draw_text(33, combo_box_y + 2, f"Value: ${self.state['combo_value']}")
            
            button_y = 20
            Button.draw_button(screen, 33, button_y, "Select", width=10)
            Button.draw_button(screen, 47, button_y, "Roll", width=10)
            Button.draw_button(screen, 61, button_y, "Pass", width=10)
            
            instruction_y = 12
            Box.draw_box(screen, 4, instruction_y, 20, 9, style="rounded", title="Commands")
            screen.draw_text(6, instruction_y + 2, "Select: '1,3,5'")
            screen.draw_text(6, instruction_y + 3, "Roll: 'roll'")
            screen.draw_text(6, instruction_y + 4, "Pass: 'pass'")
            
            if self.state['turn_message']:
                screen.draw_text(6, instruction_y + 6, self.state['turn_message'])
            
        elif self.state['phase'] == 'end':
            Box.draw_box(screen, 25, 10, 30, 8, style="rounded", title="Game Over")
            
            if self.state['winner']:
                winner = next((p for p in self.players if p.id == self.state['winner']), None)
                if winner:
                    screen.draw_text(30, 12, f"{winner.display_name} WINS!")
                    screen.draw_text(30, 14, f"Final Score: ${self.state['scores'][winner.id]}")
                    
                    if winner.id in self.state['player_balances']:
                        screen.draw_text(30, 15, f"New Balance: ${self.state['player_balances'][winner.id]}")
            
            screen.draw_text(28, 17, "Type !start dice to play again")
        
        return screen.render()
    

    async def process_command(self, message):
        """Process player commands based on game phase"""

        if not self.is_waiting_for_input or message.author != self.current_player:
            if self.is_active and message.channel.id == self.message.channel.id:
                try:
                    await message.delete()
                except:
                    pass
            return
        
        command = message.content.strip().lower()
        
        self.message_ids_to_delete.append(message.id)
        
        if self.state['phase'] == 'setup':
            await self._process_setup(message, command)
            
        elif self.state['phase'] == 'play':
            await self._process_play(message, command)
    
    
    async def _process_setup(self, message, command):
        """Process setup phase commands"""

        try:
            bet_str, goal_str = command.split(',')
            bet = int(bet_str.strip())
            goal = int(goal_str.strip())
            
            if bet <= 0 or goal <= 0:
                await message.channel.send("Bet and goal must be positive values!")
                return
                
            if goal <= bet:
                await message.channel.send("Goal must be greater than bet!")
                return
            
            if self.currency_manager:
                for player in self.players[:2]:
                    balance = await self.currency_manager.get_balance(player.id)
                    self.state['player_balances'][player.id] = balance
                    
                    if balance < bet:
                        await message.channel.send(f"{player.display_name} doesn't have enough funds for this bet! (has ${balance}, needs ${bet})")
                        return
                
                for player in self.players[:2]:
                    success, new_balance = await self.currency_manager.remove_funds(
                        player.id, 
                        bet, 
                        "dice",
                        f"Bet placed in Dice game"
                    )
                    
                    if success:
                        self.state['player_balances'][player.id] = new_balance
            
            self.state['bet'] = bet
            self.state['goal'] = goal
            self.state['phase'] = 'play'
            
            self._roll_dice()
            
            try:
                await message.delete()
            except:
                pass
            
            await self.update_display()
            start_message = await message.channel.send(f"Game started! Initial bet: ${bet}, goal: ${goal}")
            
            await asyncio.sleep(3)
            try:
                await start_message.delete()
            except:
                pass
                
            await self.start_turn(message.channel)
            
        except (ValueError, IndexError):
            error_msg = await message.channel.send("Invalid format! Please enter 'bet,goal' (e.g. '10,100')")

            await asyncio.sleep(3)
            try:
                await error_msg.delete()
                await message.delete()
            except:
                pass
    

    async def _process_play(self, message, command):
        """Process play phase commands"""
        
        try:
            await message.delete()
        except:
            pass
            
        if command in ['roll', 'reroll', 'r']:
            if not self.state['current_combo']:
                error_msg = await message.channel.send("You need to select a valid combo before rolling again!")
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass
                return
                
            self.state['scores'][self.current_player.id] += self.state['combo_value']
            
            if self.state['scores'][self.current_player.id] >= self.state['goal']:
                self.state['winner'] = self.current_player.id
                self.state['phase'] = 'end'
                self.is_active = False
                self.is_waiting_for_input = False

                await self.award_winner(self.current_player.id)
                
                await self.update_display()
                win_msg = await message.channel.send(f"{self.current_player.display_name} has reached the goal and won!")
                await asyncio.sleep(5)
                try:
                    await win_msg.delete()
                except:
                    pass
                return
            
            self._roll_dice()
            self.state['selected_dice'] = []
            self.state['current_combo'] = None
            self.state['combo_value'] = 0
            self.state['turn_message'] = "New roll! Select dice to form a combo."
            await self.update_display()
            
        elif command in ['pass', 'p']:
            if not self.state['current_combo']:
                error_msg = await message.channel.send("You need to select a valid combo before passing!")
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass
                return
                
            self.state['scores'][self.current_player.id] += self.state['combo_value']

            if self.state['scores'][self.current_player.id] >= self.state['goal']:
                self.state['winner'] = self.current_player.id
                self.state['phase'] = 'end'
                self.is_active = False
                self.is_waiting_for_input = False

                await self.award_winner(self.current_player.id)
                
                await self.update_display()
                win_msg = await message.channel.send(f"{self.current_player.display_name} has reached the goal and won!")
                await asyncio.sleep(5)
                try:
                    await win_msg.delete()
                except:
                    pass
                return
            
            self.next_turn()

            self._roll_dice()
            self.state['selected_dice'] = []
            self.state['current_combo'] = None
            self.state['combo_value'] = 0
            self.state['turn_message'] = "Your turn! Select dice to form a combo."
            
            await self.update_display()
            await self.start_turn(message.channel)

        else:
            try:

                selections = [int(x.strip()) - 1 for x in command.split(',')]
                
                if not all(0 <= s < len(self.state['dice']) for s in selections):
                    error_msg = await message.channel.send("Invalid dice selection! Numbers must be between 1 and 6.")
                    await asyncio.sleep(3)
                    try:
                        await error_msg.delete()
                    except:
                        pass
                    return
                
                if not selections:
                    error_msg = await message.channel.send("Please select at least one die.")
                    await asyncio.sleep(3)
                    try:
                        await error_msg.delete()
                    except:
                        pass
                    return
                    
                self.state['selected_dice'] = selections
                
                combo, value = self._check_combo(selections)
                
                if combo:
                    self.state['current_combo'] = combo
                    self.state['combo_value'] = value
                    self.state['turn_message'] = f"Selected {combo} worth ${value}. Type 'roll' to roll again or 'pass' to end turn."
                else:
                    self.state['current_combo'] = None
                    self.state['combo_value'] = 0
                    self.state['turn_message'] = "Not a valid combo. Try a different selection."
                
                await self.update_display()
                
            except ValueError:
                error_msg = await message.channel.send("Invalid format! Please enter comma-separated numbers (e.g. '1,3,5')")
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass
    

    async def start_turn(self, ctx):
        """Start a player's turn"""
        self.is_waiting_for_input = True
        
        turn_msg = await ctx.send(f"{self.current_player.mention} It's your turn!")
        
        await asyncio.sleep(3)
        try:
            await turn_msg.delete()
        except:
            pass
    

    def _roll_dice(self):
        """Roll 6 dice with random values 1-6"""

        self.state['dice'] = [random.randint(1, 6) for _ in range(6)]
    

    def _check_combo(self, selected_indices):
        """
        Check if selected dice form a valid combo
        Returns (combo_name, value) if valid, (None, 0) otherwise
        """

        selected_values = [self.state['dice'][i] for i in selected_indices]
        
        value_counts = {}
        for val in selected_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        
        # Combos
        
        # Straight (sequence of 5 or more consecutive values)
        if len(selected_values) >= 5:
            unique_vals = sorted(set(selected_values))
            if len(unique_vals) >= 5:
                is_straight = True
                for i in range(1, len(unique_vals)):
                    if unique_vals[i] != unique_vals[i-1] + 1:
                        is_straight = False
                        break
                
                if is_straight and len(unique_vals) >= 5:
                    return "Straight", len(unique_vals) * 10
        
        # Three or more of a kind
        for val, count in value_counts.items():
            if count >= 3:
                return f"{count} of a kind ({val}s)", count * val * 2
        
        # # 3. Two Pairs
        # pairs = [val for val, count in value_counts.items() if count >= 2]
        # if len(pairs) >= 2:
        #     return "Two Pairs", sum(pairs) * 2
        #
        # # 4. One Pair
        # if len(pairs) == 1:
        #     return f"Pair of {pairs[0]}s", pairs[0] * 2
        
        # No valid combo
        return None, 0
    

    async def award_winner(self, winner_id):
        """Award the winner with the bet amount"""

        if not self.currency_manager:
            return
        
        pot = self.state['bet'] * 2
        
        success, new_balance = await self.currency_manager.add_funds(
            winner_id,
            pot,
            "dice",
            f"Won Dice game (pot: ${pot})"
        )
        
        if success:
            self.state['player_balances'][winner_id] = new_balance
            

    async def cleanup_currency(self):
        """Handle any cleanup related to currency when game ends abruptly"""
        if self.state['phase'] == 'setup' or self.state['winner']:
            return
            
        if self.currency_manager and self.state['phase'] == 'play':
            for player in self.players[:2]:
                await self.currency_manager.add_funds(
                    player.id,
                    self.state['bet'],
                    "dice",
                    "Refund from interrupted Dice game"
                )
                

    async def process_reaction(self, reaction, user):
        """Process reactions to the game message"""
        if user.bot:
            return
            
        emoji = str(reaction.emoji)
        
        if emoji == "🎲":
            if self.is_joinable():
                success = await self.add_player_via_reaction(user)
                if success:
                    await reaction.remove(user)
                    
        elif emoji == "💰":
            if self.currency_manager:
                balance = await self.currency_manager.get_balance(user.id)

                channel = reaction.message.channel
                balance_msg = await channel.send(f"{user.mention}, your current balance is ${balance}")
                
                await asyncio.sleep(5)
                try:
                    await balance_msg.delete()
                except:
                    pass
                
            await reaction.remove(user)
