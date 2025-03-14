import random
import asyncio
from games.game_framework import BaseGame, GameStartupException
from games.display.components import Dice, Box, Button
from games.display.screen_utils import replace_emojis

from games.dice_lib.die import get_die
from games.dice_lib.logic import DieLogic
from games.dice_lib.dice_ai_v1 import DiceAI


DICE_MODES = [
    'beggars',
    'waggoners',
    'lords',
    'kings'
]


BUST_DELAY = 4
THINKING_DELAY = 3


class Game(BaseGame):
    """Dice game implementation based on Kingdom Come: Deliverance II"""
    
    def __init__(self, manager, players, *args):
        super().__init__(manager, players)

        self.logic = DieLogic()

        game_mode = 'beggars'

        self.state = {
            'dice_mode': game_mode,
            'bet': 10,
            'goal': 100,
            'scores': {},
            'dice': [],
            'dice_remaining': 6,
            'selected_dice': [],
            'combo_value': 0,
            'combo_trail': [],
            'selected_combo_value': 0,
            'phase': 'setup', # setup, play, end
            'winner': None,
            'turn_message': "",
            'player_balances': {},
            'player_die': {},
            'player_die_counter': 0,
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
        elif game_mode == 'waggoners':
            self.state['bet'] = 10
            self.state['goal'] = 2000
        elif game_mode == 'lords':
            self.state['bet'] = 30
            self.state['goal'] = 3000
        elif game_mode == 'kings':
            self.state['bet'] = 100
            self.state['goal'] = 4000

        self.required_players = 2
        self.max_players = 3
        self.uses_currency = True
        self.currency_manager = None
        self.last_activity = 0

        self.ai_handlers = {}
        self.ai_action_delay = THINKING_DELAY
    

    async def start_game(self, ctx):
        """Start the Dice game"""

        if self.message:
            await self.message.add_reaction("🎲")
            await self.message.add_reaction("💰")

        self.is_active = True

        for player in self.requested_players:
            res, error_msg = await self.add_player(player)
            if not res:
                raise GameStartupException(error_msg)
            
            print(f'Requested player {player.display_name} joined game')
    

    def is_joinable(self):
        """Check if the game can be joined"""

        return self.is_active and len(self.players) < self.required_players and (self.state['phase'] == 'setup' or self.state['waiting_for_players'])
    

    async def add_player(self, player) -> bool:
        bet = self.state['bet']

        if self.currency_manager:
            balance = await self.currency_manager.get_balance(player.id)
            error_msg = None
            
            if balance < bet:
                return False, f"{player.display_name} doesn't have enough funds for \"{self.state['dice_mode']}\"! (has {self.currency_manager.amount_string(balance)}, needs {self.currency_manager.amount_string(bet)})"
            
            if not error_msg:
                success, new_balance = await self.currency_manager.remove_funds(
                    player.id, 
                    bet, 
                    "dice",
                    f"Bet placed in Dice game"
                )
                print(f'Player {player.id} joined and bet {bet} and has {new_balance} left')
                    
                if success:
                    self.state['player_balances'][player.id] = new_balance
                else:
                    return False, f"Failed to place bet for {player.display_name}: {self.currency_manager.amount_string(bet)} ({self.currency_manager.amount_string(new_balance)})"
        
        self.state['scores'][player.id] = 0

        # this could be customized, pulled from player inventory
        self.state['player_die'][player.id] = [get_die('std')] * 5 + [get_die('luck')] * 1

        self.players.append(player)
        if len(self.players) == 1:
            self.current_player = player

        self.state['waiting_for_players'] = len(self.players) < self.required_players

        if not self.state['waiting_for_players']:
            self.state['phase'] = 'play'
            self.state['turn_message'] = "Ready to play!"

            await self.update_display()
            await self.message.clear_reactions()
            await self.start_turn(self.message.channel)
        else:
            self.state['turn_message'] = f'Waiting for another player to join "{self.state["dice_mode"]} ..."'
            await self.update_display()
            await self.message.add_reaction("💻")

        return True, None
    

    async def add_player_via_reaction(self, player):
        """Add a player who joined via reaction"""

        if not self.is_joinable() or player in self.players:
            print(f'Player {player.display_name} tried to join but game is not joinable or player is already in the game')
            return True
        
        if not await self.add_player(player):
            print(f'Player {player.display_name} tried to join but failed to be added to game')
            return False
            
        return True
    

    def render_display(self):
        """Render the Dice game board using Screen class and components"""

        screen = self.prepare_screen()
        screen.use_emoji_replacement = True
        
        Box.draw_box(screen, 1, 1, 78, 22, style="double", title=f"DICE ({self.state['dice_mode']})")

        # HEADER
        screen.draw_text(4, 3, f"BET: {self.currency_manager.amount_string(self.state['bet'], short=True)}  Goal: {self.state['goal']}")
        
        # PLAYERS
        screen.draw_text(4, 5, "PLAYERS:")
        for i, player in enumerate(self.players):
            player_text = f"P{i+1}: {player.display_name}"
            player_detail = '     '
            
            if player.id in self.state['scores']:
                player_detail += f"S: {self.state['scores'][player.id]}"
            
            # if player.id in self.state['player_balances']:
            #     player_detail += f" W: {self.currency_manager.amount_string(self.state['player_balances'][player.id], short=True)}"

            if player.id in self.state['player_die']:
                die_types = {}
                for die in self.state['player_die'][player.id]:
                    die_type = str(die)
                    die_types[die_type] = die_types.get(die_type, 0) + 1
                
                die_text = " | ".join([f"{die_type} x {count}" for die_type, count in die_types.items()])
                player_detail += f" D: [{die_text}]"

            if player == self.current_player and self.state['phase'] == 'play':
                player_text = "-> " + player_text
            else:
                player_text = "   " + player_text
                
            screen.draw_text(4, 5 + i*2, player_text)
            screen.draw_text(4, 6 + i*2, player_detail)
        
        if self.state['waiting_for_players']:
            Box.draw_box(screen, 20, 10, 40, 6, style="single", title="Waiting for Players")
            screen.draw_text(22, 12, "React with 🎲 to join the game!")
            screen.draw_text(22, 13, "React with 💻 to play against CPU!")
            screen.draw_text(3, 22, " 🎲 Join | 💰 Show Balance | 💻 vs. CPU | !end to quit ")
        
        elif self.state['phase'] == 'setup':
            Box.draw_box(screen, 20, 10, 40, 5, style="rounded", title="Setup Phase")
            screen.draw_text(22, 12, self.state['turn_message'])
            screen.draw_text(3, 22, " 🎲 Join | 💰 Show Balance | 💻 vs. CPU | !end to quit ")
        
        elif self.state['phase'] == 'play':
            player_name = self.current_player.display_name if self.current_player else "Unknown"
            # screen.draw_text(34, 3, f">>> {player_name}'s turn <<<")
            
            if self.state['dice']:
                Box.draw_box(screen, 46, 3, 29, 14, style="rounded", title="Current Dice")
                
                for i, die_value in enumerate(self.state['dice']):
                    die_x = 49 + ((i % 3) * 8)
                    die_y = 4 + ((i // 3) * 6)
                    
                    is_selected = (i in self.state['selected_dice'])
                    
                    Dice.draw_die(
                        screen, 
                        die_x, 
                        die_y, 
                        die_value, 
                        style=self.state['dice_style'], 
                        selected=is_selected
                    )
                    
                    screen.draw_text(die_x + 1, die_y + 0, str(i + 1))
            
            if self.state['combo_value']:
                combo_box_y = 12
                Box.draw_box(screen, 4, combo_box_y, screen.width // 2 - 4, 4, style="rounded", title="Turn Score")
                score_text = f"Score: {self.state['combo_value']}"
                if self.state['combo_trail']:
                    score_text += f" ({' + '.join(self.state['combo_trail'])})"
                screen.draw_text(6, combo_box_y + 1, score_text)
                screen.draw_text(6, combo_box_y + 2, f"Selection: {self.state['selected_combo_value']}")
            
            instruction_y = 17
            Box.draw_box(screen, 4, instruction_y, screen.width - 8, 5, style="rounded", title="Commands")
            # screen.draw_text(6, instruction_y + 2, "Select: ex. '1,3,5'")
            # screen.draw_text(6, instruction_y + 3, "Continue: 'continue'")
            # screen.draw_text(6, instruction_y + 4, "Pass: 'pass'")

            button_y = 19
            button_x = 10
            Button.draw_button(screen, button_x, button_y, "select (ex. 1,3,5)", width=24)
            Button.draw_button(screen, button_x + 26, button_y, "> continue", width=12)
            Button.draw_button(screen, button_x + 26 + 16, button_y, "> pass", width=12)
            
            if self.state['turn_message']:
                screen.draw_text(6, instruction_y + 1, self.state['turn_message'])
            
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
            
        # actions on selection
        if command in ['continue', 'c']:
            print(f'Player {self.current_player.display_name} opted to continue')
            if not self.state['combo_value']:
                print('No combo selected')
                error_msg = await message.channel.send("You need to select a valid combo before rolling again!")
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass
                return
                
            # self.state['scores'][self.current_player.id] += self.state['combo_value']
            
            # if self.state['combo_value'] >= self.state['goal']:
            #     print(f'Player {self.current_player.id} has reached the goal')

            #     self.state['winner'] = self.current_player.id
            #     self.state['phase'] = 'end'
            #     self.is_active = False
            #     self.is_waiting_for_input = False

            #     await self.award_winner(self.current_player.id)
            #     await self.update_display()
                
            #     win_msg = await message.channel.send(f"{self.current_player.display_name} has reached the goal and won!")
            #     await asyncio.sleep(5)
                
            #     try:
            #         await win_msg.delete()
            #     except:
            #         pass

            #     return

            print(f'Player {self.current_player.display_name} is re-rolling...')

            # when dice run out and we still can re-roll, we get a new batch
            if self.state['dice_remaining'] == 0:
                self.state['dice_remaining'] = 6
                self.state['dice'] = []
            
            if not await self._roll_dice():
                self.is_waiting_for_input = False

                self.state['turn_message'] = "BUST! You lose all score for this turn."
                await self.update_display()
                await asyncio.sleep(BUST_DELAY)

                # turn_msg = await message.channel.send(f"{self.current_player.mention} BUST! You lose all score for this turn.")

                # await asyncio.sleep(4)
                # try:
                #     await turn_msg.delete()
                # except:
                #     pass

                await self.end_turn(message.channel)
                return
            
            self.state['turn_message'] = "New roll! Select dice."
            await self.update_display()
            
        elif command in ['pass', 'p']:
            print(f'Player {self.current_player.display_name} opted to pass')
            if not self.state['combo_value']:
                print('No combo selected')
                error_msg = await message.channel.send("You need to select a valid combo before passing!")
                await asyncio.sleep(3)
                try:
                    await error_msg.delete()
                except:
                    pass
                return
                
            self.state['scores'][self.current_player.id] += self.state['combo_value']

            if self.state['scores'][self.current_player.id] >= self.state['goal']:
                print(f'Player {self.current_player.display_name} has reached the goal')
                self.state['winner'] = self.current_player.id
                self.state['phase'] = 'end'
                self.is_active = False
                self.is_waiting_for_input = False

                await self.award_winner(self.current_player.id)
                await self.update_display()

                return
            
            print(f'Player {self.current_player.display_name} passes...')
            
            self.state['turn_message'] = f"{self.current_player.display_name} your turn! Select dice to form a combo."
            
            await self.update_display()
            await self.end_turn(message.channel)

        # selection
        else:
            try:
                selections = [int(x.strip()) - 1 for x in command.split(',')]
                
                if not all(0 <= s < len(self.state['dice']) for s in selections):
                    # error_msg = await message.channel.send("Invalid dice selection! Numbers must be between 1 and 6.")
                    # await asyncio.sleep(3)
                    # try:
                    #     await error_msg.delete()
                    # except:
                    #     pass
                    # return

                    self.state['turn_message'] = "Invalid dice selection! Numbers must be 1 - 6, comma-separated."
                    await self.update_display()

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
                
                selected = []
                for i in selections:
                    selected.append(self.state['dice'][i])
                
                value = self.logic.score(selected)
                
                # make sure to reset any previous selection
                if self.state['selected_combo_value']:
                    self.state['combo_value'] -= self.state['selected_combo_value']
                    self.state['combo_trail'].pop()
                    self.state['dice_remaining'] = len(self.state['dice'])
                    self.state['selected_combo_value'] = 0

                if value:
                    self.state['combo_value'] += value
                    self.state['combo_trail'].append(f"{value}")
                    self.state['selected_combo_value'] = value
                    self.state['turn_message'] = f"Selection worth {value}. Type 'continue', or 'pass' to end turn"
                    self.state['dice_remaining'] -= len(selections)
                
                    print(f'Player {self.current_player.display_name} selected worth: {value}, {self.state["dice_remaining"]} dice remaining')
                else:
                    self.state['turn_message'] = "Not a valid combo. Try a different selection."
                    self.state['selected_dice'] = []

                    print(f'Player {self.current_player.display_name} selected invalid combo')
                
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
        
        self.state['dice_remaining'] = 6
        self.state['player_die_counter'] = 0
        self.state['dice'] = []
        self.state['combo_value'] = 0
        self.state['combo_trail'] = []
        
        if hasattr(self.current_player, 'is_ai') and self.current_player.is_ai:
            await self.handle_ai_turn(ctx)
        else:
            if not await self._roll_dice():
                self.is_waiting_for_input = False
                turn_msg = await ctx.send(f"{self.current_player.mention} BUST! You lose all score for this turn.")

                await asyncio.sleep(4)
                try:
                    await turn_msg.delete()
                except:
                    pass

                await self.end_turn(ctx)
                return
            
            self.is_waiting_for_input = True
            # turn_msg = await ctx.send(f"{self.current_player.mention} It's your turn!")
            self.state['turn_message'] = f"{self.current_player.display_name} your turn! Select dice to form a combo."

            await self.update_display()
            
            # await asyncio.sleep(3)
            # try:
            #     await turn_msg.delete()
            # except:
            #     pass


    async def handle_ai_turn(self, ctx):
        """Handle an AI player's turn"""

        self.is_waiting_for_input = False
        
        ai_handler = self.ai_handlers.get(self.current_player.id)
        if not ai_handler:
            print(f"No AI handler found for {self.current_player.display_name}")
            await self.end_turn(ctx)
            return
        
        turn_msg = None
        # turn_msg = await ctx.send(f"{self.current_player.display_name} is thinking...")
        self.state['turn_message'] = f"{self.current_player.display_name}'s turn, thinking..."
        await self.update_display()
        
        # going to think before first roll
        await asyncio.sleep(self.ai_action_delay // 2)
        if not await self._roll_dice():

            # bust_msg = await ctx.send(f"{self.current_player.display_name} BUST! Losing all points for this turn.")
            # await asyncio.sleep(3)
            # try:
            #     await bust_msg.delete()
            # except:
            #     pass

            self.state['turn_message'] = f"{self.current_player.display_name} BUST! Losing all points for this turn."
            await self.update_display()
            await asyncio.sleep(BUST_DELAY)
            
            await self.end_turn(ctx)
            return
        
        await self.update_display()
        
        turn_active = True
        while turn_active:
            await asyncio.sleep(self.ai_action_delay)
            
            opponent_id = next((p.id for p in self.players if p.id != self.current_player.id), None)
            opponent_score = self.state['scores'].get(opponent_id, 0) if opponent_id else 0
            
            decision, selected_dice = ai_handler.make_turn_decision(
                self.state['dice'],
                self.state['scores'][self.current_player.id],
                opponent_score,
                self.state['combo_value'],
                self.state['goal'],
                self.state['dice_remaining']
            )
            
            if decision == 'select' or decision == 'continue':
                self.state['selected_dice'] = selected_dice
                selected_values = [self.state['dice'][i] for i in selected_dice]
                selection_value = self.logic.score(selected_values)
                
                # selection_msg = await ctx.send(
                #     f"{self.current_player.display_name} selects dice: {', '.join(str(i+1) for i in selected_dice)} "
                #     f"(worth {selection_value} points)"
                # )

                self.state['turn_message'] = f"{self.current_player.display_name} selects dice: {', '.join(str(i+1) for i in selected_dice)}"
                self.state['combo_value'] += selection_value
                self.state['combo_trail'].append(f"{selection_value}")
                self.state['selected_combo_value'] = selection_value
                self.state['dice_remaining'] -= len(selected_dice)
                
                await self.update_display()

                await asyncio.sleep(self.ai_action_delay)

                # await asyncio.sleep(1.5)
                # try:
                #     await selection_msg.delete()
                # except:
                #     pass
                
                # continue!
                if decision == 'continue':
                    print(f'AI: continue rolling')

                    # continue_msg = await ctx.send(f"{self.current_player.display_name} decides to continue rolling!")
                    # await asyncio.sleep(1)
                    # try:
                    #     await continue_msg.delete()
                    # except:
                    #     pass

                    self.state['turn_message'] = f"{self.current_player.display_name} will continue!"
                    await self.update_display()
                    
                    if not await self._roll_dice():
                        print('AI: bust!')

                        # bust_msg = await ctx.send(f"{self.current_player.display_name} BUST! Losing all points for this turn.")
                        # await asyncio.sleep(3)
                        # try:
                        #     await bust_msg.delete()
                        # except:
                        #     pass

                        self.state['turn_message'] = f"{self.current_player.display_name} BUST! Losing all points for this turn."
                        await self.update_display()

                        await asyncio.sleep(BUST_DELAY)
                        
                        turn_active = False
                    
                    await self.update_display()
                    
                # just selected, nothing else
                elif decision == 'select':
                    print(f'AI: selected dice')

                    self.state['turn_message'] = f"{self.current_player.display_name} is deciding to continue or pass..."
                    await self.update_display()
                
            elif decision == 'pass':
                print(f'AI: pass')

                self.state['scores'][self.current_player.id] += self.state['combo_value']
                
                # pass_msg = await ctx.send(
                #     f"{self.current_player.display_name} passes with {self.state['combo_value']} points! "
                #     f"Total score: {self.state['scores'][self.current_player.id]}"
                # )

                self.state['turn_message'] = f"{self.current_player.display_name} passes with {self.state['combo_value']} points!"
                
                # CPU win?
                if self.state['scores'][self.current_player.id] >= self.state['goal']:
                    self.state['winner'] = self.current_player.id
                    self.state['phase'] = 'end'
                    self.is_active = False
                    
                    await self.award_winner(self.current_player.id)
                    await self.update_display()
                    
                    return
                
                await self.update_display()
                await asyncio.sleep(self.ai_action_delay)
                
                # await asyncio.sleep(2)
                # try:
                #     await pass_msg.delete()
                # except:
                #     pass
                
                turn_active = False
        
        if turn_msg:
            try:
                await turn_msg.delete()
            except:
                pass
        
        await self.end_turn(ctx)
    

    async def award_winner(self, winner_id):
        """Award the winner with the bet amount"""

        if not self.currency_manager:
            return
        
        pot = self.state['bet'] * len(self.players)

        # only award human players
        if winner_id >= 0:
            await self.currency_manager.add_item(winner_id, f"Dice Medal: {self.state['dice_mode'].capitalize()}", quantity=1, item_type="trophy")
            
            success, new_balance = await self.currency_manager.add_funds(
                winner_id,
                pot,
                "dice",
                f"Won Dice game (pot: ${pot})"
            )
        
            if success:
                self.state['player_balances'][winner_id] = new_balance
            else:
                print(f'Failed to award winner {winner_id} with {pot}')
                await self.message.channel.send(f"Failed to award winner {self.current_player.display_name} with {self.currency_manager.amount_string(pot)}")
                return

            print(f'Player {winner_id} won {pot} and now has {new_balance}')
        else:
            print(f'Player {winner_id} won {pot}!')

        await self.message.channel.send(f"{self.current_player.display_name} won! Earned {self.currency_manager.amount_string(pot)} and a trophy: \"Dice Medal: {self.state['dice_mode'].capitalize()}\"")

        self.manager.remove_game(self.message.channel.id)


    async def cleanup_currency(self):
        """Handle any cleanup related to currency when game ends abruptly"""

        if self.state['winner']:
            return
            
        if self.currency_manager:
            for player in self.players[:self.max_players]:
                if player.id < 0:
                    continue
                print(f'Returning bet to player {player.id}')
                await self.currency_manager.add_funds(
                    player.id,
                    self.state['bet'],
                    "dice",
                    "Refund from interrupted Dice game"
                )

    
    async def handle_add_ai_player(self):
        ai_player = self.add_ai_player("CPU")

        # risky
        risk_level = 0.6 + (random.random() * 0.4)

        if self.state['dice_mode'] == 'beggars':
            risk_level = 0.6 + (random.random() * 0.4)
        elif self.state['dice_mode'] == 'waggoners':
            risk_level = 0.5 + (random.random() * 0.4)
        elif self.state['dice_mode'] == 'lords':
            risk_level = 0.5 + (random.random() * 0.4)
        elif self.state['dice_mode'] == 'kings':
            risk_level = 0.5 + (random.random() * 0.3)

        # chill
        #risk_level = 0.4 + (random.random() * 0.2)

        ai_brain = DiceAI(risk_level=risk_level)
        ai_brain.set_logic(self.logic)

        print(f'Added AI player {ai_player.display_name} with risk level {risk_level}')

        self.ai_handlers[ai_player.id] = ai_brain

        self.state['player_die'][ai_player.id] = [get_die('std')] * 6
        # self.state['player_die'][ai_player.id] = [get_die('std')] * 4 + [get_die('luck')] * 1 + [get_die('3')] * 1
        self.state['scores'][ai_player.id] = 0

        self.players.append(ai_player)
        self.state['waiting_for_players'] = len(self.players) < self.required_players
        
        # start game
        self.state['phase'] = 'play'
        self.state['turn_message'] = "Ready to play!"

        await self.update_display()
        await self.message.clear_reactions()
        await self.start_turn(self.message.channel)
                

    async def process_reaction(self, reaction, user):
        """Process reactions to the game message"""
        if user.bot:
            return
            
        emoji = str(reaction.emoji)
        msg = None
        
        if emoji == "🎲":
            if self.is_joinable():
                success = await self.add_player_via_reaction(user)
                await reaction.remove(user)
                if not success:
                    msg = await reaction.message.channel.send(f"{user.display_name} failed to join the game")
                    
        elif emoji == "💰":
            if self.currency_manager:
                balance = await self.currency_manager.get_balance(user.id)

                channel = reaction.message.channel
                msg = await channel.send(f"{user.mention}, your current balance is ${balance}")
                
            await reaction.remove(user)
            
        elif emoji == '💻':
            print(f'Player {user.display_name} wants to play against CPU')
            await reaction.remove(user)
            await self.handle_add_ai_player()

        if msg:
            await asyncio.sleep(4)
            try:
                await msg.delete()
            except:
                pass


    async def _roll_dice(self) -> None:
        self.state['selected_dice'] = []
        self.state['selected_combo_value'] = 0

        queued_die = []
        for i in range(self.state['dice_remaining']):
            queued_die.append(self.state['player_die'][self.current_player.id][self.state['player_die_counter']])
            self.state['player_die_counter'] += 1
            if self.state['player_die_counter'] >= 6:
                self.state['player_die_counter'] = 0

        self.state['dice'] = [die.roll() for die in queued_die]

        print(f'Die roll: {(self.state["dice"])}')

        if self.logic.check_bust(self.state['dice']):
            self.state['combo_trail'].append('BUST')
            self.state['turn_message'] = "Bust!"
            
            await self.update_display()
            return False

        return True
    
