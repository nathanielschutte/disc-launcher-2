import os
import json
import importlib
import discord
from discord.ext import commands
from dotenv import load_dotenv
import traceback

from manager.currency_manager import CurrencyManager

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

currency_manager = CurrencyManager()
currency_manager.starting_balance = 100


class GameManager:
    def __init__(self):
        self.games = {}
        self.active_games = {}
        

    def load_games(self, config_path='games_config.json'):
        """Load game modules based on config file"""

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for game_info in config.get('games', []):
                if game_info.get('enabled', True) is False:
                    continue

                game_name = game_info.get('name')
                module_path = game_info.get('module')
                
                try:
                    module = importlib.import_module(module_path)
                    self.games[game_name.lower()] = module.Game
                    print(f"Loaded game: {game_name}")
                except (ImportError, AttributeError) as e:
                    print(f"Failed to load game {game_name}: {e}")
        
        except Exception as e:
            print(f"Error loading games config: {e}")
    

    async def start_game(self, game_type, channel_id, players, *args, currency_required=False):
        """Start a new game instance"""


        if game_type.lower() not in self.games:
            return False, f"Game '{game_type}' not found"
        
        if channel_id in self.active_games:
            return False, "A game is already in progress in this channel"
        
        game_class = self.games[game_type.lower()]
        
        try:
            game = game_class(players, *args)
        except Exception as e:
            return False, f"Failed to instantiate game ({type(e).__name__}): {e}"
        
        if hasattr(game, 'uses_currency') and game.uses_currency:
            game.currency_manager = currency_manager

            # if currency_required:
            #     for player in players:
            #         await currency_manager.ensure_minimum_balance(player.id, 100)
        
        self.active_games[channel_id] = game
        
        return True, game
    

    async def end_game(self, channel_id):
        """End an active game"""
        if channel_id in self.active_games:
            
            game = self.active_games[channel_id]
            extra = ''
            if hasattr(game, 'cleanup_currency') and callable(game.cleanup_currency):
                await game.cleanup_currency()
                extra = ' and currency refunded'
                
            del self.active_games[channel_id]
            return True, f"Game ended{extra}"
        
        return False, "No active game in this channel"
    

    def get_game(self, channel_id):
        """Get the active game in a channel"""

        return self.active_games.get(channel_id)


game_manager = GameManager()


@bot.event
async def on_ready():
    """Bot startup event"""

    print(f"{bot.user.name} has connected to Discord!")
    game_manager.load_games()
    print(f"Available games: {', '.join(game_manager.games.keys())}")


@bot.command(name='balance', aliases=['bal'])
async def check_balance(ctx):
    """Check your current balance"""

    balance = await currency_manager.get_balance(ctx.author.id)
    await ctx.send(f"{ctx.author.mention}, your current balance is ${balance}")


@bot.command(name='daily')
async def daily_bonus(ctx):
    """Claim your daily currency bonus"""
    
    amount = 50
    success, result = await currency_manager.add_funds(
        ctx.author.id, 
        amount, 
        "daily", 
        "Daily bonus claim"
    )
    
    if success:
        await ctx.send(f"{ctx.author.mention}, you've claimed your daily bonus of ${amount}! Your balance is now ${result}")
    else:
        await ctx.send(f"Error: {result}")


@bot.command(name='give')
async def give_currency(ctx, recipient: discord.Member, amount: int):
    """Give some of your currency to another player"""
    if amount <= 0:
        await ctx.send("Amount must be positive")
        return
        
    if recipient.bot:
        await ctx.send("You can't give currency to bots")
        return
    
    if recipient.id == ctx.author.id:
        await ctx.send("You can't give currency to yourself")
        return
    
    success, result = await currency_manager.transfer_funds(
        ctx.author.id,
        recipient.id,
        amount,
        "transfer",
        f"Gift from {ctx.author.name} to {recipient.name}"
    )
    
    if success:
        await ctx.send(f"{ctx.author.mention} gave ${amount} to {recipient.mention}")
    else:
        await ctx.send(f"Error: {result}")


@bot.command(name='leaderboard', aliases=['lb', 'rich'])
async def currency_leaderboard(ctx):
    """Display the richest players on the server"""
    leaderboard = await currency_manager.get_leaderboard(10)
    
    if not leaderboard:
        await ctx.send("No players have currency yet")
        return
    
    embed = discord.Embed(
        title="💰 Currency Leaderboard 💰",
        description="The richest players on the server",
        color=discord.Color.gold()
    )
    
    for i, (user_id, balance) in enumerate(leaderboard):
        try:
            user = await bot.fetch_user(user_id)
            name = user.name
        except:
            name = f"User {user_id}"
        
        embed.add_field(
            name=f"{i+1}. {name}",
            value=f"${balance}",
            inline=False
        )
    
    await ctx.send(embed=embed)


@bot.command(name='trophies', aliases=['tr', 'medals'])
async def currency_trophies(ctx):
    """Display the richest players on the server"""
    tr = await currency_manager.get_trophies()
    
    if not tr:
        await ctx.send("No players have trophies yet")
        return
    
    embed = discord.Embed(
        title="🏆 Trophy Leaderboard 🏆",
        description="The most successful players on the server",
        color=discord.Color.dark_gold()
    )
    
    for i, (user_id, items) in enumerate(tr):
        try:
            user = await bot.fetch_user(user_id)
            name = user.name
        except:
            name = f"User {user_id}"
        
        string = " | ".join([f"{item['count']}x {name}" for name, item in items.items()])

        embed.add_field(
            name=f"{i+1}. {name}",
            value=string,
            inline=False
        )
    
    await ctx.send(embed=embed)


@bot.command(name='addcurrency')
@commands.has_permissions(administrator=True)
async def add_currency(ctx, recipient: discord.Member, amount: int):
    """[Admin] Add currency to a player's balance"""
    if amount <= 0:
        await ctx.send("Amount must be positive")
        return
    
    success, result = await currency_manager.add_funds(
        recipient.id,
        amount,
        "admin",
        f"Admin grant from {ctx.author.name}"
    )
    
    if success:
        await ctx.send(f"Added ${amount} to {recipient.mention}'s balance. New balance: ${result}")
    else:
        await ctx.send(f"Error: {result}")


@bot.command(name='setcurrency')
@commands.has_permissions(administrator=True)
async def set_currency(ctx, recipient: discord.Member, amount: int):
    """[Admin] Set a player's currency to a specific amount"""
    if amount < 0:
        await ctx.send("Amount cannot be negative")
        return
    
    success, result = await currency_manager.reset_balance(recipient.id, amount)
    
    if success:
        await ctx.send(f"Set {recipient.mention}'s balance to ${amount}")
    else:
        await ctx.send(f"Error: {result}")


@bot.command(name='games')
async def list_games(ctx):
    """List available games"""
    if not game_manager.games:
        await ctx.send("No games are currently available")
        return
    
    game_list = ", ".join(game_manager.games.keys())
    await ctx.send(f"Available games: {game_list}")


@bot.command(name='start')
async def start_game(ctx, game_type: str, *args):
    """Start a new game"""
    
    success, result = await game_manager.start_game(game_type, ctx.channel.id, [ctx.author], *args)
    
    if success:
        game = result
        await ctx.send(f"Starting a new game of {game_type}!")
        game.message = await game.render_and_send_display(ctx)

        try:
            await game.start_game(ctx)
        except Exception as e:
            await ctx.send(f"Failed to start game ({type(e).__name__}): {e}")
            print(f"Failed to start game: {e}")
            print(traceback.format_exc())

    else:
        await ctx.send(result)


@bot.command(name='join')
async def join_game(ctx):
    """Join an existing game"""

    game = game_manager.get_game(ctx.channel.id)
    
    if not game:
        await ctx.send("No game in progress. Start one with !start [game]")
        return
    
    if ctx.author in game.players:
        await ctx.send("You're already in this game!")
        return
    
    if game.is_joinable():
        game.add_player(ctx.author)
        msg = await ctx.send(f"{ctx.author.display_name} joined the game!")
        await game.update_display()
    else:
        await ctx.send("Game cannot be joined")


@bot.command(name='end')
async def end_game(ctx):
    """End the current game"""

    game = game_manager.get_game(ctx.channel.id)
    
    if not game:
        await ctx.send("No game in progress")
        return
    
    if ctx.author not in game.players and not ctx.author.guild_permissions.administrator:
        await ctx.send("You're not in this game or don't have permission to end it")
        return
    
    success, message = await game_manager.end_game(ctx.channel.id)
    await ctx.send(message)


@bot.event
async def on_message(message):
    """Process commands during game turns"""
    
    if message.author.bot:
        return
    
    game = game_manager.get_game(message.channel.id)
    
    if message.content.strip().lower() == "!end":
        success, res = await game_manager.end_game(message.channel.id)
        await message.channel.send(res)
        return
    elif game and game.is_waiting_for_input and game.current_player == message.author:
        await game.process_command(message)
        return
    elif game and game.is_active:
        if hasattr(game, 'process_command'):
            await game.process_command(message)
            return
    
    await bot.process_commands(message)


@bot.event
async def on_reaction_add(reaction, user):
    """Handle reactions to game messages"""
    
    if user.bot:
        return
        
    channel_id = reaction.message.channel.id
    game = game_manager.get_game(channel_id)
    
    if game and game.is_active and game.message and game.message.id == reaction.message.id:
        if hasattr(game, 'process_reaction'):
            await game.process_reaction(reaction, user)


def run():
    bot.run(TOKEN)


if __name__ == "__main__":
    run(TOKEN)
