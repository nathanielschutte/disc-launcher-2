from games.game_framework import BaseGame

class Game(BaseGame):
    """Tic Tac Toe game implementation"""
    
    def __init__(self, players):
        super().__init__(players)
        # Initialize an empty 3x3 board
        self.state = {
            'board': [[' ' for _ in range(3)] for _ in range(3)],
            'symbols': {}, # Player to symbol mapping
            'winner': None,
            'is_draw': False
        }
        self.max_players = 2  # Tic Tac Toe needs exactly 2 players
        self.symbols = ['X', 'O']  # Player symbols
    
    async def start_game(self, ctx):
        """Start the Tic Tac Toe game"""
        # Wait for second player if needed
        if len(self.players) < 2:
            await ctx.send("Waiting for another player to join with !join")
            self.is_active = True
            return
        
        # Assign symbols to players
        for i, player in enumerate(self.players[:2]):
            self.state['symbols'][player.id] = self.symbols[i]
        
        # Start the game
        await super().start_game(ctx)
    
    def is_joinable(self):
        """Check if the game can be joined"""
        return self.is_active and len(self.players) < 2
    
    def render_display(self):
        """Render the Tic Tac Toe board using Screen class"""
        board = self.state['board']
        
        # Prepare screen for rendering
        screen = self.prepare_screen()
        
        # Draw title
        screen.draw_text(30, 1, "TIC TAC TOE")
        
        # Draw player info
        for i, player in enumerate(self.players[:2]):
            symbol = self.state['symbols'].get(player.id, self.symbols[i])
            screen.draw_text(5, 3 + i, f"Player {i+1}: {player.display_name} ({symbol})")
        
        # Draw board
        board_x, board_y = 30, 6
        
        # Draw column numbers
        for i in range(3):
            screen.draw_text(board_x + 2 + (i*4), board_y - 1, str(i+1))
        
        # Draw row numbers and board
        for i in range(3):
            # Row number
            screen.draw_text(board_x - 2, board_y + 1 + (i*2), str(i+1))
            
            # Draw cells and content
            for j in range(3):
                screen.draw_text(board_x + 2 + (j*4), board_y + 1 + (i*2), board[i][j])
                
                # Draw vertical lines
                if j < 2:
                    screen.draw_line(
                        board_x + 4 + (j*4), board_y, 
                        board_x + 4 + (j*4), board_y + 5, 
                        '|'
                    )
            
            # Draw horizontal lines
            if i < 2:
                screen.draw_line(
                    board_x, board_y + 2 + (i*2),
                    board_x + 12, board_y + 2 + (i*2),
                    '-'
                )
        
        # Draw game status
        status_y = board_y + 8
        if self.state['winner']:
            winner = next((p for p in self.players if p.id == self.state['winner']), None)
            if winner:
                screen.draw_text(board_x - 10, status_y, f"{winner.display_name} wins!")
        elif self.state['is_draw']:
            screen.draw_text(board_x - 10, status_y, "Game ended in a draw!")
        elif self.current_player:
            symbol = self.state['symbols'].get(self.current_player.id, '?')
            screen.draw_text(board_x - 10, status_y, f"Current turn: {self.current_player.display_name} ({symbol})")
            screen.draw_text(board_x - 10, status_y + 1, "Enter row,column to place your mark (e.g. 1,3)")
        
        return screen.render()
    
    async def process_command(self, message):
        """Process a player's move"""
        if not self.is_waiting_for_input or message.author != self.current_player:
            return
        
        # Parse the move (row,col format)
        try:
            row_str, col_str = message.content.strip().split(',')
            row = int(row_str) - 1  # Convert to 0-based index
            col = int(col_str) - 1  # Convert to 0-based index
            
            # Validate the move
            if not (0 <= row < 3 and 0 <= col < 3):
                await message.channel.send("Invalid move! Row and column must be between 1 and 3.")
                return
            
            if self.state['board'][row][col] != ' ':
                await message.channel.send("That space is already taken!")
                return
            
            # Make the move
            symbol = self.state['symbols'][message.author.id]
            self.state['board'][row][col] = symbol
            
            # Check for win or draw
            if self._check_win(row, col, symbol):
                self.state['winner'] = message.author.id
                self.is_active = False
                self.is_waiting_for_input = False
            elif self._check_draw():
                self.state['is_draw'] = True
                self.is_active = False
                self.is_waiting_for_input = False
            else:
                # Next player's turn
                self.next_turn()
            
            # Update the display
            await self.update_display()
            
            # If game is over, don't start a new turn
            if not self.is_active:
                await message.channel.send("Game over! Start a new one with !start tictactoe")
                return
            
            # Start the next turn
            await self.start_turn(message.channel)
                
        except (ValueError, IndexError):
            await message.channel.send("Invalid format! Please use 'row,column' (e.g. 1,3)")
    
    def _check_win(self, row, col, symbol):
        """Check if the last move resulted in a win"""
        board = self.state['board']
        
        # Check row
        if all(board[row][c] == symbol for c in range(3)):
            return True
        
        # Check column
        if all(board[r][col] == symbol for r in range(3)):
            return True
        
        # Check diagonals
        if row == col and all(board[i][i] == symbol for i in range(3)):
            return True
        
        if row + col == 2 and all(board[i][2-i] == symbol for i in range(3)):
            return True
        
        return False
    
    def _check_draw(self):
        """Check if the game is a draw"""
        return all(cell != ' ' for row in self.state['board'] for cell in row)
