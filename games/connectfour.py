from games.game_framework import BaseGame

class Game(BaseGame):
    """Connect Four game implementation"""
    
    def __init__(self, players):
        super().__init__(players)
        # Initialize a 7x6 board (columns x rows)
        self.state = {
            'board': [[' ' for _ in range(6)] for _ in range(7)],  # column-based
            'symbols': {},  # Player to symbol mapping
            'winner': None,
            'last_move': None
        }
        self.max_players = 2  # Connect Four needs exactly 2 players
        self.symbols = ['🔴', '🔵']  # Player symbols (using emoji or can use 'O' and 'X')
    
    async def start_game(self, ctx):
        """Start the Connect Four game"""
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
        """Render the Connect Four board using Screen class"""
        board = self.state['board']
        
        # Prepare screen for rendering
        screen = self.prepare_screen()
        
        # Draw title
        screen.draw_text(30, 1, "CONNECT FOUR")
        
        # Draw player info
        for i, player in enumerate(self.players[:2]):
            symbol = self.state['symbols'].get(player.id, self.symbols[i])
            screen.draw_text(5, 3 + i, f"Player {i+1}: {player.display_name} ({symbol})")
        
        # Draw board
        board_x, board_y = 20, 6
        
        # Draw column numbers
        for i in range(7):
            screen.draw_text(board_x + 2 + (i*4), board_y - 1, str(i+1))
        
        # Draw board outline
        screen.draw_rect(board_x, board_y, 30, 14)
        
        # Draw grid and pieces
        for col in range(7):
            for row in range(6):
                # Position for this cell
                cell_x = board_x + 2 + (col*4)
                cell_y = board_y + 12 - (row*2)  # Bottom up
                
                # Draw the piece (or empty space)
                screen.draw_text(cell_x, cell_y, board[col][row])
                
                # Draw cell boundary
                if row < 5:
                    screen.draw_text(cell_x, cell_y - 1, '-')
                if col < 6:
                    screen.draw_text(cell_x + 2, cell_y, '|')
        
        # Draw game status
        status_y = board_y + 16
        if self.state['winner']:
            winner = next((p for p in self.players if p.id == self.state['winner']), None)
            if winner:
                screen.draw_text(board_x, status_y, f"{winner.display_name} wins!")
        elif self.current_player:
            symbol = self.state['symbols'].get(self.current_player.id, '?')
            screen.draw_text(board_x, status_y, f"Current turn: {self.current_player.display_name} ({symbol})")
            screen.draw_text(board_x, status_y + 1, "Enter column number to drop your piece (1-7)")
        
        # Highlight last move if exists
        if self.state['last_move']:
            last_col, last_row = self.state['last_move']
            highlight_x = board_x + 2 + (last_col*4)
            highlight_y = board_y + 12 - (last_row*2)
            screen.draw_text(highlight_x - 1, highlight_y, '[')
            screen.draw_text(highlight_x + 1, highlight_y, ']')
        
        return screen.render()
    
    async def process_command(self, message):
        """Process a player's move"""
        if not self.is_waiting_for_input or message.author != self.current_player:
            return
        
        # Parse the move (column number)
        try:
            col = int(message.content.strip()) - 1  # Convert to 0-based index
            
            # Validate the column
            if not (0 <= col < 7):
                await message.channel.send("Invalid move! Column must be between 1 and 7.")
                return
            
            # Check if column is full
            if self.state['board'][col][5] != ' ':
                await message.channel.send("That column is full! Choose another.")
                return
            
            # Find the next available row in this column (bottom up)
            row = 0
            while row < 6 and self.state['board'][col][row] == ' ':
                row += 1
            row -= 1  # Go back to the last empty space
            
            # Make the move
            symbol = self.state['symbols'][message.author.id]
            self.state['board'][col][row] = symbol
            self.state['last_move'] = (col, row)
            
            # Check for win
            if self._check_win(col, row, symbol):
                self.state['winner'] = message.author.id
                self.is_active = False
                self.is_waiting_for_input = False
            elif self._check_draw():
                self.is_active = False
                self.is_waiting_for_input = False
            else:
                # Next player's turn
                self.next_turn()
            
            # Update the display
            await self.update_display()
            
            # If game is over, don't start a new turn
            if not self.is_active:
                await message.channel.send("Game over! Start a new one with !start connectfour")
                return
            
            # Start the next turn
            await self.start_turn(message.channel)
                
        except (ValueError, IndexError):
            await message.channel.send("Invalid format! Please enter a column number (1-7)")
    
    def _check_win(self, col, row, symbol):
        """Check if the last move resulted in a win"""
        board = self.state['board']
        directions = [
            [(0, 1), (0, -1)],  # Vertical
            [(1, 0), (-1, 0)],  # Horizontal
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        for dir_pair in directions:
            count = 1  # Start with the piece just placed
            
            # Check in both directions
            for dx, dy in dir_pair:
                c, r = col, row
                
                # Count consecutive pieces in this direction
                for _ in range(3):  # Need 3 more to make 4 in a row
                    c, r = c + dx, r + dy
                    if 0 <= c < 7 and 0 <= r < 6 and board[c][r] == symbol:
                        count += 1
                    else:
                        break
                
            if count >= 4:
                return True
        
        return False
    
    def _check_draw(self):
        """Check if the game is a draw (board is full)"""
        # Check if all top row cells are filled
        return all(self.state['board'][col][5] != ' ' for col in range(7))
