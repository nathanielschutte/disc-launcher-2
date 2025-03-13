from games.display.screen_utils import replace_emojis, get_string_display_width

class Screen:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.clear()
        self.use_emoji_replacement = True  # Option to replace emojis with ASCII alternatives
        self.double_width = {} # dict with coords to characters that are double width


    def clear(self):
        self.matrix = [[' ' for _ in range(self.width)] for _ in range(self.height)]


    def draw_text(self, x, y, text, replace_emoji=None):
        """
        Draw text on the screen, handling emojis appropriately
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            text (str): Text to draw
            replace_emoji (bool): Whether to replace emojis (defaults to class setting)
        """
        if replace_emoji is None:
            replace_emoji = self.use_emoji_replacement
            
        if replace_emoji:
            text = replace_emojis(text)
        
        current_x = x
        
        for char in text:
            if not (0 <= current_x < self.width and 0 <= y < self.height):
                break
                
            self.matrix[y][current_x] = char
            
            char_width = get_string_display_width(char)
            current_x += 1
            
            if char_width > 1 and current_x < self.width:
                self.matrix[y][current_x] = ' '
                current_x += 1


    def draw_text_centered(self, x, y, width, text, replace_emoji=None):
        """Draw text centered within a given width"""
        if replace_emoji is None:
            replace_emoji = self.use_emoji_replacement
            
        if replace_emoji:
            text = replace_emojis(text)
            
        text_width = get_string_display_width(text)
        start_x = x + max(0, (width - text_width) // 2)
        self.draw_text(start_x, y, text, replace_emoji=False)  # Already replaced if needed


    def draw_rect(self, x, y, width, height, char='#'):
        for i in range(width):
            if 0 <= x + i < self.width:
                if 0 <= y < self.height:
                    self.matrix[y][x + i] = char
                if 0 <= y + height - 1 < self.height:
                    self.matrix[y + height - 1][x + i] = char
        for i in range(height):
            if 0 <= y + i < self.height:
                if 0 <= x < self.width:
                    self.matrix[y + i][x] = char
                if 0 <= x + width - 1 < self.width:
                    self.matrix[y + i][x + width - 1] = char


    def draw_line(self, x1, y1, x2, y2, char='-'):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                self.matrix[y1][x1] = char
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def render(self):
        return '\n'.join(''.join(row) for row in self.matrix)
    