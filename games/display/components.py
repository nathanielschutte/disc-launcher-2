"""
Game component library for rendering common game elements
"""

class Dice:
    """Dice component with various rendering options"""
    
    @staticmethod
    def get_face(value, style="ascii"):
        """Get the visual representation of a die face"""
        if style == "ascii":
            # ASCII art dice faces
            faces = [
                ["+-----+", 
                 "|     |", 
                 "|  •  |", 
                 "|     |", 
                 "+-----+"],  # 1
                
                ["+-----+", 
                 "| •   |", 
                 "|     |", 
                 "|   • |", 
                 "+-----+"],  # 2
                
                ["+-----+", 
                 "| •   |", 
                 "|  •  |", 
                 "|   • |", 
                 "+-----+"],  # 3
                
                ["+-----+", 
                 "| • • |", 
                 "|     |", 
                 "| • • |", 
                 "+-----+"],  # 4
                
                ["+-----+", 
                 "| • • |", 
                 "|  •  |", 
                 "| • • |", 
                 "+-----+"],  # 5
                
                ["+-----+", 
                 "| • • |", 
                 "| • • |", 
                 "| • • |", 
                 "+-----+"]   # 6
            ]
            return faces[value - 1]
        
        elif style == "unicode":
            # Unicode dice characters
            faces = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]
            return faces[value - 1]
        
        elif style == "fancy":
            # Fancy bordered dice
            faces = [
                ["╭─────╮", 
                 "│     │", 
                 "│  •  │", 
                 "│     │", 
                 "╰─────╯"],  # 1
                
                ["╭─────╮", 
                 "│ •   │", 
                 "│     │", 
                 "│   • │", 
                 "╰─────╯"],  # 2
                
                ["╭─────╮", 
                 "│ •   │", 
                 "│  •  │", 
                 "│   • │", 
                 "╰─────╯"],  # 3
                
                ["╭─────╮", 
                 "│ • • │", 
                 "│     │", 
                 "│ • • │", 
                 "╰─────╯"],  # 4
                
                ["╭─────╮", 
                 "│ • • │", 
                 "│  •  │", 
                 "│ • • │", 
                 "╰─────╯"],  # 5
                
                ["╭─────╮", 
                 "│ • • │", 
                 "│ • • │", 
                 "│ • • │", 
                 "╰─────╯"]   # 6
            ]
            return faces[value - 1]
        
        # Default to simple representation
        return [f"[ {value} ]"]
    
    @staticmethod
    def draw_die(screen, x, y, value, style="ascii", selected=False, highlight_char="*"):
        """Draw a die on the screen"""
        
        face = Dice.get_face(value, style)
        
        for i, line in enumerate(face):
            if selected:
                if i == 0 or i == len(face) - 1:
                    if style == "ascii":
                        line = line.replace("+", highlight_char)
                    elif style == "fancy":
                        if i == 0:
                            line = line.replace("╭", "╓").replace("╮", "╖")
                        else:
                            line = line.replace("╰", "╙").replace("╯", "╜")

            screen.draw_text(x, y + i, line)
        
        return len(face)


class Card:
    """Playing card component with various rendering options"""
    
    # Card suits
    SUITS = {
        'spades': '♠',
        'hearts': '♥',
        'diamonds': '♦',
        'clubs': '♣'
    }
    
    # Card ranks
    RANKS = {
        1: 'A',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: '10',
        11: 'J',
        12: 'Q',
        13: 'K'
    }
    
    @staticmethod
    def get_card(rank, suit, style="ascii", face_up=True):
        """Get the visual representation of a playing card"""
        if not face_up:
            if style == "ascii":
                return ["+-----+",
                        "|/////|",
                        "|/////|",
                        "|/////|",
                        "+-----+"]
            elif style == "fancy":
                return ["╭─────╮",
                        "│░░░░░│",
                        "│░░░░░│",
                        "│░░░░░│",
                        "╰─────╯"]
        
        rank_symbol = Card.RANKS[rank]
        suit_symbol = Card.SUITS[suit]
        
        # Color indicators for hearts and diamonds
        heart_diamond = suit in ['hearts', 'diamonds']
        color_indicator = "♥♦" if heart_diamond else ""
        
        if style == "ascii":
            if len(rank_symbol) == 1:
                return ["+-----+",
                        f"|{rank_symbol}{color_indicator}    |",
                        f"|  {suit_symbol}  |",
                        f"|    {rank_symbol}|",
                        "+-----+"]
            else:  # For "10"
                return ["+-----+",
                        f"|{rank_symbol}{color_indicator}   |",
                        f"|  {suit_symbol}  |",
                        f"|   {rank_symbol}|",
                        "+-----+"]
        
        elif style == "fancy":
            if len(rank_symbol) == 1:
                return ["╭─────╮",
                        f"│{rank_symbol}{color_indicator}    │",
                        f"│  {suit_symbol}  │",
                        f"│    {rank_symbol}│",
                        "╰─────╯"]
            else:  # For "10"
                return ["╭─────╮",
                        f"│{rank_symbol}{color_indicator}   │",
                        f"│  {suit_symbol}  │",
                        f"│   {rank_symbol}│",
                        "╰─────╯"]
        
        # Simple fallback
        return [f"[{rank_symbol}{suit_symbol}]"]
    
    @staticmethod
    def draw_card(screen, x, y, rank, suit, style="ascii", face_up=True, selected=False, highlight_char="*"):
        """Draw a card on the screen"""
        card = Card.get_card(rank, suit, style, face_up)
        
        for i, line in enumerate(card):
            if selected:
                # Modify first and last lines for selection highlighting
                if i == 0 or i == len(card) - 1:
                    if style == "ascii":
                        line = line.replace("+", highlight_char)
                    elif style == "fancy":
                        if i == 0:
                            line = line.replace("╭", "╓").replace("╮", "╖")
                        else:
                            line = line.replace("╰", "╙").replace("╯", "╜")

            screen.draw_text(x, y + i, line)
        
        return len(card)  # Return height of the card


class Box:
    """Box component for creating UI elements"""
    
    @staticmethod
    def draw_box(screen, x, y, width, height, style="single", title=None):
        """Draw a box with optional title"""
        styles = {
            "single": {
                "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
                "h": "─", "v": "│"
            },
            "double": {
                "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
                "h": "═", "v": "║"
            },
            "rounded": {
                "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
                "h": "─", "v": "│"
            },
            "ascii": {
                "tl": "+", "tr": "+", "bl": "+", "br": "+",
                "h": "-", "v": "|"
            }
        }
        
        chars = styles.get(style, styles["single"])
        
        top_border = chars["tl"] + chars["h"] * (width - 2) + chars["tr"]
        screen.draw_text(x, y, top_border)
        
        if title:
            title_pos = x + 2
            if len(title) > width - 4:
                title = title[:width - 7] + "..."
            
            screen.draw_text(title_pos, y, chars["h"] + " " + title + " " + chars["h"])
        
        for i in range(1, height - 1):
            screen.draw_text(x, y + i, chars["v"])
            screen.draw_text(x + width - 1, y + i, chars["v"])
        
        bottom_border = chars["bl"] + chars["h"] * (width - 2) + chars["br"]
        screen.draw_text(x, y + height - 1, bottom_border)
        
        return height


class Button:
    """Button component for interactive elements"""
    
    @staticmethod
    def draw_button(screen, x, y, text, width=None, style="single", selected=False):
        """Draw a button with text"""
        if width is None:
            width = len(text) + 4  # Default padding
        
        # Ensure minimum width
        width = max(width, len(text) + 4)
        
        # Box drawing characters
        styles = {
            "single": {
                "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
                "h": "─", "v": "│"
            },
            "rounded": {
                "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
                "h": "─", "v": "│"
            },
            "ascii": {
                "tl": "+", "tr": "+", "bl": "+", "br": "+",
                "h": "-", "v": "|"
            }
        }
        
        chars = styles.get(style, styles["single"])
        
        # Modify style for selected state
        if selected:
            chars["tl"] = "╓"
            chars["tr"] = "╖"
            chars["bl"] = "╙"
            chars["br"] = "╜"
        
        # Draw top border
        screen.draw_text(x, y, chars["tl"] + chars["h"] * (width - 2) + chars["tr"])
        
        # Draw middle with text
        text_padded = text.center(width - 2)
        screen.draw_text(x, y + 1, chars["v"] + text_padded + chars["v"])
        
        # Draw bottom border
        screen.draw_text(x, y + 2, chars["bl"] + chars["h"] * (width - 2) + chars["br"])
        
        return 3  # Height of button
