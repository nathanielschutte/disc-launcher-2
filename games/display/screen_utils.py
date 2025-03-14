"""
Utilities for handling text display in monospace environments
"""

import unicodedata
import re

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    # "\U0001F300-\U0001F5FF"  # symbols & pictographs
    # "\U0001F680-\U0001F6FF"  # transport & map symbols
    # "\U0001F700-\U0001F77F"  # alchemical symbols
    # "\U0001F780-\U0001F7FF"  # geometric shapes
    # "\U0001F800-\U0001F8FF"  # supplemental arrows
    # "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    # "\U0001FA00-\U0001FA6F"  # chess symbols
    # "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    # "\U00002702-\U000027B0"  # dingbats
    # "\U000024C2-\U0001F251" 
    "]+", flags=re.UNICODE
)


EMOJI_REPLACEMENTS = {
    "🎲": "[DICE]",
    "💰": "[MONEY]",
    # "➤": "->",
    "💻": "[CPU]",
    # "♠": "S",
    # "♥": "H",
    # "♦": "D",
    # "♣": "C",
    # "•": "*"
}


def get_string_display_width(text):
    """
    Calculate the display width of a string in a monospace environment
    accounting for full-width and emoji characters.
    """

    width = 0
    for char in text:
        if EMOJI_PATTERN.fullmatch(char):
            width += 2
        else:
            eaw = unicodedata.east_asian_width(char)
            if eaw in ('F', 'W'):
                width += 1
            else:
                width += 1

    return width


def replace_emojis(text):
    """
    Replace emoji characters with ASCII alternatives to maintain monospace alignment
    """

    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    return text


def truncate_to_display_width(text, max_width):
    """
    Truncate a string to fit within a given display width, 
    accounting for double-width characters
    """

    if get_string_display_width(text) <= max_width:
        return text
        
    result = ""
    current_width = 0
    
    for char in text:
        char_width = 2 if EMOJI_PATTERN.fullmatch(char) else 1
        if current_width + char_width > max_width:
            break
        result += char
        current_width += char_width
        
    if len(result) < len(text) and current_width <= max_width - 3:
        result += "..."
        
    return result


def pad_to_display_width(text, width, align="left"):
    """
    Pad a string to a specific display width, accounting for double-width characters
    """

    current_width = get_string_display_width(text)
    padding_needed = max(0, width - current_width)
    
    if align == "left":
        return text + " " * padding_needed
    elif align == "right":
        return " " * padding_needed + text
    elif align == "center":
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return " " * left_pad + text + " " * right_pad
    
    return text
