from games.dice_lib.dice_ai_dqn_round2 import train_dice_ai_improved
from games.dice_lib.logic import DieLogic

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.test.is_gpu_available())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Create the die logic
die_logic = DieLogic()

# Train the model - adjust episodes based on available time/computing power
# 5000 episodes should give decent results, 15000+ for better performance
dqn_agent = train_dice_ai_improved(die_logic, episodes=2000, batch_size=128, save_interval=500)

print("Training completed successfully!")