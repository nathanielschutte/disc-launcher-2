"""
Improved Deep Q-Learning AI player for the Dice game

This implementation fixes several issues with the original DQN implementation:
1. Improves the reward structure to better incentivize strategic play
2. Enhances state representation to capture more game information
3. Uses a better neural network architecture
4. Creates a more realistic training environment
5. Adds supervised learning from expert demonstrations
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import itertools
import os


# Force TensorFlow to use the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Make sure TensorFlow can access your GPU.")


class DiceGameEnvironment:
    """Environment that simulates the dice game for training"""
    
    def __init__(self, die_logic, goal=2000):
        self.die_logic = die_logic
        self.goal = goal
        self.current_roll = []
        self.player_score = 0
        self.opponent_score = 0
        self.turn_score = 0
        self.selected_dice = []
        self.dice_remaining = 6
        
    def reset(self, player_score=0, opponent_score=0):
        """Reset the environment for a new game/turn with more realistic scenarios"""
        # Add variability to starting position
        if random.random() < 0.7:  # 70% of the time use provided scores
            self.player_score = player_score
            self.opponent_score = opponent_score
        else:  # 30% of the time, generate realistic mid-game scenarios
            scenario_type = random.random()
            if scenario_type < 0.33:  # Player ahead
                self.player_score = random.randint(int(self.goal * 0.3), int(self.goal * 0.7))
                self.opponent_score = random.randint(0, self.player_score - 200)
            elif scenario_type < 0.66:  # Player behind
                self.opponent_score = random.randint(int(self.goal * 0.3), int(self.goal * 0.7))
                self.player_score = random.randint(0, self.opponent_score - 200)
            else:  # Close race
                base = random.randint(int(self.goal * 0.4), int(self.goal * 0.8))
                self.player_score = base + random.randint(-200, 200)
                self.opponent_score = base + random.randint(-200, 200)
        
        # Generate initial roll with natural bust probability
        self.current_roll = self._generate_roll(6)
        self.turn_score = 0
        self.dice_remaining = 6
        return self._get_state()

    def _generate_roll(self, num_dice):
        """Generate a more realistic dice roll with proper bust probability"""
        # For empty dice, return empty roll
        if num_dice <= 0:
            return []
        
        # First generate a completely random roll
        roll = [random.randint(1, 6) for _ in range(num_dice)]
        
        # Check if it's a bust
        is_bust = self.die_logic.check_bust(roll)
        
        # About 1/3 of the time, if it's a bust, regenerate with at least one scorable die
        # This maintains realistic bust probability while ensuring training variety
        if is_bust and random.random() < 0.35:
            roll = [random.randint(1, 6) for _ in range(num_dice - 1)]
            # Add one guaranteed scorable die (1 or 5)
            roll.append(random.choice([1, 5]))
            random.shuffle(roll)
        
        return roll
    
    def step(self, action):
        """
        Take an action in the environment with improved reward structure
        Action is a tuple of (dice_selection, continue_or_pass)
        """
        if action is None:
            # No valid actions available
            return self._get_state(), -100, True, {"action_type": "invalid"}
        
        selection, decision = action
        reward = 0
        done = False
        info = {"action_type": f"select_{decision}"}
        
        # Validate the selection
        if not all(0 <= idx < len(self.current_roll) for idx in selection):
            return self._get_state(), -50, True, {"action_type": "invalid_selection"}
        
        # Get selected dice values
        selected_dice = [self.current_roll[i] for i in selection]
        selection_score = self.die_logic.score(selected_dice)
        
        if selection_score == 0:
            return self._get_state(), -50, True, {"action_type": "invalid_score"}
        
        # Update state based on selection
        self.turn_score += selection_score
        self.dice_remaining -= len(selection)
        self.selected_dice = selection
        
        # Process decision (continue or pass)
        if decision == 'continue':
            # If no dice left, roll new dice
            if self.dice_remaining == 0:
                self.dice_remaining = 6
                self.current_roll = self._generate_roll(6)
                
                # Check for bust
                if self.die_logic.check_bust(self.current_roll):
                    # Bigger penalty for busting with high turn score
                    bust_penalty = -5 - (self.turn_score / 100)
                    self.turn_score = 0
                    done = True  # End of turn
                    return self._get_state(), bust_penalty, done, {"action_type": "bust"}
            else:
                # Remove selected dice
                new_roll = []
                for i, die in enumerate(self.current_roll):
                    if i not in selection:
                        new_roll.append(die)
                self.current_roll = new_roll
            
            # Small reward for selection and continuing
            reward = selection_score / 200
            
            # Small bonus for continuing with many dice left (safer)
            dice_factor = self.dice_remaining / 6
            reward += dice_factor * 0.5
            
            # Small penalty for risky continues with few dice
            if self.dice_remaining <= 2 and self.turn_score > 300:
                reward -= 0.5
                
        else:  # 'pass'
            # Add turn score to player score
            self.player_score += self.turn_score
            
            # Higher reward for banking points when close to winning
            progress_factor = min(1.5, self.turn_score / (self.goal - self.player_score + self.turn_score))
            
            # Banking points is better when opponent is close to winning
            urgency_factor = 1.0
            if (self.goal - self.opponent_score) < 500:
                urgency_factor = 1.5
            
            # Base reward is proportional to score gained with multipliers
            reward = (self.turn_score / 50) * progress_factor * urgency_factor
            
            # Bonus for winning the game
            if self.player_score >= self.goal:
                reward += 20
            
            # End of turn
            done = True
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """
        Create a more informative state representation for the neural network
        """
        # One-hot encode current roll (for up to 6 dice)
        roll_features = np.zeros(36)  # 6 dice positions × 6 possible values
        for i, die in enumerate(self.current_roll[:6]):  # Limit to 6 dice
            roll_features[i*6 + (die-1)] = 1
        
        # Count value frequencies for pattern detection (e.g., pairs, triples)
        dice_counts = np.zeros(6)  # Count of each value 1-6
        for die in self.current_roll:
            dice_counts[die-1] += 1
        
        # Normalize frequencies against total dice
        if len(self.current_roll) > 0:
            dice_counts = dice_counts / len(self.current_roll)
        
        # Compute value distance from winning
        goal_distance = (self.goal - self.player_score) / self.goal
        turn_progress = self.turn_score / max(1000, self.goal / 2)
        
        # Compute relative position to opponent
        relative_position = (self.player_score - self.opponent_score) / self.goal
        opponent_danger = max(0, 1 - ((self.goal - self.opponent_score) / self.goal)) 
        
        # Risk metrics
        dice_ratio = self.dice_remaining / 6.0
        bust_risk = max(0, 1 - (dice_ratio * 0.8))  # Higher with fewer dice
        
        # Compute whether we can win by passing
        can_win_by_passing = float(self.player_score + self.turn_score >= self.goal)
        
        # Combine features into state vector
        state = np.concatenate([
            roll_features,           # 36 values for dice positions
            dice_counts,             # 6 values for dice frequencies
            [goal_distance],         # Distance to goal
            [turn_progress],         # Current turn accumulated value
            [relative_position],     # Position relative to opponent
            [opponent_danger],       # How close opponent is to winning
            [dice_ratio],            # Available dice ratio
            [bust_risk],             # Risk of busting
            [can_win_by_passing]     # Binary flag if we can win by passing
        ])
        
        return state


class DQNAgent:
    """Improved Deep Q-Learning agent for the dice game"""
    
    def __init__(self, state_size, die_logic):
        self.state_size = state_size
        self.die_logic = die_logic
        
        # Hyperparameters
        self.gamma = 0.98  # Discount factor (increased for longer-term planning)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # Slower decay for better exploration
        self.learning_rate = 0.0005  # Lower learning rate for more stability
        
        # Get all possible selections and actions
        self.possible_selections = self._get_all_possible_selections(6)
        self.possible_actions = self._get_all_actions(6)
        
        # Action size is all possible (selection, decision) pairs
        self.action_size = len(self.possible_actions)
        
        # Replay memory with larger capacity
        self.memory = deque(maxlen=50000)
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    
    def _build_model(self):
        """Build a more sophisticated neural network for deep Q-learning"""
        model = Sequential()
        
        # Larger, deeper network with dropout for regularization
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout to prevent overfitting
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        
        # Output layer - one node per action
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use a more sophisticated optimizer with gradient clipping
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)  # Huber loss is more robust
        
        return model
    
    def update_target_model(self):
        """Update the target model to match the primary model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def _get_all_possible_selections(self, max_dice):
        """Get all possible dice selection combinations"""
        selections = []
        for i in range(1, max_dice + 1):
            for combo in itertools.combinations(range(max_dice), i):
                selections.append(list(combo))
        return selections

    def _get_all_actions(self, max_dice):
        """Get all possible actions as (dice_selection, continue_or_pass) pairs"""
        actions = []
        
        # Get all possible dice selections
        selections = []
        for i in range(1, max_dice + 1):
            for combo in itertools.combinations(range(max_dice), i):
                selections.append(list(combo))
        
        # For each selection, create two actions (continue or pass)
        for selection in selections:
            actions.append((selection, 'continue'))
            actions.append((selection, 'pass'))
        
        return actions
    
    def _get_valid_actions(self, state, current_roll, turn_score):
        """Get valid actions for the current state"""
        valid_actions = []
        
        # If no dice to select, return empty list
        if not current_roll or len(current_roll) == 0:
            return valid_actions
        
        # Get all possible dice selections
        possible_selections = []
        for i in range(1, len(current_roll) + 1):
            for combo in itertools.combinations(range(len(current_roll)), i):
                possible_selections.append(list(combo))
        
        # Score each selection
        for sel in possible_selections:
            try:
                selected_dice = [current_roll[i] for i in sel]
                score = self.die_logic.score(selected_dice)
                
                if score > 0:  # Valid scoring selection
                    # First time selecting this turn, can only continue
                    if turn_score == 0:
                        valid_actions.append((sel, 'continue'))
                    else:
                        # Can choose to continue or pass after initial selection
                        valid_actions.append((sel, 'continue'))
                        valid_actions.append((sel, 'pass'))
            except Exception as e:
                print(f"Error scoring selection {sel} from roll {current_roll}: {e}")
                continue
        
        return valid_actions
    
    def _action_to_index(self, action):
        """Convert an action to its index in the output layer"""
        if action is None:
            return -1
            
        selection, decision = action
        
        # First, find the index of this selection within all possible selections
        selection_tuple = tuple(sorted(selection))
        selection_index = -1
        for i, sel in enumerate(self.possible_selections):
            if tuple(sorted(sel)) == selection_tuple:
                selection_index = i
                break
        
        if selection_index == -1:
            return -1  # Invalid selection
        
        # Calculate action index based on selection index and decision
        if decision == 'continue':
            return selection_index * 2
        else:  # 'pass'
            return selection_index * 2 + 1
    
    def _index_to_action(self, index):
        """Convert an output index to the corresponding action"""
        selection_index = index // 2
        is_continue = (index % 2) == 0
        
        if selection_index >= len(self.possible_selections):
            return None  # Invalid index
        
        selection = self.possible_selections[selection_index]
        decision = 'continue' if is_continue else 'pass'
        
        return (selection, decision)
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Skip storing experiences with None actions
        if action is None:
            return
        
        action_index = self._action_to_index(action)
        if action_index >= 0:
            self.memory.append((state, action_index, reward, next_state, done))
    
    def act(self, state, current_roll, turn_score, training=True):
        """Choose an action based on the current state"""
        valid_actions = self._get_valid_actions(state, current_roll, turn_score)
        
        if not valid_actions:
            # No valid actions available
            return None
        
        # Exploration: choose random action
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best action using Double DQN approach
        act_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # Filter to only valid actions
        valid_indices = [self._action_to_index(action) for action in valid_actions]
        valid_values = [(i, act_values[i]) for i in valid_indices if i >= 0]
        
        if not valid_values:
            return random.choice(valid_actions)
        
        # Choose action with highest Q-value
        best_index, _ = max(valid_values, key=lambda x: x[1])
        return self._index_to_action(best_index)
    
    def replay(self, batch_size):
        """Improved training with Double DQN approach"""
        if len(self.memory) < batch_size:
            return
        
        # Sample based on recency (more recent experiences have higher probability)
        sample_weights = [0.5 + i/len(self.memory) for i in range(len(self.memory))]
        minibatch_indices = random.choices(range(len(self.memory)), weights=sample_weights, k=batch_size)
        minibatch = [self.memory[i] for i in minibatch_indices]
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Double DQN approach - use primary network to select action, target network to evaluate
        target_f = self.model.predict(states, verbose=0)
        
        # For each state in the batch
        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if done:
                target = reward
            else:
                # Double DQN formula: Q(s,a) = r + γ * Q_target(s', argmax_a' Q(s', a'))
                actions_by_primary = np.argmax(self.model.predict(np.array([next_state]), verbose=0)[0])
                next_q_value = self.target_model.predict(np.array([next_state]), verbose=0)[0][actions_by_primary]
                target = reward + self.gamma * next_q_value
            
            target_f[i][action] = target
        
        # Train in smaller mini-batches for stability
        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=min(32, batch_size))
        
        # Decay epsilon with a better curve - slower at first, faster in middle, slower at end
        if self.epsilon > self.epsilon_min:
            decay_rate = (self.epsilon - self.epsilon_min) / (1.0 - self.epsilon_min)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate * self.epsilon_decay)

    def load(self, name):
        """Load a saved model"""
        self.model = load_model(name)
    
    def save(self, name):
        """Save the current model"""
        self.model.save(name)


def supervised_pretraining(agent, die_logic, episodes=5000):
    """
    Pretrain the DQN agent using supervised learning from a rule-based expert
    
    Args:
        agent: The DQN agent to pretrain
        die_logic: The dice scoring logic
        episodes: Number of pretraining episodes
    """
    # Import the Claude AI expert agent
    try:
        from games.dice_lib.dice_ai_claude import DiceAI as ExpertAI
    except ImportError:
        print("Could not import Claude AI expert. Falling back to standard AI.")
        from games.dice_lib.dice_ai_v1 import DiceAI as ExpertAI
    
    # Create environment for collecting experiences
    env = DiceGameEnvironment(die_logic, goal=2000)
    
    # Create expert agent with high skill level
    expert = ExpertAI(risk_level=0.6)
    expert.set_logic(die_logic)
    
    print("Starting supervised pretraining from expert agent...")
    
    # Collect experiences from expert
    pretraining_memory = []
    
    for e in range(episodes):
        state = env.reset(
            player_score=random.randint(0, 1500),
            opponent_score=random.randint(0, 1500)
        )
        done = False
        
        # One full episode of expert play
        while not done:
            # Get expert action (dice selection and decision)
            action_decision = expert.make_turn_decision(
                env.current_roll,
                env.player_score,
                env.opponent_score,
                env.turn_score,
                1,  # roll count doesn't matter for expert
                env.goal,
                len(env.current_roll)
            )
            
            # Convert to agent's action format
            continue_decision, selected_dice = action_decision
            action = (selected_dice, 'continue' if continue_decision else 'pass')
            
            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            
            # Store in pretraining memory with expert's action
            action_index = agent._action_to_index(action)
            if action_index >= 0:
                pretraining_memory.append((state, action_index, reward, next_state, done))
            
            # Move to next state
            state = next_state
        
        # Progress reporting
        if (e+1) % 500 == 0:
            print(f"Collected {e+1}/{episodes} expert episodes")
    
    # Now train the DQN on these expert experiences
    print("Training DQN on expert demonstrations...")
    
    # Create batches
    batch_size = 64
    pretraining_iterations = min(10000, len(pretraining_memory) // batch_size)
    
    for i in range(pretraining_iterations):
        # Sample a batch
        minibatch = random.sample(pretraining_memory, batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Compute targets (simplified for supervised learning - just imitate the expert)
        target_f = agent.model.predict(states, verbose=0)
        
        for j, (state, action, reward, next_state, done) in enumerate(
                zip(states, actions, rewards, next_states, dones)):
            if done:
                target = reward
            else:
                # Use standard Bellman equation for Q-learning
                target = reward + agent.gamma * np.amax(agent.target_model.predict(
                    np.array([next_state]), verbose=0)[0])
            
            target_f[j][action] = target
        
        # Train the model to predict the expert's actions
        agent.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)
        
        # Update progress regularly
        if (i+1) % 500 == 0:
            print(f"Completed {i+1}/{pretraining_iterations} pretraining iterations")
            # Update target network occasionally
            agent.update_target_model()
    
    # Final update to target network
    agent.update_target_model()
    
    print("Supervised pretraining completed.")
    return agent


def train_dice_ai_improved(die_logic, episodes=10000, batch_size=128, save_interval=500):
    """
    Improved training process for the DQN agent
    
    Args:
        die_logic: The dice scoring logic
        episodes: Number of training episodes
        batch_size: Size of replay batches
        save_interval: How often to save model checkpoints
    """

    # Configure TensorFlow to use GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPUs")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
    else:
        print("No GPU found, using CPU")

    print("Initializing improved DQN training...")
    
    # Initialize environment
    env = DiceGameEnvironment(die_logic, goal=2000)
    
    # Determine state size from the improved state representation
    state = env.reset()
    state_size = len(state)
    
    # Initialize agent with improved parameters
    agent = DQNAgent(state_size, die_logic)
    
    # Slow exploration decay for more thorough learning
    agent.epsilon_decay = 0.9995
    
    # Create output directory if needed
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # First, apply supervised pretraining if desired
    agent = supervised_pretraining(agent, die_logic, episodes=3000)
    
    # Save the pretrained model
    agent.save('models/dice_ai_pretrained.h5')
    print("Pretrained model saved.")
    
    # Training metrics
    all_rewards = []
    all_steps = []
    all_scores = []
    all_busts = []
    
    # Moving averages for evaluation
    reward_window = deque(maxlen=100)
    score_window = deque(maxlen=100)
    bust_window = deque(maxlen=100)
    
    print("Starting main training loop...")
    
    # Main training loop with progressive difficulty
    for e in range(episodes):
        # Gradually increase opponent skill level
        opponent_skill = min(0.3 + (e / episodes) * 0.4, 0.7)
        
        # Generate realistic game state
        opponent_score = random.randint(0, 1500)
        player_score = random.randint(max(0, opponent_score - 800), min(1500, opponent_score + 800))
        
        state = env.reset(player_score=player_score, opponent_score=opponent_score)
        total_reward = 0
        steps = 0
        bust_count = 0
        
        # Episode loop
        done = False
        while not done:
            # Get action from agent
            action = agent.act(state, env.current_roll, env.turn_score)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Record bust events
            if info.get("action_type") == "bust":
                bust_count += 1
            
            # Store experience
            agent.memorize(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Periodically train the network
            # Train more intensively as we get more data
            training_intensity = min(4, 1 + len(agent.memory) // 5000)
            for _ in range(training_intensity):
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
        
        # Update target network periodically
        if e % 10 == 0:
            agent.update_target_model()
        
        # Record metrics
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_scores.append(env.player_score)
        all_busts.append(bust_count)
        
        # Update moving averages
        reward_window.append(total_reward)
        score_window.append(env.player_score)
        bust_window.append(bust_count)
        
        # Detailed logging
        if (e+1) % 100 == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            avg_score = sum(score_window) / len(score_window)
            avg_busts = sum(bust_window) / len(bust_window)
            
            print(f"Episode: {e+1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Avg Busts: {avg_busts:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically with validation score
        if (e+1) % save_interval == 0:
            # Run a quick validation
            val_scores = []
            for _ in range(20):
                state = env.reset(player_score=random.randint(0, 1200))
                done = False
                while not done:
                    action = agent.act(state, env.current_roll, env.turn_score, training=False)
                    next_state, _, done, _ = env.step(action)
                    state = next_state
                val_scores.append(env.player_score)
            
            val_score = sum(val_scores) / len(val_scores)
            agent.save(f'models/dice_ai_episode_{e+1}_score_{int(val_score)}.h5')
            print(f"Model saved at episode {e+1} with validation score: {val_score:.2f}")
    
    # Save final model
    agent.save('models/dice_ai_final.h5')
    print("Training completed. Final model saved.")
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(all_scores)
    plt.title('Final Player Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(2, 2, 3)
    plt.plot(all_steps)
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 4)
    plt.plot(all_busts)
    plt.title('Busts per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Busts')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    return agent


class DiceAI:
    """
    Improved AI player for the Dice game using deep Q-learning
    This class wraps the DQN agent to match your existing interface
    """
    
    def __init__(self, risk_level=0.5, logic=None, continue_learning=False):
        self.risk_level = max(0.0, min(1.0, risk_level))
        self.die_logic = logic
        self.continue_learning = continue_learning
        self.learning_frequency = 10  # How many decisions before training
        self.decision_counter = 0
        self.experiences = []  # Store game experiences
        
        # State size calculation
        dice_onehot = 36  # 6 dice × 6 values
        dice_counts = 6   # Frequency of each value
        game_state = 7    # Various game state features
        self.state_size = dice_onehot + dice_counts + game_state
        
        # Calculate action size based on possible dice selections
        self.possible_selections = self._get_all_possible_selections(6)
        self.action_size = len(self.possible_selections) * 2  # For each selection: continue or pass
        
        # Initialize or load the agent
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize or load the DQN agent"""
        if not self.die_logic:
            print("Warning: die_logic not set during initialization")
            return
            
        self.agent = DQNAgent(self.state_size, self.die_logic)
        
        # Try to load a pre-trained model if available
        try:
            self.agent.load('models/dice_ai_final.h5')
            print("Loaded pre-trained model")
        except Exception as e:
            print(f'Error loading pre-trained model: {e}')
            print("No pre-trained model found, using new model")
    
    def set_logic(self, die_logic):
        """Set the die logic for scoring calculations"""
        self.die_logic = die_logic
        if self.agent:
            self.agent.die_logic = die_logic
        else:
            self._initialize_agent()
    
    def _get_all_possible_selections(self, dice):
        """Get all possible dice selections"""
        selections = []
        for i in range(1, dice + 1):
            for combo in itertools.combinations(range(dice), i):
                selections.append(list(combo))
        return selections
    
    def _prepare_game_state(self, current_roll, player_score, opponent_score, turn_score, roll_count, goal, dice_remaining):
        """Create enhanced state representation for the model"""
        # Create environment to use its state representation
        env = DiceGameEnvironment(self.die_logic)
        env.current_roll = current_roll
        env.player_score = player_score
        env.opponent_score = opponent_score if isinstance(opponent_score, int) else opponent_score[0]  # Handle list input
        env.turn_score = turn_score
        env.dice_remaining = dice_remaining
        env.goal = goal
        
        return env._get_state()
    
    def make_turn_decision(self, current_roll, player_score, opponent_scores, turn_score, roll_count, goal, dice_remaining=None):
        """
        Make a complete turn decision using the deep Q-learning approach.
        Returns (continue_decision, selected_dice) to match the existing interface.
        """
        if not self.die_logic:
            raise ValueError("Die logic not set. Call set_logic() first.")

        # Calculate dice_remaining for backward compatibility
        if dice_remaining is None:
            dice_remaining = 6 - len(current_roll)
        
        # If no agent is available, fall back to rule-based logic
        if not self.agent:
            # Get all possible selections
            possible_selections = self._get_all_possible_selections(len(current_roll))
            
            # Score each selection
            selection_scores = {}
            for sel in possible_selections:
                selected_dice = [current_roll[i] for i in sel]
                score = self.die_logic.score(selected_dice)
                if score > 0:  # Only include valid scoring selections
                    selection_scores[tuple(sel)] = score
            
            # If no valid selections, return default
            if not selection_scores:
                return True, [0] if len(current_roll) > 0 else []
            
            # Find selection with highest score
            best_selection = max(selection_scores.items(), key=lambda x: x[1])
            best_indices = list(best_selection[0])
            
            # Simple decision logic based on risk level
            continue_decision = turn_score < (500 * self.risk_level) and dice_remaining > 2
            
            return continue_decision, best_indices
        
        # Prepare enhanced state for the agent
        state = self._prepare_game_state(
            current_roll, player_score, opponent_scores, turn_score, roll_count, goal, dice_remaining
        )
        
        # Get unified action from agent
        action = self.agent.act(state, current_roll, turn_score, training=False)
        
        if action is None:
            # No valid actions - fallback strategy based on risk level
            if turn_score > 300 and self.risk_level < 0.7:
                return False, [0] if len(current_roll) > 0 else []  # Pass when risk-averse with good score
            else:
                return True, [0] if len(current_roll) > 0 else []   # Continue when risk-tolerant or low score
        
        # Unpack the action
        selected_dice, decision = action
        continue_decision = (decision == 'continue')
        
        # Handle online learning if enabled
        if self.continue_learning and self.agent:
            # Record this state-action pair for learning
            state = self._prepare_game_state(
                current_roll, player_score, opponent_scores, turn_score, roll_count, goal, dice_remaining
            )
            action = (selected_dice, decision)

            # Store experience for later batch updates
            self.experiences.append({
                'state': state,
                'action': action,
                'reward': 0,  # Will be updated with the outcome
                'turn_score': turn_score
            })
            
            # Update previous experience reward based on this turn's outcome
            if len(self.experiences) > 1:
                prev_exp = self.experiences[-2]
                if decision == 'pass':
                    # Reward for successfully banking points
                    reward = (turn_score - prev_exp['turn_score']) / 100
                    prev_exp['reward'] = reward
                    prev_exp['next_state'] = state
                    prev_exp['done'] = True
                elif decision == 'continue':
                    # Reward for points gained by continuing
                    reward = (turn_score - prev_exp['turn_score']) / 200
                    prev_exp['reward'] = reward
                    prev_exp['next_state'] = state
                    prev_exp['done'] = False
                
                # Add to agent's memory for batch learning
                if 'next_state' in prev_exp:
                    self.agent.memorize(
                        prev_exp['state'], 
                        prev_exp['action'],
                        prev_exp['reward'],
                        prev_exp['next_state'],
                        prev_exp['done']
                    )
            
            # Periodically train the model on recent experiences
            self.decision_counter += 1
            if self.decision_counter >= self.learning_frequency and len(self.agent.memory) > 32:
                self.agent.replay(batch_size=32)
                self.decision_counter = 0
        
        return continue_decision, selected_dice