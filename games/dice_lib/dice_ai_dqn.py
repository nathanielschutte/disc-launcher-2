import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import itertools
import os


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
        """Reset the environment for a new game/turn with debugging"""
        # print(f"DEBUG: Resetting environment, player_score={player_score}, opponent_score={opponent_score}")
        self.current_roll = self._generate_roll(6)
        # print(f"DEBUG: Initial roll: {self.current_roll}")
        self.player_score = player_score
        self.opponent_score = opponent_score
        self.turn_score = 0
        self.dice_remaining = 6
        return self._get_state()
    

    def _generate_roll(self, num_dice):
        """Generate a random dice roll that's guaranteed to not be a bust"""

        # max_attempts = 10  # Limit attempts to prevent infinite loops
        
        # for attempt in range(max_attempts):
        #     # Generate a roll
        #     roll = [random.randint(1, 6) for _ in range(num_dice)]
            
        #     # Check if it's a bust
        #     if not self.die_logic.check_bust(roll):
        #         return roll
            
        #     print(f"DEBUG: Generated a bust roll {roll}, retrying ({attempt+1}/{max_attempts})")
        
        # If we've tried max_attempts times and still got busts, 
        # force some 1's and 5's which are always scorable
        # print("DEBUG: Forcing a non-bust roll after maximum attempts")

        roll = [random.randint(1, 6) for _ in range(num_dice)]
        
        # # Ensure at least one scorable die (1 or 5)
        # if num_dice > 0:
        #     roll[0] = random.choice([1, 5])
        
        return roll
    
    def step(self, action):
        """
        Take an action in the environment.
        Action is now a tuple of (dice_selection, continue_or_pass)
        """

        if action is None:
            # No valid actions available
            return self._get_state(), -100, True, {"action_type": "invalid"}
        
        selection, decision = action
        reward = 0
        done = False
        info = {"action_type": f"select_{decision}"}
        
        # First process the dice selection
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
                    reward = -self.turn_score / 100  # Negative reward
                    self.turn_score = 0
                    done = True  # End of turn
                    return self._get_state(), reward, done, info
            else:
                # Remove selected dice
                new_roll = []
                for i, die in enumerate(self.current_roll):
                    if i not in selection:
                        new_roll.append(die)
                self.current_roll = new_roll
            
            # Reward for selection and continuing
            reward = selection_score / 100 + 0.1
        else:  # 'pass'
            # Add turn score to player score
            self.player_score += self.turn_score
            
            # Reward is proportional to score gained
            reward = self.turn_score / 50
            
            # Check for win
            if self.player_score >= self.goal:
                reward += 100  # Big reward for winning
            
            # End of turn
            done = True
            
            # Additional reward if ahead
            if self.player_score > self.opponent_score:
                reward += 2
        
        return self._get_state(), reward, done, info
    

    def _get_state(self):
        """
        Convert the current game state to a feature vector for the neural network
        """
        # Features we want to include:
        # 1. Current roll (one-hot encoded for each die value, padded if needed)
        # 2. Current player score
        # 3. Opponent score
        # 4. Current turn score
        # 5. Number of dice remaining
        # 6. How far we are from the goal
        
        # One-hot encode current roll (for up to 6 dice)
        roll_features = np.zeros(36)  # 6 dice positions × 6 possible values
        for i, die in enumerate(self.current_roll[:6]):  # Limit to 6 dice
            roll_features[i*6 + (die-1)] = 1
        
        # Normalize scores and distances relative to goal
        normalized_player_score = self.player_score / self.goal
        normalized_opponent_score = self.opponent_score / self.goal
        normalized_turn_score = self.turn_score / self.goal
        normalized_dice_remaining = self.dice_remaining / 6
        normalized_goal_distance = (self.goal - self.player_score) / self.goal
        
        # Combine features into a single state vector
        state = np.concatenate([
            roll_features,
            [normalized_player_score],
            [normalized_opponent_score],
            [normalized_turn_score],
            [normalized_dice_remaining],
            [normalized_goal_distance]
        ])
        
        return state

class DQNAgent:
    """Deep Q-Learning agent for the dice game"""
    
    def __init__(self, state_size, die_logic):
        self.state_size = state_size
        self.die_logic = die_logic
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Get all possible selections and actions
        self.possible_selections = self._get_all_possible_selections(6)
        self.possible_actions = self._get_all_actions(6)
        
        # Action size is all possible (selection, decision) pairs
        self.action_size = len(self.possible_actions)
        
        # Replay memory
        self.memory = deque(maxlen=10000)
        
        # Build model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    
    def _build_model(self):
        """Neural network model for deep Q-learning"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
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
        """Get valid actions for the current state with enhanced debugging"""
        valid_actions = []
        
        # If no dice to select, return empty list
        if not current_roll or len(current_roll) == 0:
            print(f"WARNING: Empty current_roll: {current_roll}")
            return valid_actions
        
        # print(f"DEBUG: Finding valid actions for roll: {current_roll}, turn_score: {turn_score}")
        
        # Get all possible dice selections
        possible_selections = []
        for i in range(1, len(current_roll) + 1):
            for combo in itertools.combinations(range(len(current_roll)), i):
                possible_selections.append(list(combo))
        
        # print(f"DEBUG: Generated {len(possible_selections)} possible selections")
        
        # Score each selection
        for sel in possible_selections:
            try:
                selected_dice = [current_roll[i] for i in sel]
                score = self.die_logic.score(selected_dice)
                
                if score > 0:  # Valid scoring selection
                    # print(f"DEBUG: Selection {sel} -> dice {selected_dice} -> score {score}")
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
        
        # print(f"DEBUG: Found {len(valid_actions)} valid actions")
        return valid_actions
    

    def _action_to_index(self, action):
        """Convert an action to its index in the output layer"""

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
            # Return a default action rather than None
            # This is a fallback and shouldn't happen in normal gameplay
            if len(current_roll) > 0:
                # Try selecting the first die and continuing
                default_action = ([0], 'continue')
                # Check if this would be valid
                selected_die = [current_roll[0]]
                if self.die_logic.score(selected_die) > 0:
                    return default_action
                
            # If we can't construct a valid default action, log a warning
            # print("WARNING: No valid actions available, returning None")
            return None
        
        # Exploration: choose random action
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose best action
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
        """Train the model using randomly sampled experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Compute target Q values
        target = rewards + (1 - dones) * self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1)
        
        # Update primary model
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load a saved model"""
        self.model = load_model(name)
    
    def save(self, name):
        """Save the current model"""
        self.model.save(name)


def train_dice_ai_debug(die_logic, episodes=10000, batch_size=64, save_interval=1000, max_steps_per_episode=100):
    """
    Train the DQN agent for the dice game with debugging and infinite loop protection
    
    Args:
        die_logic: The logic for scoring dice
        episodes: Number of training episodes
        batch_size: Size of batches for replay training
        save_interval: How often to save the model
        max_steps_per_episode: Maximum steps allowed per episode to prevent infinite loops
    """
    # Initialize environment
    env = DiceGameEnvironment(die_logic, goal=2000)
    
    # Determine state and action sizes
    state = env.reset()
    state_size = len(state)
    
    # Create all possible dice selections for 6 dice
    possible_selections = []
    for i in range(1, 6 + 1):
        for combo in itertools.combinations(range(6), i):
            possible_selections.append(list(combo))
    
    # Calculate action size
    action_size = len(possible_selections) + 2  # Dice selections + continue/pass
    
    # Initialize agent with the correct action size
    agent = DQNAgent(state_size, action_size, die_logic)
    
    # Create output directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Metrics for tracking progress
    all_rewards = []
    all_steps = []
    all_scores = []
    
    # Training loop
    for e in range(episodes):
        # Reset environment
        state = env.reset(player_score=random.randint(0, 1000),
                          opponent_score=random.randint(0, 1500))
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Starting episode {e+1}/{episodes}")
        
        # Episode loop with step limit to prevent infinite loops
        while not done and steps < max_steps_per_episode:
            # Debugging
            if steps % 10 == 0:
                print(f"  Step {steps}, Turn Score: {env.turn_score}, Player Score: {env.player_score}")
            
            # Get action from agent
            action = agent.act(state, env.current_roll, env.dice_remaining, env.turn_score)
            
            # Debug the action
            action_str = f"{action}" if isinstance(action, list) else action
            print(f"  Selected action: {action_str}")
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Debug the outcome
            print(f"  Reward: {reward}, Done: {done}")
            if isinstance(action, list) and len(action) > 0:
                selected_dice = [env.current_roll[i] for i in action if i < len(env.current_roll)]
                print(f"  Selected dice: {selected_dice}, Score: {env.die_logic.score(selected_dice)}")
            
            # Store experience in memory
            agent.memorize(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train network
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Check if we hit the step limit
        if steps >= max_steps_per_episode:
            print(f"WARNING: Episode {e+1} hit the step limit of {max_steps_per_episode}!")
        
        # Update target network every episode
        if e % 10 == 0:
            agent.update_target_model()
        
        # Save model periodically
        if e % save_interval == 0:
            agent.save(f'models/dice_ai_episode_{e}.h5')
        
        # Record metrics
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_scores.append(env.player_score)
        
        print(f"Episode: {e+1}/{episodes}, Steps: {steps}, Player Score: {env.player_score}, "
              f"Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    agent.save('models/dice_ai_final.h5')
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(all_steps)
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 3, 3)
    plt.plot(all_scores)
    plt.title('Final Player Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    return agent
        

def train_dice_ai(die_logic, episodes=10000, batch_size=64, save_interval=1000, max_steps_per_episode=100):
    # Initialize environment
    env = DiceGameEnvironment(die_logic, goal=2000)
    
    # Determine state size
    state = env.reset()
    state_size = len(state)
    
    # Initialize agent with the state size
    agent = DQNAgent(state_size, die_logic)
    
    # Create output directory
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training metrics
    all_rewards = []
    all_steps = []
    all_scores = []
    
    # Training loop
    for e in range(episodes):
        state = env.reset(player_score=random.randint(0, 1000),
                         opponent_score=random.randint(0, 1500))
        done = False
        total_reward = 0
        steps = 0
        
        # Episode loop
        while not done and steps < max_steps_per_episode:
            # Get action from agent
            action = agent.act(state, env.current_roll, env.turn_score)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.memorize(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train network
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target network
        if e % 10 == 0:
            agent.update_target_model()

        print(f'Episode {e+1}/{episodes}, Steps: {steps}, Player Score: {env.player_score}, '
              f'Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}')
        
        # Save model periodically
        if e % save_interval == 0:
            agent.save(f'models/dice_ai_episode_{e}.h5')
            print(f"Episode: {e}/{episodes}, Score: {env.player_score}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        # Record metrics
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_scores.append(env.player_score)
    
    # Save final model
    agent.save('models/dice_ai_final.h5')
    
    return agent


class DiceAI:
    """
    AI player for the Dice game using deep Q-learning
    This class wraps the DQN agent to match your existing interface
    """
    
    def __init__(self, risk_level=0.5, logic=None):
        self.risk_level = max(0.0, min(1.0, risk_level))
        self.die_logic = logic
        
        # Get or initialize the DQN agent
        self.state_size = 41  # 36 (dice one-hot) + 5 game state features
        
        # Calculate action size based on possible dice selections
        self.possible_selections = self._get_all_possible_selections(6)
        self.action_size = len(self.possible_selections) + 2  # Dice selections + continue/pass
        
        # Initialize or load the agent
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize or load the DQN agent"""

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
    
    def _get_all_possible_selections(self, dice):
        """Get all possible dice selections"""
        selections = []
        for i in range(1, dice + 1):
            for combo in itertools.combinations(range(dice), i):
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
    
    
    def _score_selections(self, dice, selections):
        """Score all possible selections and return as {selection_indices: score}"""
        scores = {}
        for sel in selections:
            selected_dice = [dice[i] for i in sel]
            score = self.die_logic.score(selected_dice)
            if score > 0:  # Only include valid scoring selections
                scores[tuple(sel)] = score
        return scores
    
    def _prepare_game_state(self, current_roll, player_score, opponent_score, turn_score, dice_remaining):
        """Convert game state to the format expected by the DQN agent"""
        # Create environment to use its state representation
        env = DiceGameEnvironment(self.die_logic)
        env.current_roll = current_roll
        env.player_score = player_score
        env.opponent_score = opponent_score if isinstance(opponent_score, int) else opponent_score[0]  # Handle list input
        env.turn_score = turn_score
        env.dice_remaining = dice_remaining
        
        return env._get_state()
    
    def choose_dice(self, current_roll, player_score, opponent_score, turn_score, goal, dice_remaining):
        """Choose which dice to select based on current game state"""
        if not self.die_logic:
            raise ValueError("Die logic not set. Call set_logic() first.")
        
        # If no agent is available, fall back to rule-based logic
        if not self.agent:
            # Get all possible selections
            possible_selections = self._get_all_possible_selections(len(current_roll))
            
            # Score each selection
            selection_scores = self._score_selections(current_roll, possible_selections)
            
            # If no valid selections, return empty
            if not selection_scores:
                return []
            
            # Find selection with highest score
            best_selection = max(selection_scores.items(), key=lambda x: x[1])
            best_indices = list(best_selection[0])
            
            return best_indices
        
        # Prepare state for the agent
        state = self._prepare_game_state(current_roll, player_score, opponent_score, turn_score, dice_remaining)
        
        # Get action from agent - this returns either dice indices or 'continue'/'pass'
        action = self.agent.act(state, current_roll, dice_remaining, turn_score, training=False)
        
        # If the action is a dice selection, return it
        if isinstance(action, list):
            return action
        
        # If we got a continue/pass action but need to select dice,
        # fall back to selecting the highest scoring combination
        possible_selections = self._get_all_possible_selections(len(current_roll))
        selection_scores = self._score_selections(current_roll, possible_selections)
        
        if not selection_scores:
            return []
        
        best_selection = max(selection_scores.items(), key=lambda x: x[1])
        return list(best_selection[0])
    
    def decide_continue_or_pass(self, current_score, turn_score, opponent_score, goal, dice_remaining):
        """Decide whether to continue rolling or pass"""
        if not self.die_logic:
            raise ValueError("Die logic not set. Call set_logic() first.")
        
        # If we can win by passing, always pass
        if current_score + turn_score >= goal:
            return False
        
        # If no agent is available, use risk level to decide
        if not self.agent:
            # More conservative as score gets higher
            if turn_score > 300 * (1 + self.risk_level):
                return False
            
            # More aggressive with more dice
            if dice_remaining >= 4:
                return True
            
            # Random factor based on risk level
            return random.random() < self.risk_level
        
        # We need current_roll to prepare the state, but we don't have it
        # Let's simulate a roll with even distribution of die values
        simulated_roll = [random.randint(1, 6) for _ in range(dice_remaining)]
        
        # Prepare state for the agent
        state = self._prepare_game_state(simulated_roll, current_score, opponent_score, turn_score, dice_remaining)
        
        # Get action from agent
        action = self.agent.act(state, simulated_roll, dice_remaining, turn_score, training=False)
        
        # If the action is 'continue', continue rolling
        if action == 'continue':
            return True
        
        # If the action is 'pass', pass
        if action == 'pass':
            return False
        
        # If the action is a dice selection, we should continue
        # (this shouldn't happen here, but just in case)
        return True


    def make_turn_decision(self, current_roll, player_score, opponent_score, turn_score, roll_count, goal, dice_remaining=None):
        """
        Make a complete turn decision using the unified approach.
        Returns (continue_decision, selected_dice) to match the existing interface.
        """

        # Calculate dice_remaining for backward compatibility
        if dice_remaining is None:
            dice_remaining = 6 - len(current_roll)
        
        # Prepare state for the agent
        state = self._prepare_game_state(current_roll, player_score, opponent_score, turn_score, dice_remaining)
        
        # Get unified action from agent
        action = self.agent.act(state, current_roll, turn_score, training=False)
        
        if action is None:
            # No valid actions - shouldn't happen in normal game
            # Return a default based on the risk level
            if turn_score > 0 and self.risk_level < 0.5:
                return False, [0] if len(current_roll) > 0 else []
            else:
                return True, [0] if len(current_roll) > 0 else []
        
        # Unpack the action
        selected_dice, decision = action
        continue_decision = (decision == 'continue')
        
        return continue_decision, selected_dice


# Example usage:
if __name__ == "__main__":
    from logic import DieLogic

    die_logic = DieLogic()
    
    # Create and train the agent
    print(f'Training the AI agent...')
    trained_agent = train_dice_ai(die_logic, episodes=10000)
    
    # Test the trained AI
    ai = DiceAI(risk_level=0.7)
    ai.set_logic(die_logic)
    
    # Example game scenario
    roll = [1, 3, 3, 4, 5, 6]
    player_score = 500
    opponent_score = 750
    turn_score = 300
    goal = 2000
    dice_remaining = 6
    
    continue_decision, selected_dice = ai.make_turn_decision(
        roll, player_score, opponent_score, turn_score, 1, goal, dice_remaining
    )
    
    print(f"Roll: {roll}")
    print(f"Selected dice: {selected_dice}")
    print(f"Decision: {'Continue' if continue_decision else 'Pass'}")
