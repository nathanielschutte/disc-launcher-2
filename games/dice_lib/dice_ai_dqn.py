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
        """Reset the environment for a new game/turn"""
        self.current_roll = self._generate_roll(6)
        self.player_score = player_score
        self.opponent_score = opponent_score
        self.turn_score = 0
        self.dice_remaining = 6
        return self._get_state()
    
    def _generate_roll(self, num_dice):
        """Generate a random dice roll"""
        return [random.randint(1, 6) for _ in range(num_dice)]
    
    def step(self, action):
        """
        Take an action in the environment
        
        Actions:
        - For dice selection: tuple or list of dice indices to select
        - For continue/pass decision: 'continue' or 'pass'
        
        Returns:
        - next_state: The new state after taking the action
        - reward: The reward for taking the action
        - done: Whether the game is finished
        - info: Additional information
        """
        reward = 0
        done = False
        info = {"action_type": "unknown"}
        
        # Handle empty or None actions
        if action is None or (isinstance(action, (list, tuple)) and len(action) == 0):
            print("WARNING: Empty action received")
            return self._get_state(), -100, True, {"action_type": "invalid"}
        
        # Dice selection action
        if isinstance(action, (tuple, list)):
            info["action_type"] = "select"
            # Convert to list if tuple
            action = list(action) if isinstance(action, tuple) else action
            
            # Validate the action
            if not all(0 <= idx < len(self.current_roll) for idx in action):
                print(f"Invalid selection: {action} for roll {self.current_roll}")
                return self._get_state(), -50, True, info
            
            # Get the selected dice values
            selected_dice = [self.current_roll[i] for i in action]
            selection_score = self.die_logic.score(selected_dice)
            
            if selection_score == 0:
                print(f"Invalid scoring selection: {selected_dice}")
                return self._get_state(), -50, True, info
            
            # Update state based on selection
            self.turn_score += selection_score
            self.dice_remaining -= len(action)
            self.selected_dice = action
            
            # Small positive reward for valid selection, proportional to score
            reward = selection_score / 100
            
            # If no dice left, automatically roll again with 6 dice
            if self.dice_remaining == 0:
                self.dice_remaining = 6
                self.current_roll = self._generate_roll(6)
                # Check if new roll is a bust
                if self.die_logic.check_bust(self.current_roll):
                    print("BUST after using all dice!")
                    reward = -self.turn_score / 100  # Negative reward proportional to lost score
                    self.turn_score = 0
                    done = True  # End of turn
            else:
                # Remove selected dice
                new_roll = []
                for i, die in enumerate(self.current_roll):
                    if i not in action:
                        new_roll.append(die)
                self.current_roll = new_roll
            
        elif action == 'continue':
            info["action_type"] = "continue"
            # Check if we have a turn score (must have selected dice first)
            if self.turn_score == 0:
                print("Tried to continue without scoring first")
                return self._get_state(), -50, True, info
            
            # Roll new dice
            self.current_roll = self._generate_roll(self.dice_remaining)
            
            # Check if the roll is a bust
            if self.die_logic.check_bust(self.current_roll):
                print(f"BUST on continue! Turn score lost: {self.turn_score}")
                reward = -self.turn_score / 100  # Negative reward proportional to lost score
                self.turn_score = 0
                done = True  # End of turn
            else:
                # Small reward for successful reroll
                reward = 0.1
                
        elif action == 'pass':
            info["action_type"] = "pass"
            # Check if we have a turn score (must have selected dice first)
            if self.turn_score == 0:
                print("Tried to pass without scoring first")
                return self._get_state(), -50, True, info
            
            # Add turn score to player score
            self.player_score += self.turn_score
            
            # Reward is proportional to score gained
            reward = self.turn_score / 50
            
            # Check for win
            if self.player_score >= self.goal:
                print(f"WIN! Player reached {self.player_score} points, goal was {self.goal}")
                reward += 100  # Big reward for winning
                done = True
            else:
                # End of turn
                done = True
                
                # Additional reward if we're catching up or extending our lead
                if self.player_score > self.opponent_score:
                    reward += 2
                    
            print(f"PASS with {self.turn_score} points. New score: {self.player_score}")
        else:
            print(f"Unknown action type: {action}")
            return self._get_state(), -100, True, {"action_type": "unknown"}
        
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
    
    def __init__(self, state_size, action_size, die_logic):
        self.state_size = state_size
        self.action_size = action_size
        self.die_logic = die_logic

        # Action mapping
        self.possible_dice_selections = self._get_all_possible_selections(6)
        self.dice_selection_actions = len(self.possible_dice_selections)
        self.special_actions = 2  # continue, pass

        if self.action_size is None:
            self.action_size = len(self.possible_dice_selections) + self.special_actions
            print(f'Set action size to {self.action_size} based on possible selections and special actions')
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
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
    
    def _get_valid_actions(self, state, current_roll, dice_remaining, turn_score):
        """Get the valid actions for the current state"""
        valid_actions = []
        
        # Get all possible dice selections
        possible_selections = []
        for i in range(1, len(current_roll) + 1):
            for combo in itertools.combinations(range(len(current_roll)), i):
                possible_selections.append(list(combo))
        
        # Score each selection
        for sel in possible_selections:
            selected_dice = [current_roll[i] for i in sel]
            score = self.die_logic.score(selected_dice)
            if score > 0:  # Only include valid scoring selections
                valid_actions.append(sel)
        
        # Add continue/pass options if there's a turn score
        if turn_score > 0 and dice_remaining > 0:
            valid_actions.append('continue')
            valid_actions.append('pass')
        elif turn_score > 0:
            valid_actions.append('pass')
            
        return valid_actions
    
    def _action_to_index(self, action):
        """Convert an action to its index in the output layer"""
        if isinstance(action, list) or isinstance(action, tuple):
            # Find the index of this dice selection in our possible selections
            action_tuple = tuple(sorted(action))
            for i, sel in enumerate(self.possible_dice_selections):
                if tuple(sorted(sel)) == action_tuple:
                    return i
        elif action == 'continue':
            return self.dice_selection_actions
        elif action == 'pass':
            return self.dice_selection_actions + 1
        
        # Invalid action
        return -1
    
    def _index_to_action(self, index):
        """Convert an output index to the corresponding action"""
        if index < self.dice_selection_actions:
            return self.possible_dice_selections[index]
        elif index == self.dice_selection_actions:
            return 'continue'
        elif index == self.dice_selection_actions + 1:
            return 'pass'
        
        # Invalid index
        return None
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        action_index = self._action_to_index(action)
        if action_index >= 0:
            self.memory.append((state, action_index, reward, next_state, done))
    
    def act(self, state, current_roll, dice_remaining, turn_score, training=True):
        """Choose an action based on the current state"""
        valid_actions = self._get_valid_actions(state, current_roll, dice_remaining, turn_score)
        
        if not valid_actions:
            # If no valid actions, return pass (will end the turn if possible)
            print("No valid actions available!")
            if turn_score > 0:
                return 'pass'
            # If we have no turn score yet but no valid moves, just select the first die
            # This shouldn't happen in a proper game but prevents getting stuck
            if len(current_roll) > 0:
                return [0]
            return []
        
        # Exploration: choose a random action with probability epsilon
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: choose the best action based on the model
        act_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # Filter to only valid actions
        valid_indices = [self._action_to_index(action) for action in valid_actions]
        valid_values = [(i, act_values[i]) for i in valid_indices if i >= 0]
        
        if not valid_values:
            return random.choice(valid_actions)
        
        # Choose the action with the highest Q-value
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
        

def train_dice_ai(die_logic, episodes=10000, batch_size=64, save_interval=1000):
    """
    Train the DQN agent for the dice game
    
    Args:
        die_logic: The logic for scoring dice
        episodes: Number of training episodes
        batch_size: Size of batches for replay training
        save_interval: How often to save the model
    """
    # Initialize environment
    env = DiceGameEnvironment(die_logic, goal=2000)
    
    # Determine state and action sizes
    state = env.reset()
    state_size = len(state)
    
    # Calculate action size
    agent = DQNAgent(state_size, None, die_logic)
    
    # Create output directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    for e in range(episodes):
        # Reset environment
        state = env.reset(player_score=random.randint(0, 1000),
                          opponent_score=random.randint(0, 1500))
        done = False
        total_reward = 0
        
        while not done:
            print(f'Episode: {e}, State: {state}, Player Score: {env.player_score}, Opponent Score: {env.opponent_score}, Turn Score: {env.turn_score}, Dice Remaining: {env.dice_remaining}')

            # Get action from agent
            action = agent.act(state, env.current_roll, env.dice_remaining, env.turn_score)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience in memory
            agent.memorize(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Train network
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target network every episode
        if e % 10 == 0:
            agent.update_target_model()
        
        # Save model periodically
        if e % save_interval == 0:
            agent.save(f'models/dice_ai_episode_{e}.h5')
            print(f"Episode: {e}/{episodes}, Score: {env.player_score}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
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
        self.agent = DQNAgent(self.state_size, self.action_size, self.die_logic)
        
        # Try to load a pre-trained model if available
        try:
            self.agent.load('models/dice_ai_final.h5')
            print("Loaded pre-trained model")
        except:
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
    
    def make_turn_decision(self, current_roll, player_score, opponent_score, turn_score, roll_count, goal, dice_remaining):
        """
        Make a complete turn decision - what dice to select and whether to continue
        
        This method matches the interface used in your game code
        """
        # Handle both interface versions for backward compatibility  
        if dice_remaining is None:
            dice_remaining = 6 - len(current_roll)
        
        # If first selection of turn
        if turn_score == 0:
            selected_indices = self.choose_dice(
                current_roll, player_score, opponent_score, turn_score, goal, dice_remaining
            )
            return True, selected_indices
        
        # Choose dice
        selected_indices = self.choose_dice(
            current_roll, player_score, opponent_score, turn_score, goal, dice_remaining
        )
        
        # Calculate new turn score after this selection
        selected_dice = [current_roll[i] for i in selected_indices]
        selection_score = self.die_logic.score(selected_dice)
        new_turn_score = turn_score + selection_score
        new_dice_remaining = dice_remaining - len(selected_indices)
        
        # Decide whether to continue or pass
        continue_decision = self.decide_continue_or_pass(
            player_score, new_turn_score, opponent_score, goal, new_dice_remaining
        )
        
        # Return decision and selected dice
        return continue_decision, selected_indices


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
