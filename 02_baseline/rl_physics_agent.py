#!/usr/bin/env python3
"""
Reinforcement Learning Agent for Physics Question Answering
A simple RL system that learns to answer physics questions correctly using Q-learning.
"""

import numpy as np
import random
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from physics_question_generator import PhysicsQuestionGenerator


class PhysicsRLEnvironment:
    """Environment for training an RL agent on physics questions."""
    
    def __init__(self, question_generator: PhysicsQuestionGenerator):
        self.question_generator = question_generator
        self.current_question = None
        self.current_state = None
        self.episode_reward = 0
        self.episode_count = 0
        self.correct_answers = 0
        self.total_answers = 0
        
    def reset(self) -> str:
        """Reset environment and return initial state."""
        self.current_question = self.question_generator.generate_question()
        self.current_state = self._encode_state(self.current_question)
        self.episode_reward = 0
        return self.current_state
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take action (answer) and return next state, reward, done, info."""
        self.total_answers += 1
        
        # Check if answer is correct
        is_correct = self._check_answer(action)
        
        if is_correct:
            reward = 1.0
            self.correct_answers += 1
        else:
            reward = -0.1  # Small negative reward for wrong answers
        
        self.episode_reward += reward
        
        # Generate new question for next state
        self.current_question = self.question_generator.generate_question()
        next_state = self._encode_state(self.current_question)
        
        # Episode is done after each question (single-step episodes)
        done = True
        
        info = {
            'correct': is_correct,
            'question_type': self.current_question['type'],
            'correct_answer': self.current_question['answer'],
            'user_answer': action,
            'episode_reward': self.episode_reward
        }
        
        return next_state, reward, done, info
    
    def _encode_state(self, question_data: Dict) -> str:
        """Encode question into a state representation."""
        # Simple state encoding: question type + key words from question
        question_text = question_data['question'].lower()
        question_type = question_data['type']
        
        # Extract key physics terms
        key_terms = []
        physics_terms = ['velocity', 'acceleration', 'force', 'energy', 'work', 'mass', 'distance', 'time', 'height']
        for term in physics_terms:
            if term in question_text:
                key_terms.append(term)
        
        # Create state representation
        state = f"{question_type}_{'_'.join(sorted(key_terms))}"
        return state
    
    def _check_answer(self, user_answer: str) -> bool:
        """Check if user answer matches the correct answer."""
        correct_answer = self.current_question['answer']
        
        # Normalize answers for comparison
        def normalize_answer(answer):
            # Remove units and convert to lowercase
            answer = answer.lower().strip()
            # Extract numerical value
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return float(numbers[0])
            return answer
        
        try:
            user_num = normalize_answer(user_answer)
            correct_num = normalize_answer(correct_answer)
            
            if isinstance(user_num, float) and isinstance(correct_num, float):
                # Allow small tolerance for floating point answers
                return abs(user_num - correct_num) < 0.01
            else:
                # String comparison for non-numerical answers
                return user_num == correct_num
        except:
            # If parsing fails, do string comparison
            return user_answer.strip().lower() == correct_answer.strip().lower()
    
    def get_accuracy(self) -> float:
        """Get current accuracy rate."""
        if self.total_answers == 0:
            return 0.0
        return self.correct_answers / self.total_answers


class QLearningAgent:
    """Q-learning agent for physics question answering."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Action space: possible answers (we'll generate these dynamically)
        self.action_space = set()
        
        # Training statistics
        self.training_rewards = []
        self.training_accuracy = []
        
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy."""
        # Add new actions to action space
        for action in available_actions:
            self.action_space.add(action)
        
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, done: bool):
        """Update Q-value using Q-learning formula."""
        if done:
            # Terminal state
            target = reward
        else:
            # Non-terminal state
            next_q_values = [self.q_table[next_state][a] for a in self.action_space]
            target = reward + self.discount_factor * max(next_q_values) if next_q_values else reward
        
        # Q-learning update
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters."""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'action_space': list(self.action_space)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.epsilon = model_data['epsilon']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.action_space = set(model_data['action_space'])


class PhysicsAnswerGenerator:
    """Generates possible answers for physics questions."""
    
    def __init__(self):
        self.question_generator = PhysicsQuestionGenerator()
    
    def generate_answer_options(self, question_data: Dict, num_options: int = 4) -> List[str]:
        """Generate multiple choice options for a physics question."""
        correct_answer = question_data['answer']
        question_type = question_data['type']
        
        # Generate correct answer
        options = [correct_answer]
        
        # Generate plausible wrong answers based on question type
        wrong_answers = self._generate_wrong_answers(question_data, num_options - 1)
        options.extend(wrong_answers)
        
        # Shuffle options
        random.shuffle(options)
        
        return options
    
    def _generate_wrong_answers(self, question_data: Dict, num_wrong: int) -> List[str]:
        """Generate plausible wrong answers for a question."""
        correct_answer = question_data['answer']
        question_type = question_data['type']
        question_text = question_data['question']
        
        wrong_answers = []
        
        # Extract numerical value from correct answer
        import re
        numbers = re.findall(r'-?\d+\.?\d*', correct_answer)
        if numbers:
            correct_value = float(numbers[0])
            unit = correct_answer.replace(numbers[0], '').strip()
            
            # Generate wrong answers by modifying the correct value
            for i in range(num_wrong):
                if question_type == 'kinematics_velocity':
                    # Wrong velocity calculations
                    multiplier = random.choice([0.5, 1.5, 2.0, 0.8])
                    wrong_value = correct_value * multiplier
                elif question_type == 'kinematics_acceleration':
                    # Wrong acceleration calculations
                    multiplier = random.choice([0.5, 1.5, 2.0, 0.7])
                    wrong_value = correct_value * multiplier
                elif question_type == 'force_newton_second':
                    # Wrong force calculations
                    multiplier = random.choice([0.5, 1.5, 2.0, 0.6])
                    wrong_value = correct_value * multiplier
                elif question_type in ['kinetic_energy', 'potential_energy', 'work_energy']:
                    # Wrong energy calculations
                    multiplier = random.choice([0.5, 1.5, 2.0, 0.4])
                    wrong_value = correct_value * multiplier
                else:
                    # Generic wrong answers
                    multiplier = random.choice([0.5, 1.5, 2.0, 0.8])
                    wrong_value = correct_value * multiplier
                
                wrong_answer = f"{wrong_value:.2f} {unit}"
                wrong_answers.append(wrong_answer)
        else:
            # For non-numerical answers, generate generic wrong answers
            generic_wrong = ["Option A", "Option B", "Option C", "Option D"]
            wrong_answers = generic_wrong[:num_wrong]
        
        return wrong_answers


def train_rl_agent(episodes: int = 1000, save_interval: int = 100):
    """Train the RL agent on physics questions."""
    # Initialize components
    question_generator = PhysicsQuestionGenerator()
    environment = PhysicsRLEnvironment(question_generator)
    agent = QLearningAgent()
    answer_generator = PhysicsAnswerGenerator()
    
    print("Starting RL Training on Physics Questions")
    print("=" * 50)
    
    for episode in range(episodes):
        # Reset environment
        state = environment.reset()
        
        # Generate answer options
        answer_options = answer_generator.generate_answer_options(environment.current_question)
        
        # Agent chooses action
        action = agent.get_action(state, answer_options)
        
        # Environment processes action
        next_state, reward, done, info = environment.step(action)
        
        # Update Q-values
        agent.update_q_value(state, action, reward, next_state, done)
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record training statistics
        if episode % 10 == 0:
            accuracy = environment.get_accuracy()
            agent.training_accuracy.append(accuracy)
            agent.training_rewards.append(environment.episode_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode:4d} | Accuracy: {accuracy:.3f} | Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"physics_rl_model_episode_{episode}.pkl")
    
    # Final save
    agent.save_model("physics_rl_model_final.pkl")
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {environment.get_accuracy():.3f}")
    print(f"Total questions answered: {environment.total_answers}")
    print(f"Correct answers: {environment.correct_answers}")
    
    return agent, environment


def evaluate_agent(agent: QLearningAgent, num_questions: int = 100):
    """Evaluate the trained agent on new questions."""
    question_generator = PhysicsQuestionGenerator()
    environment = PhysicsRLEnvironment(question_generator)
    answer_generator = PhysicsAnswerGenerator()
    
    print(f"\nEvaluating agent on {num_questions} new questions...")
    print("=" * 50)
    
    # Reset statistics
    environment.correct_answers = 0
    environment.total_answers = 0
    
    # Set agent to exploitation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for i in range(num_questions):
        state = environment.reset()
        answer_options = answer_generator.generate_answer_options(environment.current_question)
        action = agent.get_action(state, answer_options)
        next_state, reward, done, info = environment.step(action)
        
        if i < 5:  # Show first 5 questions
            print(f"\nQuestion {i+1}: {environment.current_question['question']}")
            print(f"Correct Answer: {environment.current_question['answer']}")
            print(f"Agent Answer: {action}")
            print(f"Correct: {info['correct']}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    final_accuracy = environment.get_accuracy()
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {final_accuracy:.3f}")
    print(f"Correct: {environment.correct_answers}/{environment.total_answers}")
    
    return final_accuracy


if __name__ == "__main__":
    # Train the agent
    agent, env = train_rl_agent(episodes=100_000)
    
    # Evaluate the agent
    evaluate_agent(agent, num_questions=50)
