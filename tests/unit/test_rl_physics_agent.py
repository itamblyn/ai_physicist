"""
Unit tests for RL Physics Agent components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from rl_physics_agent import PhysicsRLEnvironment, QLearningAgent, PhysicsAnswerGenerator
from physics_question_generator import PhysicsQuestionGenerator


class TestPhysicsRLEnvironment:
    """Test cases for PhysicsRLEnvironment class."""
    
    def test_init(self):
        """Test environment initialization."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        assert env.question_generator is generator
        assert env.current_question is None
        assert env.current_state is None
        assert env.episode_reward == 0
        assert env.episode_count == 0
        assert env.correct_answers == 0
        assert env.total_answers == 0
    
    def test_reset(self):
        """Test environment reset."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        state = env.reset()
        
        assert env.current_question is not None
        assert env.current_state is not None
        assert isinstance(state, str)
        assert env.episode_reward == 0
        
        # Check question structure
        assert 'question' in env.current_question
        assert 'answer' in env.current_question
        assert 'type' in env.current_question
    
    def test_step_correct_answer(self):
        """Test step with correct answer."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        # Reset to get a question
        state = env.reset()
        correct_answer = env.current_question['answer']
        
        # Step with correct answer
        next_state, reward, done, info = env.step(correct_answer)
        
        assert reward == 1.0
        assert done is True
        assert info['correct'] is True
        assert info['question_type'] == env.current_question['type']
        assert env.correct_answers == 1
        assert env.total_answers == 1
        assert env.episode_reward == 1.0
    
    def test_step_incorrect_answer(self):
        """Test step with incorrect answer."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        # Reset to get a question
        state = env.reset()
        wrong_answer = "999.99 m/s"  # Obviously wrong
        
        # Step with wrong answer
        next_state, reward, done, info = env.step(wrong_answer)
        
        assert reward == -0.1
        assert done is True
        assert info['correct'] is False
        assert env.correct_answers == 0
        assert env.total_answers == 1
        assert env.episode_reward == -0.1
    
    def test_check_answer_numerical(self):
        """Test answer checking with numerical values."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        # Create a mock question with numerical answer
        env.current_question = {
            'answer': '10.5 m/s',
            'type': 'kinematics_velocity'
        }
        
        # Test exact match
        assert env._check_answer('10.5 m/s') is True
        
        # Test with different formatting
        assert env._check_answer('10.50 m/s') is True
        assert env._check_answer('10.5 m/s ') is True
        assert env._check_answer(' 10.5 m/s') is True
        
        # Test with tolerance
        assert env._check_answer('10.51 m/s') is True  # Within tolerance
        assert env._check_answer('10.49 m/s') is True  # Within tolerance
        assert env._check_answer('10.6 m/s') is False  # Outside tolerance
    
    def test_check_answer_string(self):
        """Test answer checking with string values."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        # Create a mock question with string answer
        env.current_question = {
            'answer': 'inelastic',
            'type': 'collision'
        }
        
        # Test exact match
        assert env._check_answer('inelastic') is True
        assert env._check_answer('Inelastic') is True
        assert env._check_answer('INELASTIC') is True
        assert env._check_answer('elastic') is False
    
    def test_encode_state(self):
        """Test state encoding."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        question_data = {
            'question': 'A car travels 100 meters in 10 seconds. What is the velocity?',
            'type': 'kinematics_velocity'
        }
        
        state = env._encode_state(question_data)
        
        assert isinstance(state, str)
        assert 'kinematics_velocity' in state
        assert 'velocity' in state or 'distance' in state or 'time' in state
    
    def test_get_accuracy(self):
        """Test accuracy calculation."""
        generator = PhysicsQuestionGenerator()
        env = PhysicsRLEnvironment(generator)
        
        # Initially no answers
        assert env.get_accuracy() == 0.0
        
        # Add some correct and incorrect answers
        env.correct_answers = 3
        env.total_answers = 5
        assert env.get_accuracy() == 0.6
        
        # All correct
        env.correct_answers = 5
        env.total_answers = 5
        assert env.get_accuracy() == 1.0


class TestQLearningAgent:
    """Test cases for QLearningAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = QLearningAgent()
        
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.9
        assert agent.epsilon == 0.1
        assert agent.epsilon_decay == 0.995
        assert agent.epsilon_min == 0.01
        assert isinstance(agent.q_table, dict)
        assert isinstance(agent.action_space, set)
        assert len(agent.training_rewards) == 0
        assert len(agent.training_accuracy) == 0
    
    def test_init_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = QLearningAgent(
            learning_rate=0.2,
            discount_factor=0.8,
            epsilon=0.2,
            epsilon_decay=0.99,
            epsilon_min=0.05
        )
        
        assert agent.learning_rate == 0.2
        assert agent.discount_factor == 0.8
        assert agent.epsilon == 0.2
        assert agent.epsilon_decay == 0.99
        assert agent.epsilon_min == 0.05
    
    def test_get_action_explore(self):
        """Test action selection in exploration mode."""
        agent = QLearningAgent(epsilon=1.0)  # Always explore
        available_actions = ['10.0 m/s', '20.0 m/s', '30.0 m/s']
        
        # Test multiple times to ensure randomness
        actions = [agent.get_action('test_state', available_actions) for _ in range(10)]
        
        # Should get different actions due to exploration
        assert len(set(actions)) > 1
        # All actions should be from available actions
        assert all(action in available_actions for action in actions)
    
    def test_get_action_exploit(self):
        """Test action selection in exploitation mode."""
        agent = QLearningAgent(epsilon=0.0)  # Never explore
        available_actions = ['10.0 m/s', '20.0 m/s', '30.0 m/s']
        
        # Set up Q-values
        agent.q_table['test_state']['10.0 m/s'] = 0.5
        agent.q_table['test_state']['20.0 m/s'] = 1.0  # Highest
        agent.q_table['test_state']['30.0 m/s'] = 0.3
        
        # Should always choose the best action
        action = agent.get_action('test_state', available_actions)
        assert action == '20.0 m/s'
    
    def test_update_q_value_terminal(self):
        """Test Q-value update for terminal state."""
        agent = QLearningAgent(learning_rate=0.1)
        
        # Initial Q-value
        agent.q_table['state1']['action1'] = 0.0
        
        # Update with terminal state
        agent.update_q_value('state1', 'action1', 1.0, 'state2', True)
        
        # Q-value should be updated
        assert agent.q_table['state1']['action1'] > 0.0
    
    def test_update_q_value_non_terminal(self):
        """Test Q-value update for non-terminal state."""
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
        
        # Set up Q-values
        agent.q_table['state1']['action1'] = 0.0
        agent.q_table['state2']['action2'] = 0.5
        agent.q_table['state2']['action3'] = 0.8  # Max Q-value for next state
        
        # Update with non-terminal state
        agent.update_q_value('state1', 'action1', 0.5, 'state2', False)
        
        # Q-value should be updated
        assert agent.q_table['state1']['action1'] > 0.0
    
    def test_decay_epsilon(self):
        """Test epsilon decay."""
        agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1)
        
        # Decay epsilon
        agent.decay_epsilon()
        assert agent.epsilon == 0.45  # 0.5 * 0.9
        
        # Decay multiple times
        for _ in range(10):
            agent.decay_epsilon()
        
        # Should not go below minimum
        assert agent.epsilon >= agent.epsilon_min
    
    def test_save_load_model(self, temp_dir):
        """Test model saving and loading."""
        agent = QLearningAgent()
        
        # Set up some data
        agent.q_table['state1']['action1'] = 0.5
        agent.q_table['state2']['action2'] = 0.8
        agent.action_space.add('action1')
        agent.action_space.add('action2')
        agent.epsilon = 0.05
        
        # Save model
        model_path = temp_dir / "test_model.pkl"
        agent.save_model(str(model_path))
        
        # Create new agent and load model
        new_agent = QLearningAgent()
        new_agent.load_model(str(model_path))
        
        # Check that data was loaded correctly
        assert new_agent.q_table['state1']['action1'] == 0.5
        assert new_agent.q_table['state2']['action2'] == 0.8
        assert 'action1' in new_agent.action_space
        assert 'action2' in new_agent.action_space
        assert new_agent.epsilon == 0.05


class TestPhysicsAnswerGenerator:
    """Test cases for PhysicsAnswerGenerator class."""
    
    def test_init(self):
        """Test answer generator initialization."""
        generator = PhysicsAnswerGenerator()
        
        assert hasattr(generator, 'question_generator')
        assert isinstance(generator.question_generator, PhysicsQuestionGenerator)
    
    def test_generate_answer_options(self):
        """Test answer options generation."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': '10.0 m/s',
            'type': 'kinematics_velocity'
        }
        
        options = generator.generate_answer_options(question_data, num_options=4)
        
        assert len(options) == 4
        assert '10.0 m/s' in options  # Correct answer should be included
        assert all(isinstance(option, str) for option in options)
        assert all(len(option) > 0 for option in options)
    
    def test_generate_wrong_answers_kinematics(self):
        """Test wrong answer generation for kinematics."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': '10.0 m/s',
            'type': 'kinematics_velocity',
            'question': 'A car travels 100 meters in 10 seconds. What is the velocity?'
        }
        
        wrong_answers = generator._generate_wrong_answers(question_data, 3)
        
        assert len(wrong_answers) == 3
        assert all('m/s' in answer for answer in wrong_answers)
        assert all(answer != '10.0 m/s' for answer in wrong_answers)
    
    def test_generate_wrong_answers_force(self):
        """Test wrong answer generation for force problems."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': '50.0 N',
            'type': 'force_newton_second',
            'question': 'A 10 kg object accelerates at 5 m/s². What is the force?'
        }
        
        wrong_answers = generator._generate_wrong_answers(question_data, 3)
        
        assert len(wrong_answers) == 3
        assert all('N' in answer for answer in wrong_answers)
        assert all(answer != '50.0 N' for answer in wrong_answers)
    
    def test_generate_wrong_answers_energy(self):
        """Test wrong answer generation for energy problems."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': '100.0 J',
            'type': 'kinetic_energy',
            'question': 'A 2 kg object moves at 10 m/s. What is its kinetic energy?'
        }
        
        wrong_answers = generator._generate_wrong_answers(question_data, 3)
        
        assert len(wrong_answers) == 3
        assert all('J' in answer for answer in wrong_answers)
        assert all(answer != '100.0 J' for answer in wrong_answers)
    
    def test_generate_wrong_answers_non_numerical(self):
        """Test wrong answer generation for non-numerical answers."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': 'elastic',
            'type': 'collision',
            'question': 'What type of collision is this?'
        }
        
        wrong_answers = generator._generate_wrong_answers(question_data, 3)
        
        assert len(wrong_answers) == 3
        assert all(answer != 'elastic' for answer in wrong_answers)
        # Should fall back to generic options
        assert any('Option' in answer for answer in wrong_answers)
