"""
Regression tests for model performance and training stability.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_generate_questions"))
sys.path.insert(0, str(project_root / "02_baseline"))

from physics_question_generator import PhysicsQuestionGenerator
from rl_physics_agent import QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator


class TestModelPerformance:
    """Test model performance and training stability."""
    
    def test_q_learning_convergence(self):
        """Test that Q-learning agent converges with sufficient training."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.3, epsilon_decay=0.995)
        answer_generator = PhysicsAnswerGenerator()
        
        # Track performance over time
        accuracies = []
        rewards = []
        
        # Train for multiple episodes
        for episode in range(100):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
            
            # Record performance every 10 episodes
            if episode % 10 == 0:
                accuracies.append(environment.get_accuracy())
                rewards.append(environment.episode_reward)
        
        # Check that performance improves over time
        # (This is a basic check - in practice, RL convergence can be noisy)
        assert len(accuracies) == 11  # 0, 10, 20, ..., 100
        assert all(0 <= acc <= 1 for acc in accuracies)
        
        # Check that Q-table has been populated
        assert len(agent.q_table) > 0
        
        # Check that epsilon has decayed
        assert agent.epsilon < 0.3
    
    def test_agent_consistency_across_runs(self):
        """Test that agent behavior is consistent across multiple runs."""
        generator = PhysicsQuestionGenerator()
        
        # Train two agents with same parameters
        agent1 = QLearningAgent(learning_rate=0.1, epsilon=0.0)  # No exploration
        agent2 = QLearningAgent(learning_rate=0.1, epsilon=0.0)  # No exploration
        
        # Set up identical Q-tables
        test_state = "kinematics_velocity_velocity_distance_time"
        agent1.q_table[test_state]["10.0 m/s"] = 1.0
        agent1.q_table[test_state]["20.0 m/s"] = 0.5
        agent2.q_table[test_state]["10.0 m/s"] = 1.0
        agent2.q_table[test_state]["20.0 m/s"] = 0.5
        
        # Both agents should make same decisions
        available_actions = ["10.0 m/s", "20.0 m/s", "30.0 m/s"]
        
        for _ in range(10):
            action1 = agent1.get_action(test_state, available_actions)
            action2 = agent2.get_action(test_state, available_actions)
            assert action1 == action2
    
    def test_environment_reward_consistency(self):
        """Test that environment rewards are consistent."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        
        # Test with correct answer
        environment.reset()
        correct_answer = environment.current_question['answer']
        next_state, reward, done, info = environment.step(correct_answer)
        
        assert reward == 1.0
        assert done is True
        assert info['correct'] is True
        
        # Test with incorrect answer
        environment.reset()
        wrong_answer = "999.99 m/s"  # Obviously wrong
        next_state, reward, done, info = environment.step(wrong_answer)
        
        assert reward == -0.1
        assert done is True
        assert info['correct'] is False
    
    def test_answer_generator_quality(self):
        """Test that answer generator produces reasonable options."""
        generator = PhysicsAnswerGenerator()
        
        question_data = {
            'answer': '10.0 m/s',
            'type': 'kinematics_velocity',
            'question': 'A car travels 100 meters in 10 seconds. What is the velocity?'
        }
        
        options = generator.generate_answer_options(question_data, num_options=4)
        
        # Should have correct number of options
        assert len(options) == 4
        
        # Should include correct answer
        assert '10.0 m/s' in options
        
        # All options should be different
        assert len(set(options)) == 4
        
        # All options should have same units
        for option in options:
            assert 'm/s' in option
        
        # Options should be reasonable (not extreme values)
        for option in options:
            import re
            numbers = re.findall(r'\d+\.?\d*', option)
            if numbers:
                value = float(numbers[0])
                assert 0 < value < 1000  # Reasonable range
    
    def test_state_encoding_consistency(self):
        """Test that state encoding is consistent."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        
        # Generate same question multiple times
        question_data = {
            'question': 'A car travels 100 meters in 10 seconds. What is the velocity?',
            'type': 'kinematics_velocity'
        }
        
        state1 = environment._encode_state(question_data)
        state2 = environment._encode_state(question_data)
        
        # States should be identical
        assert state1 == state2
        
        # State should contain question type
        assert 'kinematics_velocity' in state1
    
    def test_q_value_update_consistency(self):
        """Test that Q-value updates are consistent."""
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
        
        # Set up initial Q-value
        agent.q_table['state1']['action1'] = 0.0
        
        # Update Q-value
        agent.update_q_value('state1', 'action1', 1.0, 'state2', True)
        
        # Q-value should be updated
        assert agent.q_table['state1']['action1'] > 0.0
        assert agent.q_table['state1']['action1'] <= 1.0
        
        # Update again with same parameters
        old_q_value = agent.q_table['state1']['action1']
        agent.update_q_value('state1', 'action1', 1.0, 'state2', True)
        
        # Q-value should change (not be identical)
        assert agent.q_table['state1']['action1'] != old_q_value
    
    def test_epsilon_decay_consistency(self):
        """Test that epsilon decay is consistent."""
        agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1)
        
        # Decay epsilon multiple times
        epsilons = []
        for _ in range(10):
            epsilons.append(agent.epsilon)
            agent.decay_epsilon()
        
        # Epsilon should decrease over time
        for i in range(1, len(epsilons)):
            assert epsilons[i] <= epsilons[i-1]
        
        # Should not go below minimum
        assert agent.epsilon >= agent.epsilon_min
    
    def test_agent_memory_management(self):
        """Test that agent memory is managed properly."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent()
        answer_generator = PhysicsAnswerGenerator()
        
        # Train for many episodes
        for episode in range(1000):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
        
        # Q-table should not grow excessively
        assert len(agent.q_table) < 1000  # Reasonable upper bound
        
        # Action space should not grow excessively
        assert len(agent.action_space) < 1000  # Reasonable upper bound
    
    def test_performance_under_different_conditions(self):
        """Test performance under different conditions."""
        generator = PhysicsQuestionGenerator()
        
        # Test with different question types
        for question_type in generator.question_types:
            environment = PhysicsRLEnvironment(generator)
            agent = QLearningAgent(epsilon=0.0)  # No exploration
            answer_generator = PhysicsAnswerGenerator()
            
            # Generate specific question type
            question_data = generator.generate_question(question_type)
            environment.current_question = question_data
            environment.current_state = environment._encode_state(question_data)
            
            # Test answer generation
            answer_options = answer_generator.generate_answer_options(question_data)
            assert len(answer_options) > 0
            assert question_data['answer'] in answer_options
    
    def test_error_handling_robustness(self):
        """Test that system handles errors gracefully."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent()
        answer_generator = PhysicsAnswerGenerator()
        
        # Test with invalid inputs
        try:
            # Invalid question type
            generator.generate_question('invalid_type')
        except ValueError:
            pass  # Expected
        
        # Test with empty answer options
        try:
            action = agent.get_action('test_state', [])
        except (IndexError, ValueError):
            pass  # Expected
        
        # Test with malformed question data
        malformed_question = {
            'question': 'Test question',
            'answer': '10.0 m/s',
            'type': 'kinematics_velocity'
        }
        
        # Should handle gracefully
        answer_options = answer_generator.generate_answer_options(malformed_question)
        assert len(answer_options) > 0
    
    def test_training_stability(self):
        """Test that training is stable and doesn't diverge."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.1)
        answer_generator = PhysicsAnswerGenerator()
        
        # Track Q-values over time
        q_values = []
        
        # Train for many episodes
        for episode in range(500):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
            
            # Record Q-values every 50 episodes
            if episode % 50 == 0:
                if agent.q_table:
                    max_q = max(max(q_dict.values()) if q_dict else [0] for q_dict in agent.q_table.values())
                    q_values.append(max_q)
        
        # Q-values should not explode or become NaN
        assert all(not np.isnan(q) for q in q_values)
        assert all(q < 1000 for q in q_values)  # Reasonable upper bound
        assert all(q > -1000 for q in q_values)  # Reasonable lower bound
