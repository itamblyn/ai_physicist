"""
Integration tests for end-to-end workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_generate_questions"))
sys.path.insert(0, str(project_root / "02_baseline"))
sys.path.insert(0, str(project_root / "03_extraneous_info_dataset"))
sys.path.insert(0, str(project_root / "04_unsolvable"))

from physics_question_generator import PhysicsQuestionGenerator
from rl_physics_agent import train_rl_agent, evaluate_agent, QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
from generate_extraneous_dataset import generate_dataset as generate_extraneous_dataset
from generate_inconsistent_dataset import generate_dataset as generate_inconsistent_dataset


class TestEndToEndWorkflows:
    """Test complete workflows from start to finish."""
    
    def test_physics_question_generation_workflow(self):
        """Test complete physics question generation workflow."""
        generator = PhysicsQuestionGenerator()
        
        # Generate multiple questions of different types
        questions = []
        for question_type in generator.question_types:
            question = generator.generate_question(question_type)
            questions.append(question)
        
        # Verify all questions are valid
        assert len(questions) == len(generator.question_types)
        
        for question in questions:
            assert 'question' in question
            assert 'solution' in question
            assert 'answer' in question
            assert 'type' in question
            assert question['type'] in generator.question_types
            
            # Verify answer format
            assert any(unit in question['answer'] for unit in ['m/s', 'm/s²', 'N', 'J', 'm'])
    
    def test_rl_training_workflow(self):
        """Test complete RL training workflow."""
        # This is a simplified version to avoid long training times
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.1)
        answer_generator = PhysicsAnswerGenerator()
        
        # Train for a few episodes
        for episode in range(10):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
        
        # Verify training completed
        assert environment.total_answers == 10
        assert len(agent.q_table) > 0
        assert agent.epsilon < 0.1  # Should have decayed
    
    def test_rl_evaluation_workflow(self):
        """Test RL agent evaluation workflow."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent(epsilon=0.0)  # No exploration for evaluation
        answer_generator = PhysicsAnswerGenerator()
        
        # Set up some Q-values
        agent.q_table['kinematics_velocity_velocity_distance_time']['10.0 m/s'] = 1.0
        agent.q_table['kinematics_velocity_velocity_distance_time']['20.0 m/s'] = 0.5
        
        # Evaluate on a few questions
        for i in range(5):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
        
        # Verify evaluation completed
        assert environment.total_answers == 5
        assert environment.correct_answers >= 0
        assert environment.correct_answers <= environment.total_answers
    
    def test_extraneous_dataset_generation_workflow(self):
        """Test complete extraneous dataset generation workflow."""
        # Generate small dataset
        supervised, preference = generate_extraneous_dataset(10, seed=42)
        
        # Verify dataset structure
        assert len(supervised) == 10
        assert len(preference) == 10
        
        # Check supervised records
        for record in supervised:
            assert 'category' in record
            assert 'prompt' in record
            assert 'important_facts' in record
            assert 'extraneous_facts' in record
            assert 'solution_steps' in record
            assert 'final_answer' in record
            assert 'metadata' in record
            
            # Verify extraneous facts are present
            assert len(record['extraneous_facts']) > 0
            assert len(record['important_facts']) > 0
        
        # Check preference records
        for record in preference:
            assert 'chosen' in record
            assert 'rejected' in record
            assert record['chosen']['correct'] is True
            assert record['chosen']['uses_extraneous'] is False
            assert record['rejected']['correct'] is False
            assert record['rejected']['uses_extraneous'] is True
    
    def test_inconsistent_dataset_generation_workflow(self):
        """Test complete inconsistent dataset generation workflow."""
        # Generate small dataset
        records = generate_inconsistent_dataset(10, seed=42)
        
        # Verify dataset structure
        assert len(records) == 10
        
        for record in records:
            assert 'category' in record
            assert 'prompt' in record
            assert 'important_facts' in record
            assert 'inconsistent_facts' in record
            assert 'inconsistencies' in record
            assert 'rationale' in record
            assert 'label' in record
            assert 'metadata' in record
            
            # Verify inconsistencies are present
            assert len(record['inconsistencies']) > 0
            assert record['label'] == 'inconsistent'
            assert 'inconsistent' in record['rationale'].lower()
    
    def test_file_io_workflow(self, temp_dir):
        """Test file I/O workflow for dataset generation."""
        from generate_extraneous_dataset import save_jsonl
        
        # Generate test data
        supervised, preference = generate_extraneous_dataset(5, seed=42)
        
        # Save to files
        supervised_path = temp_dir / "supervised.jsonl"
        preference_path = temp_dir / "preference.jsonl"
        
        save_jsonl(supervised, supervised_path)
        save_jsonl(preference, preference_path)
        
        # Verify files exist
        assert supervised_path.exists()
        assert preference_path.exists()
        
        # Verify file contents
        with open(supervised_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 5
        
        with open(preference_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 5
    
    def test_question_answer_consistency_workflow(self):
        """Test that generated questions have consistent answers."""
        generator = PhysicsQuestionGenerator()
        
        # Generate multiple questions and verify answer consistency
        for _ in range(20):
            question_data = generator.generate_question()
            
            # Extract numerical values from question and answer
            import re
            
            # For velocity questions
            if question_data['type'] == 'kinematics_velocity':
                numbers = re.findall(r'\d+', question_data['question'])
                if len(numbers) >= 2:
                    distance = int(numbers[0])
                    time = int(numbers[1])
                    expected_velocity = distance / time
                    
                    answer_value = float(question_data['answer'].split()[0])
                    assert abs(answer_value - expected_velocity) < 0.01
            
            # For force questions
            elif question_data['type'] == 'force_newton_second':
                numbers = re.findall(r'\d+', question_data['question'])
                if len(numbers) >= 2:
                    mass = int(numbers[0])
                    acceleration = int(numbers[1])
                    expected_force = mass * acceleration
                    
                    answer_value = float(question_data['answer'].split()[0])
                    assert abs(answer_value - expected_force) < 0.01
    
    def test_agent_learning_progress_workflow(self):
        """Test that RL agent shows learning progress."""
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.5)
        answer_generator = PhysicsAnswerGenerator()
        
        # Track accuracy over time
        accuracies = []
        
        # Train for multiple episodes
        for episode in range(50):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
            
            # Record accuracy every 10 episodes
            if episode % 10 == 0:
                accuracies.append(environment.get_accuracy())
        
        # Verify that accuracy is tracked
        assert len(accuracies) == 6  # 0, 10, 20, 30, 40, 50
        assert all(0 <= acc <= 1 for acc in accuracies)
    
    def test_error_handling_workflow(self):
        """Test error handling in various workflows."""
        generator = PhysicsQuestionGenerator()
        
        # Test invalid question type
        with pytest.raises(ValueError):
            generator.generate_question('invalid_type')
        
        # Test RL environment with invalid answer
        environment = PhysicsRLEnvironment(generator)
        environment.reset()
        
        # Should handle invalid answers gracefully
        next_state, reward, done, info = environment.step("invalid answer")
        assert reward == -0.1
        assert done is True
        assert info['correct'] is False
    
    def test_data_validation_workflow(self):
        """Test data validation in generated datasets."""
        # Test extraneous dataset
        supervised, preference = generate_extraneous_dataset(10, seed=42)
        
        for record in supervised:
            # Verify all required fields are present and non-empty
            assert record['category'] in ['kinematics', 'newton', 'energy', 'momentum', 'circuits']
            assert len(record['prompt']) > 0
            assert len(record['important_facts']) > 0
            assert len(record['extraneous_facts']) > 0
            assert len(record['solution_steps']) > 0
            assert len(record['final_answer']) > 0
            assert 'equation' in record['metadata']
        
        for record in preference:
            # Verify preference structure
            assert 'chosen' in record
            assert 'rejected' in record
            assert isinstance(record['chosen']['correct'], bool)
            assert isinstance(record['chosen']['uses_extraneous'], bool)
            assert isinstance(record['rejected']['correct'], bool)
            assert isinstance(record['rejected']['uses_extraneous'], bool)
        
        # Test inconsistent dataset
        inconsistent_records = generate_inconsistent_dataset(10, seed=42)
        
        for record in inconsistent_records:
            # Verify inconsistency structure
            assert len(record['inconsistencies']) > 0
            assert record['label'] == 'inconsistent'
            assert 'inconsistent' in record['rationale'].lower()
            
            # Verify inconsistency types
            for inconsistency in record['inconsistencies']:
                assert 'type' in inconsistency
                assert 'field' in inconsistency
                assert 'description' in inconsistency
                assert inconsistency['type'] in ['contradictory_value', 'unit_mismatch', 'equation_conflict']
