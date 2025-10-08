"""
Unit tests for PhysicsQuestionGenerator.
"""

import pytest
import math
from physics_question_generator import PhysicsQuestionGenerator


class TestPhysicsQuestionGenerator:
    """Test cases for PhysicsQuestionGenerator class."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = PhysicsQuestionGenerator()
        assert hasattr(generator, 'question_types')
        assert len(generator.question_types) > 0
        assert 'kinematics_velocity' in generator.question_types
    
    def test_generate_question_random(self):
        """Test random question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_question()
        
        assert isinstance(question_data, dict)
        assert 'question' in question_data
        assert 'solution' in question_data
        assert 'answer' in question_data
        assert 'type' in question_data
        assert question_data['type'] in generator.question_types
    
    def test_generate_question_specific_type(self):
        """Test specific question type generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_question('kinematics_velocity')
        
        assert question_data['type'] == 'kinematics_velocity'
        assert 'velocity' in question_data['question'].lower()
        assert 'm/s' in question_data['answer']
    
    def test_generate_question_invalid_type(self):
        """Test invalid question type raises error."""
        generator = PhysicsQuestionGenerator()
        with pytest.raises(ValueError, match="Unknown question type"):
            generator.generate_question('invalid_type')
    
    def test_kinematics_velocity(self):
        """Test velocity calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_kinematics_velocity()
        
        assert question_data['type'] == 'kinematics_velocity'
        assert 'velocity' in question_data['question'].lower()
        assert 'm/s' in question_data['answer']
        
        # Check that the answer is numerically correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        distance = int(numbers[0])
        time = int(numbers[1])
        expected_velocity = distance / time
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_velocity) < 0.01
    
    def test_kinematics_acceleration(self):
        """Test acceleration calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_kinematics_acceleration()
        
        assert question_data['type'] == 'kinematics_acceleration'
        assert 'acceleration' in question_data['question'].lower()
        assert 'm/s²' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        initial_velocity = int(numbers[0])
        final_velocity = int(numbers[1])
        time = int(numbers[2])
        expected_acceleration = (final_velocity - initial_velocity) / time
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_acceleration) < 0.01
    
    def test_kinematics_displacement(self):
        """Test displacement calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_kinematics_displacement()
        
        assert question_data['type'] == 'kinematics_displacement'
        assert 'displacement' in question_data['question'].lower()
        assert 'm' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        initial_velocity = int(numbers[0])
        acceleration = int(numbers[1])
        time = int(numbers[2])
        expected_displacement = initial_velocity * time + 0.5 * acceleration * time**2
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_displacement) < 0.01
    
    def test_force_newton_second(self):
        """Test force calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_force_newton_second()
        
        assert question_data['type'] == 'force_newton_second'
        assert 'force' in question_data['question'].lower()
        assert 'N' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        mass = int(numbers[0])
        acceleration = int(numbers[1])
        expected_force = mass * acceleration
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_force) < 0.01
    
    def test_kinetic_energy(self):
        """Test kinetic energy calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_kinetic_energy()
        
        assert question_data['type'] == 'kinetic_energy'
        assert 'kinetic energy' in question_data['question'].lower()
        assert 'J' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        mass = int(numbers[0])
        velocity = int(numbers[1])
        expected_energy = 0.5 * mass * velocity**2
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_energy) < 0.01
    
    def test_potential_energy(self):
        """Test potential energy calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_potential_energy()
        
        assert question_data['type'] == 'potential_energy'
        assert 'potential energy' in question_data['question'].lower()
        assert 'J' in question_data['answer']
        assert '9.8' in question_data['question']  # Should include g value
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        mass = int(numbers[0])
        height = int(numbers[1])
        g = 9.8
        expected_energy = mass * g * height
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_energy) < 0.01
    
    def test_work_energy(self):
        """Test work calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_work_energy()
        
        assert question_data['type'] == 'work_energy'
        assert 'work' in question_data['question'].lower()
        assert 'J' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        force = int(numbers[0])
        distance = int(numbers[1])
        expected_work = force * distance
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_work) < 0.01
    
    def test_simple_collision(self):
        """Test collision calculation question generation."""
        generator = PhysicsQuestionGenerator()
        question_data = generator.generate_simple_collision()
        
        assert question_data['type'] == 'simple_collision'
        assert 'collision' in question_data['question'].lower()
        assert 'm/s' in question_data['answer']
        
        # Verify the calculation is correct
        import re
        numbers = re.findall(r'\d+', question_data['question'])
        m1 = int(numbers[0])
        m2 = int(numbers[1])
        v1_initial = int(numbers[2])
        v2_initial = 0  # stationary target
        
        # Elastic collision formula
        expected_v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
        
        answer_value = float(question_data['answer'].split()[0])
        assert abs(answer_value - expected_v1_final) < 0.01
    
    def test_question_structure(self):
        """Test that all question types have proper structure."""
        generator = PhysicsQuestionGenerator()
        
        for question_type in generator.question_types:
            question_data = generator.generate_question(question_type)
            
            # Check required fields
            assert 'question' in question_data
            assert 'solution' in question_data
            assert 'answer' in question_data
            assert 'type' in question_data
            
            # Check field types
            assert isinstance(question_data['question'], str)
            assert isinstance(question_data['solution'], str)
            assert isinstance(question_data['answer'], str)
            assert isinstance(question_data['type'], str)
            
            # Check non-empty fields
            assert len(question_data['question']) > 0
            assert len(question_data['solution']) > 0
            assert len(question_data['answer']) > 0
            assert len(question_data['type']) > 0
    
    def test_randomness(self):
        """Test that questions are different when generated multiple times."""
        generator = PhysicsQuestionGenerator()
        
        # Generate multiple questions of the same type
        questions = [generator.generate_kinematics_velocity() for _ in range(5)]
        
        # Check that questions are different (very unlikely to be identical)
        question_texts = [q['question'] for q in questions]
        assert len(set(question_texts)) > 1  # At least some should be different
        
        # Check that answers are different (very unlikely to be identical)
        answers = [q['answer'] for q in questions]
        assert len(set(answers)) > 1  # At least some should be different
