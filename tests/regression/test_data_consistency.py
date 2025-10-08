"""
Regression tests for data consistency and reproducibility.
"""

import pytest
import hashlib
import json
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_generate_questions"))
sys.path.insert(0, str(project_root / "02_baseline"))
sys.path.insert(0, str(project_root / "03_extraneous_info_dataset"))
sys.path.insert(0, str(project_root / "04_unsolvable"))

from physics_question_generator import PhysicsQuestionGenerator
from generate_extraneous_dataset import generate_dataset as generate_extraneous_dataset
from generate_inconsistent_dataset import generate_dataset as generate_inconsistent_dataset


class TestDataConsistency:
    """Test data consistency and reproducibility."""
    
    def test_question_generation_reproducibility(self):
        """Test that question generation is reproducible with same seed."""
        generator = PhysicsQuestionGenerator()
        
        # Generate questions with fixed seed
        questions1 = []
        questions2 = []
        
        # Reset random state and generate
        import random
        random.seed(42)
        for _ in range(10):
            questions1.append(generator.generate_question())
        
        # Reset random state and generate again
        random.seed(42)
        for _ in range(10):
            questions2.append(generator.generate_question())
        
        # Questions should be identical
        for q1, q2 in zip(questions1, questions2):
            assert q1['question'] == q2['question']
            assert q1['answer'] == q2['answer']
            assert q1['type'] == q2['type']
    
    def test_extraneous_dataset_reproducibility(self):
        """Test that extraneous dataset generation is reproducible."""
        # Generate dataset twice with same seed
        supervised1, preference1 = generate_extraneous_dataset(20, seed=42)
        supervised2, preference2 = generate_extraneous_dataset(20, seed=42)
        
        # Datasets should be identical
        assert len(supervised1) == len(supervised2)
        assert len(preference1) == len(preference2)
        
        for s1, s2 in zip(supervised1, supervised2):
            assert s1['prompt'] == s2['prompt']
            assert s1['final_answer'] == s2['final_answer']
            assert s1['category'] == s2['category']
        
        for p1, p2 in zip(preference1, preference2):
            assert p1['prompt'] == p2['prompt']
            assert p1['chosen']['answer'] == p2['chosen']['answer']
            assert p1['rejected']['answer'] == p2['rejected']['answer']
    
    def test_inconsistent_dataset_reproducibility(self):
        """Test that inconsistent dataset generation is reproducible."""
        # Generate dataset twice with same seed
        records1 = generate_inconsistent_dataset(20, seed=42)
        records2 = generate_inconsistent_dataset(20, seed=42)
        
        # Datasets should be identical
        assert len(records1) == len(records2)
        
        for r1, r2 in zip(records1, records2):
            assert r1['prompt'] == r2['prompt']
            assert r1['rationale'] == r2['rationale']
            assert r1['category'] == r2['category']
            assert r1['label'] == r2['label']
    
    def test_question_answer_mathematical_consistency(self):
        """Test that generated questions have mathematically consistent answers."""
        generator = PhysicsQuestionGenerator()
        
        # Test multiple questions of each type
        for question_type in generator.question_types:
            for _ in range(5):
                question_data = generator.generate_question(question_type)
                self._verify_mathematical_consistency(question_data)
    
    def _verify_mathematical_consistency(self, question_data):
        """Verify that a question's answer is mathematically consistent."""
        import re
        
        question_text = question_data['question']
        answer = question_data['answer']
        question_type = question_data['type']
        
        # Extract numbers from question
        numbers = re.findall(r'\d+\.?\d*', question_text)
        numbers = [float(n) for n in numbers]
        
        if len(numbers) < 2:
            return  # Skip if not enough numbers
        
        # Extract answer value
        answer_match = re.search(r'(\d+\.?\d*)', answer)
        if not answer_match:
            return  # Skip if no number in answer
        
        answer_value = float(answer_match.group(1))
        
        # Verify based on question type
        if question_type == 'kinematics_velocity':
            if len(numbers) >= 2:
                distance, time = numbers[0], numbers[1]
                expected = distance / time
                assert abs(answer_value - expected) < 0.01
        
        elif question_type == 'kinematics_acceleration':
            if len(numbers) >= 3:
                initial_vel, final_vel, time = numbers[0], numbers[1], numbers[2]
                expected = (final_vel - initial_vel) / time
                assert abs(answer_value - expected) < 0.01
        
        elif question_type == 'force_newton_second':
            if len(numbers) >= 2:
                mass, acceleration = numbers[0], numbers[1]
                expected = mass * acceleration
                assert abs(answer_value - expected) < 0.01
        
        elif question_type == 'kinetic_energy':
            if len(numbers) >= 2:
                mass, velocity = numbers[0], numbers[1]
                expected = 0.5 * mass * velocity**2
                assert abs(answer_value - expected) < 0.01
        
        elif question_type == 'potential_energy':
            if len(numbers) >= 2:
                mass, height = numbers[0], numbers[1]
                g = 9.8
                expected = mass * g * height
                assert abs(answer_value - expected) < 0.01
        
        elif question_type == 'work_energy':
            if len(numbers) >= 2:
                force, distance = numbers[0], numbers[1]
                expected = force * distance
                assert abs(answer_value - expected) < 0.01
    
    def test_dataset_schema_consistency(self):
        """Test that generated datasets maintain consistent schema."""
        # Test extraneous dataset schema
        supervised, preference = generate_extraneous_dataset(10, seed=42)
        
        # Check supervised schema
        for record in supervised:
            required_fields = ['category', 'prompt', 'important_facts', 'extraneous_facts', 
                             'solution_steps', 'final_answer', 'metadata']
            for field in required_fields:
                assert field in record
                assert record[field] is not None
        
        # Check preference schema
        for record in preference:
            assert 'chosen' in record
            assert 'rejected' in record
            assert 'category' in record
            assert 'prompt' in record
            assert 'metadata' in record
            
            # Check chosen/rejected structure
            for response in ['chosen', 'rejected']:
                assert 'reasoning' in record[response]
                assert 'answer' in record[response]
                assert 'uses_extraneous' in record[response]
                assert 'correct' in record[response]
                assert isinstance(record[response]['uses_extraneous'], bool)
                assert isinstance(record[response]['correct'], bool)
        
        # Test inconsistent dataset schema
        inconsistent_records = generate_inconsistent_dataset(10, seed=42)
        
        for record in inconsistent_records:
            required_fields = ['category', 'prompt', 'important_facts', 'inconsistent_facts',
                             'inconsistencies', 'rationale', 'label', 'metadata']
            for field in required_fields:
                assert field in record
                assert record[field] is not None
            
            # Check inconsistencies structure
            for inconsistency in record['inconsistencies']:
                required_inconsistency_fields = ['type', 'field', 'value_a', 'value_b', 
                                               'units', 'description']
                for field in required_inconsistency_fields:
                    assert field in inconsistency
    
    def test_data_quality_metrics(self):
        """Test data quality metrics for generated datasets."""
        # Test extraneous dataset quality
        supervised, preference = generate_extraneous_dataset(50, seed=42)
        
        # Check that extraneous facts are actually extraneous
        for record in supervised:
            important_facts = record['important_facts']
            extraneous_facts = record['extraneous_facts']
            
            # Important facts should contain physics terms
            physics_terms = ['velocity', 'acceleration', 'force', 'energy', 'mass', 'distance', 'time']
            has_physics_terms = any(term in ' '.join(important_facts).lower() for term in physics_terms)
            assert has_physics_terms, "Important facts should contain physics terms"
            
            # Extraneous facts should not contain physics terms
            has_physics_in_extraneous = any(term in ' '.join(extraneous_facts).lower() for term in physics_terms)
            assert not has_physics_in_extraneous, "Extraneous facts should not contain physics terms"
        
        # Check that preference pairs are correctly labeled
        for record in preference:
            chosen = record['chosen']
            rejected = record['rejected']
            
            assert chosen['correct'] is True
            assert chosen['uses_extraneous'] is False
            assert rejected['correct'] is False
            assert rejected['uses_extraneous'] is True
        
        # Test inconsistent dataset quality
        inconsistent_records = generate_inconsistent_dataset(50, seed=42)
        
        # Check that all records are labeled as inconsistent
        for record in inconsistent_records:
            assert record['label'] == 'inconsistent'
            assert 'inconsistent' in record['rationale'].lower()
            assert len(record['inconsistencies']) > 0
    
    def test_dataset_size_consistency(self):
        """Test that dataset generation produces expected sizes."""
        # Test extraneous dataset
        for size in [10, 25, 50, 100]:
            supervised, preference = generate_extraneous_dataset(size, seed=42)
            assert len(supervised) == size
            assert len(preference) == size
        
        # Test inconsistent dataset
        for size in [10, 25, 50, 100]:
            records = generate_inconsistent_dataset(size, seed=42)
            assert len(records) == size
    
    def test_category_distribution(self):
        """Test that datasets have reasonable category distribution."""
        supervised, preference = generate_extraneous_dataset(100, seed=42)
        
        # Count categories
        categories = [record['category'] for record in supervised]
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Should have all expected categories
        expected_categories = ['kinematics', 'newton', 'energy', 'momentum', 'circuits']
        for category in expected_categories:
            assert category in category_counts
        
        # Should have reasonable distribution (not all in one category)
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        assert max_count - min_count < 50  # Not too skewed
    
    def test_answer_format_consistency(self):
        """Test that answers have consistent format."""
        generator = PhysicsQuestionGenerator()
        
        for question_type in generator.question_types:
            for _ in range(10):
                question_data = generator.generate_question(question_type)
                answer = question_data['answer']
                
                # Answer should contain a number
                import re
                numbers = re.findall(r'\d+\.?\d*', answer)
                assert len(numbers) > 0, f"Answer should contain a number: {answer}"
                
                # Answer should contain appropriate units
                if question_type in ['kinematics_velocity', 'kinematics_acceleration', 'simple_collision']:
                    assert 'm/s' in answer or 'm/s²' in answer
                elif question_type == 'force_newton_second':
                    assert 'N' in answer
                elif question_type in ['kinetic_energy', 'potential_energy', 'work_energy']:
                    assert 'J' in answer
                elif question_type == 'kinematics_displacement':
                    assert 'm' in answer
