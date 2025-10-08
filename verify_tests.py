#!/usr/bin/env python3
"""
Simple test verification script for AI Physicist testing framework.
This script verifies that the testing framework is properly set up.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_generate_questions"))
sys.path.insert(0, str(project_root / "02_baseline"))
sys.path.insert(0, str(project_root / "03_extraneous_info_dataset"))
sys.path.insert(0, str(project_root / "04_unsolvable"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from physics_question_generator import PhysicsQuestionGenerator
        print("✅ PhysicsQuestionGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PhysicsQuestionGenerator: {e}")
        return False
    
    try:
        from rl_physics_agent import QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
        print("✅ RL agent modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RL agent modules: {e}")
        return False
    
    try:
        from llm_rl_agent import LLMRLAgent, LLMRLEnvironment
        print("✅ LLM agent modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LLM agent modules: {e}")
        return False
    
    try:
        from generate_extraneous_dataset import generate_dataset as generate_extraneous_dataset
        print("✅ Extraneous dataset generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import extraneous dataset generator: {e}")
        return False
    
    try:
        from generate_inconsistent_dataset import generate_dataset as generate_inconsistent_dataset
        print("✅ Inconsistent dataset generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import inconsistent dataset generator: {e}")
        return False
    
    return True


def test_physics_question_generator():
    """Test physics question generator functionality."""
    print("\nTesting PhysicsQuestionGenerator...")
    
    try:
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        
        # Test initialization
        assert hasattr(generator, 'question_types')
        assert len(generator.question_types) > 0
        print("✅ Generator initialization successful")
        
        # Test question generation
        question_data = generator.generate_question()
        assert 'question' in question_data
        assert 'answer' in question_data
        assert 'type' in question_data
        print("✅ Question generation successful")
        
        # Test specific question type
        question_data = generator.generate_question('kinematics_velocity')
        assert question_data['type'] == 'kinematics_velocity'
        print("✅ Specific question type generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ PhysicsQuestionGenerator test failed: {e}")
        return False


def test_rl_agent():
    """Test RL agent functionality."""
    print("\nTesting RL Agent...")
    
    try:
        from rl_physics_agent import QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
        from physics_question_generator import PhysicsQuestionGenerator
        
        # Test agent initialization
        agent = QLearningAgent()
        assert hasattr(agent, 'q_table')
        assert hasattr(agent, 'epsilon')
        print("✅ QLearningAgent initialization successful")
        
        # Test environment initialization
        generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(generator)
        assert hasattr(environment, 'question_generator')
        print("✅ PhysicsRLEnvironment initialization successful")
        
        # Test answer generator
        answer_generator = PhysicsAnswerGenerator()
        assert hasattr(answer_generator, 'question_generator')
        print("✅ PhysicsAnswerGenerator initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ RL Agent test failed: {e}")
        return False


def test_dataset_generators():
    """Test dataset generator functionality."""
    print("\nTesting Dataset Generators...")
    
    try:
        from generate_extraneous_dataset import generate_dataset as generate_extraneous_dataset
        from generate_inconsistent_dataset import generate_dataset as generate_inconsistent_dataset
        
        # Test extraneous dataset generation
        supervised, preference = generate_extraneous_dataset(5, seed=42)
        assert len(supervised) == 5
        assert len(preference) == 5
        print("✅ Extraneous dataset generation successful")
        
        # Test inconsistent dataset generation
        records = generate_inconsistent_dataset(5, seed=42)
        assert len(records) == 5
        print("✅ Inconsistent dataset generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset generator test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/unit/__init__.py",
        "tests/unit/test_physics_question_generator.py",
        "tests/unit/test_rl_physics_agent.py",
        "tests/unit/test_llm_rl_agent.py",
        "tests/unit/test_dataset_generators.py",
        "tests/integration/__init__.py",
        "tests/integration/test_end_to_end_workflows.py",
        "tests/regression/__init__.py",
        "tests/regression/test_data_consistency.py",
        "tests/regression/test_model_performance.py",
        "tests/data/sample_questions.json",
        "tests/data/sample_extraneous.jsonl",
        "pytest.ini",
        "TEST.md",
        "run_tests.py",
        ".github/workflows/test.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True


def main():
    """Main verification function."""
    print("AI Physicist Testing Framework Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_physics_question_generator,
        test_rl_agent,
        test_dataset_generators,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The testing framework is properly set up.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
