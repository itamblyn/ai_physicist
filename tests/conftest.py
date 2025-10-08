"""
Pytest configuration and shared fixtures for AI Physicist tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add module paths
sys.path.insert(0, str(project_root / "01_generate_questions"))
sys.path.insert(0, str(project_root / "02_baseline"))
sys.path.insert(0, str(project_root / "03_extraneous_info_dataset"))
sys.path.insert(0, str(project_root / "04_unsolvable"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_question_data():
    """Sample question data for testing."""
    return {
        'question': 'A car travels 100 meters in 10 seconds at constant speed. What is the car\'s velocity?',
        'solution': 'Solution:\nUsing the formula: velocity = distance / time\nv = 100 m / 10 s = 10.00 m/s\n\nAnswer: 10.00 m/s',
        'answer': '10.00 m/s',
        'type': 'kinematics_velocity'
    }


@pytest.fixture
def sample_physics_problem():
    """Sample physics problem for testing."""
    return {
        'category': 'kinematics',
        'prompt': 'A runner moves with uniform acceleration. Initially, their speed is 5.0 m/s. Their acceleration is 2.0 m/s^2 for 3.0 s. What is the displacement?',
        'important_facts': [
            'Initial speed u = 5.0 m/s',
            'Acceleration a = 2.0 m/s^2',
            'Time t = 3.0 s',
            'Use s = ut + 1/2 a t^2'
        ],
        'extraneous_facts': [
            'The day is sunny.',
            'The runner\'s shoe size is 10.',
            'The track is painted blue.'
        ],
        'solution_steps': [
            'Identify given values u, a, t and the equation s = ut + 1/2 a t^2.',
            'Compute ut = 5.0 * 3.0 = 15.0 m.',
            'Compute 1/2 a t^2 = 0.5 * 2.0 * 3.0^2 = 9.0 m.',
            'Sum to get s = 24.0 m.'
        ],
        'final_answer': '24.0 m',
        'metadata': {'equation': 's = ut + 1/2 a t^2'}
    }


@pytest.fixture
def mock_openai_api(monkeypatch):
    """Mock OpenAI API responses for testing."""
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockResponse(content)
    
    class MockCompletion:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    def mock_create(*args, **kwargs):
        return MockCompletion("10.0 m/s")
    
    monkeypatch.setattr("openai.OpenAI.chat.completions.create", mock_create)
    return mock_create


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_jsonl_data():
    """Sample JSONL data for testing."""
    return [
        {"category": "kinematics", "prompt": "Test question 1", "answer": "10.0 m/s"},
        {"category": "newton", "prompt": "Test question 2", "answer": "5.0 N"},
        {"category": "energy", "prompt": "Test question 3", "answer": "100.0 J"}
    ]
