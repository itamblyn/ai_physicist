"""
Unit tests for dataset generation modules.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "03_extraneous_info_dataset"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "04_unsolvable"))

from generate_extraneous_dataset import (
    PhysicsProblem, format_number, generate_kinematics_rng, 
    generate_newton_rng, generate_energy_rng, generate_momentum_rng,
    generate_circuit_rng, build_supervised_record, build_preference_pair,
    generate_dataset, save_jsonl
)

from generate_inconsistent_dataset import (
    InconsistentProblem, generate_kinematics_inconsistent,
    generate_newton_inconsistent, generate_energy_inconsistent,
    generate_momentum_inconsistent, generate_circuit_inconsistent,
    build_unsolvable_record, generate_dataset as generate_inconsistent_dataset
)


class TestExtraneousDatasetGeneration:
    """Test cases for extraneous dataset generation."""
    
    def test_physics_problem_dataclass(self):
        """Test PhysicsProblem dataclass."""
        problem = PhysicsProblem(
            category="kinematics",
            prompt="Test prompt",
            important_facts=["fact1", "fact2"],
            extraneous_facts=["ext1", "ext2"],
            solution_steps=["step1", "step2"],
            final_answer="10.0 m/s",
            metadata={"equation": "v = d/t"}
        )
        
        assert problem.category == "kinematics"
        assert problem.prompt == "Test prompt"
        assert len(problem.important_facts) == 2
        assert len(problem.extraneous_facts) == 2
        assert len(problem.solution_steps) == 2
        assert problem.final_answer == "10.0 m/s"
        assert problem.metadata["equation"] == "v = d/t"
    
    def test_format_number(self):
        """Test number formatting function."""
        # Test normal numbers
        assert format_number(123.456) == "123"
        assert format_number(0.001) == "0.00100e-03"
        assert format_number(10000) == "1.00e+04"
        
        # Test with custom digits
        assert format_number(123.456, digits=2) == "1.2e+02"
        assert format_number(0.001, digits=1) == "1e-03"
    
    def test_generate_kinematics_rng(self):
        """Test kinematics problem generation."""
        import random
        rng = random.Random(42)  # Fixed seed for reproducibility
        
        problem = generate_kinematics_rng(rng)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.category == "kinematics"
        assert "displacement" in problem.prompt.lower()
        assert "m" in problem.final_answer
        assert len(problem.important_facts) > 0
        assert len(problem.extraneous_facts) > 0
        assert len(problem.solution_steps) > 0
        assert "s = ut + 1/2 a t^2" in problem.metadata["equation"]
    
    def test_generate_newton_rng(self):
        """Test Newton's law problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_newton_rng(rng)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.category == "newton"
        assert "acceleration" in problem.prompt.lower()
        assert "m/s²" in problem.final_answer
        assert len(problem.important_facts) > 0
        assert len(problem.extraneous_facts) > 0
        assert "F = ma" in problem.metadata["equation"]
    
    def test_generate_energy_rng(self):
        """Test energy problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_energy_rng(rng)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.category == "energy"
        assert "speed" in problem.prompt.lower() or "velocity" in problem.prompt.lower()
        assert "m/s" in problem.final_answer
        assert len(problem.important_facts) > 0
        assert len(problem.extraneous_facts) > 0
        assert "mgh = 1/2 m v^2" in problem.metadata["equation"]
    
    def test_generate_momentum_rng(self):
        """Test momentum problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_momentum_rng(rng)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.category == "momentum"
        assert "collision" in problem.prompt.lower()
        assert "m/s" in problem.final_answer
        assert len(problem.important_facts) > 0
        assert len(problem.extraneous_facts) > 0
        assert "elastic" in problem.metadata["equations"]
    
    def test_generate_circuit_rng(self):
        """Test circuit problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_circuit_rng(rng)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.category == "circuits"
        assert "current" in problem.prompt.lower()
        assert "A" in problem.final_answer
        assert len(problem.important_facts) > 0
        assert len(problem.extraneous_facts) > 0
        assert "V = IR" in problem.metadata["equation"]
    
    def test_build_supervised_record(self):
        """Test supervised record building."""
        problem = PhysicsProblem(
            category="kinematics",
            prompt="Test prompt",
            important_facts=["fact1"],
            extraneous_facts=["ext1"],
            solution_steps=["step1"],
            final_answer="10.0 m/s",
            metadata={"equation": "v = d/t"}
        )
        
        record = build_supervised_record(problem)
        
        assert isinstance(record, dict)
        assert record["category"] == "kinematics"
        assert record["prompt"] == "Test prompt"
        assert record["important_facts"] == ["fact1"]
        assert record["extraneous_facts"] == ["ext1"]
        assert record["solution_steps"] == ["step1"]
        assert record["final_answer"] == "10.0 m/s"
        assert record["metadata"]["equation"] == "v = d/t"
    
    def test_build_preference_pair(self):
        """Test preference pair building."""
        import random
        rng = random.Random(42)
        
        problem = PhysicsProblem(
            category="kinematics",
            prompt="Test prompt",
            important_facts=["fact1"],
            extraneous_facts=["ext1"],
            solution_steps=["step1"],
            final_answer="10.0 m/s",
            metadata={"equation": "v = d/t"}
        )
        
        pair = build_preference_pair(problem, rng)
        
        assert isinstance(pair, dict)
        assert pair["category"] == "kinematics"
        assert pair["prompt"] == "Test prompt"
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"]["correct"] is True
        assert pair["chosen"]["uses_extraneous"] is False
        assert pair["rejected"]["correct"] is False
        assert pair["rejected"]["uses_extraneous"] is True
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        supervised, preference = generate_dataset(5, seed=42)
        
        assert len(supervised) == 5
        assert len(preference) == 5
        
        # Check supervised records
        for record in supervised:
            assert isinstance(record, dict)
            assert "category" in record
            assert "prompt" in record
            assert "important_facts" in record
            assert "extraneous_facts" in record
            assert "solution_steps" in record
            assert "final_answer" in record
            assert "metadata" in record
        
        # Check preference records
        for record in preference:
            assert isinstance(record, dict)
            assert "category" in record
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert "metadata" in record
    
    def test_save_jsonl(self, temp_dir):
        """Test JSONL file saving."""
        records = [
            {"category": "kinematics", "prompt": "Test 1", "answer": "10.0 m/s"},
            {"category": "newton", "prompt": "Test 2", "answer": "5.0 N"}
        ]
        
        file_path = temp_dir / "test.jsonl"
        save_jsonl(records, file_path)
        
        assert file_path.exists()
        
        # Read and verify content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        for i, line in enumerate(lines):
            record = json.loads(line.strip())
            assert record["category"] == records[i]["category"]
            assert record["prompt"] == records[i]["prompt"]
            assert record["answer"] == records[i]["answer"]


class TestInconsistentDatasetGeneration:
    """Test cases for inconsistent dataset generation."""
    
    def test_inconsistent_problem_dataclass(self):
        """Test InconsistentProblem dataclass."""
        problem = InconsistentProblem(
            category="kinematics",
            prompt="Test prompt",
            important_facts=["fact1"],
            inconsistent_facts=["incons1"],
            inconsistencies=[{"type": "contradictory_value", "field": "time"}],
            rationale="Test rationale",
            label="inconsistent",
            metadata={"equation": "v = d/t"}
        )
        
        assert problem.category == "kinematics"
        assert problem.prompt == "Test prompt"
        assert len(problem.important_facts) == 1
        assert len(problem.inconsistent_facts) == 1
        assert len(problem.inconsistencies) == 1
        assert problem.rationale == "Test rationale"
        assert problem.label == "inconsistent"
        assert problem.metadata["equation"] == "v = d/t"
    
    def test_generate_kinematics_inconsistent(self):
        """Test inconsistent kinematics problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_kinematics_inconsistent(rng)
        
        assert isinstance(problem, InconsistentProblem)
        assert problem.category == "kinematics"
        assert problem.label == "inconsistent"
        assert "displacement" in problem.prompt.lower()
        assert len(problem.inconsistencies) > 0
        assert "contradictory" in problem.rationale.lower()
        assert "s = ut + 1/2 a t^2" in problem.metadata["equation"]
    
    def test_generate_newton_inconsistent(self):
        """Test inconsistent Newton's law problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_newton_inconsistent(rng)
        
        assert isinstance(problem, InconsistentProblem)
        assert problem.category == "newton"
        assert problem.label == "inconsistent"
        assert "acceleration" in problem.prompt.lower()
        assert len(problem.inconsistencies) > 0
        assert "inconsistent" in problem.rationale.lower()
        assert "F = ma" in problem.metadata["equation"]
    
    def test_generate_energy_inconsistent(self):
        """Test inconsistent energy problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_energy_inconsistent(rng)
        
        assert isinstance(problem, InconsistentProblem)
        assert problem.category == "energy"
        assert problem.label == "inconsistent"
        assert "speed" in problem.prompt.lower() or "velocity" in problem.prompt.lower()
        assert len(problem.inconsistencies) > 0
        assert "mgh = 1/2 m v^2" in problem.metadata["equation"]
    
    def test_generate_momentum_inconsistent(self):
        """Test inconsistent momentum problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_momentum_inconsistent(rng)
        
        assert isinstance(problem, InconsistentProblem)
        assert problem.category == "momentum"
        assert problem.label == "inconsistent"
        assert "collision" in problem.prompt.lower()
        assert len(problem.inconsistencies) > 0
        assert "elastic" in problem.metadata["equations"]
    
    def test_generate_circuit_inconsistent(self):
        """Test inconsistent circuit problem generation."""
        import random
        rng = random.Random(42)
        
        problem = generate_circuit_inconsistent(rng)
        
        assert isinstance(problem, InconsistentProblem)
        assert problem.category == "circuits"
        assert problem.label == "inconsistent"
        assert "current" in problem.prompt.lower()
        assert len(problem.inconsistencies) > 0
        assert "V = IR" in problem.metadata["equation"]
    
    def test_build_unsolvable_record(self):
        """Test unsolvable record building."""
        problem = InconsistentProblem(
            category="kinematics",
            prompt="Test prompt",
            important_facts=["fact1"],
            inconsistent_facts=["incons1"],
            inconsistencies=[{"type": "contradictory_value", "field": "time"}],
            rationale="Test rationale",
            label="inconsistent",
            metadata={"equation": "v = d/t"}
        )
        
        record = build_unsolvable_record(problem)
        
        assert isinstance(record, dict)
        assert record["category"] == "kinematics"
        assert record["prompt"] == "Test prompt"
        assert record["important_facts"] == ["fact1"]
        assert record["inconsistent_facts"] == ["incons1"]
        assert record["inconsistencies"] == [{"type": "contradictory_value", "field": "time"}]
        assert record["rationale"] == "Test rationale"
        assert record["label"] == "inconsistent"
        assert record["metadata"]["equation"] == "v = d/t"
    
    def test_generate_inconsistent_dataset(self):
        """Test inconsistent dataset generation."""
        records = generate_inconsistent_dataset(5, seed=42)
        
        assert len(records) == 5
        
        for record in records:
            assert isinstance(record, dict)
            assert "category" in record
            assert "prompt" in record
            assert "important_facts" in record
            assert "inconsistent_facts" in record
            assert "inconsistencies" in record
            assert "rationale" in record
            assert "label" in record
            assert "metadata" in record
            assert record["label"] == "inconsistent"
