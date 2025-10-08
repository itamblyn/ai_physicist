"""
Unit tests for LLM RL Agent components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_rl_agent import LLMRLAgent, LLMRLEnvironment


class TestLLMRLAgent:
    """Test cases for LLMRLAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            assert agent.model == "gpt-3.5-turbo"
            assert agent.learning_rate == 0.1
            assert agent.epsilon == 0.1
            assert isinstance(agent.q_table, dict)
            assert len(agent.conversation_history) == 0
            assert agent.correct_answers == 0
            assert agent.total_answers == 0
            assert len(agent.reward_history) == 0
    
    def test_init_custom_params(self):
        """Test agent initialization with custom parameters."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(
                api_key="test_key",
                model="gpt-4",
                learning_rate=0.2,
                epsilon=0.2
            )
            
            assert agent.model == "gpt-4"
            assert agent.learning_rate == 0.2
            assert agent.epsilon == 0.2
    
    @patch('llm_rl_agent.openai.OpenAI')
    def test_generate_answer_success(self, mock_openai):
        """Test successful answer generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "10.5 m/s"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        agent = LLMRLAgent(api_key="test_key")
        
        question = "A car travels 100 meters in 10 seconds. What is the velocity?"
        answer = agent.generate_answer(question, "kinematics_velocity")
        
        assert answer == "10.5 m/s"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('llm_rl_agent.openai.OpenAI')
    def test_generate_answer_api_error(self, mock_openai):
        """Test answer generation with API error."""
        # Mock OpenAI to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        agent = LLMRLAgent(api_key="test_key")
        
        question = "A car travels 100 meters in 10 seconds. What is the velocity?"
        answer = agent.generate_answer(question, "kinematics_velocity")
        
        # Should return fallback answer
        assert "m/s" in answer
        assert answer != "10.5 m/s"  # Should be fallback, not the mocked response
    
    def test_build_context(self):
        """Test context building."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Test with no previous attempts
            context = agent._build_context("kinematics_velocity", None)
            assert "kinematics_velocity" in context
            assert "Question type:" in context
            
            # Test with previous attempts
            previous_attempts = [
                {
                    'question': 'Test question 1',
                    'user_answer': '5.0 m/s',
                    'correct': True,
                    'correct_answer': '5.0 m/s'
                },
                {
                    'question': 'Test question 2',
                    'user_answer': '10.0 m/s',
                    'correct': False,
                    'correct_answer': '15.0 m/s'
                }
            ]
            
            context = agent._build_context("kinematics_velocity", previous_attempts)
            assert "Previous attempts:" in context
            assert "Test question 1" in context
            assert "Test question 2" in context
            assert "5.0 m/s" in context
            assert "10.0 m/s" in context
    
    def test_create_prompt(self):
        """Test prompt creation."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            question = "A car travels 100 meters in 10 seconds. What is the velocity?"
            question_type = "kinematics_velocity"
            context = "Question type: Kinematics Velocity"
            
            prompt = agent._create_prompt(question, question_type, context)
            
            assert question in prompt
            assert "kinematics velocity" in prompt.lower()
            assert "Answer:" in prompt
            assert "numerical value and unit" in prompt
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            system_prompt = agent._get_system_prompt()
            
            assert "physics tutor" in system_prompt.lower()
            assert "accurate" in system_prompt.lower()
            assert "step-by-step" in system_prompt.lower()
    
    def test_extract_numerical_answer_with_units(self):
        """Test numerical answer extraction with units."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Test various formats
            test_cases = [
                ("The answer is 10.5 m/s", "10.5 m/s"),
                ("Velocity = 15.2 meters per second", "15.2 m/s"),
                ("Result: 8.3 ms⁻¹", "8.3 m/s"),
                ("Just the number: 12.7", "12.7 m/s"),  # Should use default unit
            ]
            
            for llm_response, expected in test_cases:
                result = agent._extract_numerical_answer(llm_response, "kinematics_velocity")
                assert "m/s" in result
                assert any(char.isdigit() for char in result)
    
    def test_extract_numerical_answer_fallback(self):
        """Test numerical answer extraction fallback."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Test with no numbers
            result = agent._extract_numerical_answer("No numbers here", "kinematics_velocity")
            assert "m/s" in result  # Should be fallback answer
    
    def test_get_default_unit(self):
        """Test default unit mapping."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            test_cases = [
                ("kinematics_velocity", "m/s"),
                ("kinematics_acceleration", "m/s²"),
                ("kinematics_displacement", "m"),
                ("force_newton_second", "N"),
                ("kinetic_energy", "J"),
                ("potential_energy", "J"),
                ("work_energy", "J"),
                ("simple_collision", "m/s"),
                ("unknown_type", "units")
            ]
            
            for question_type, expected_unit in test_cases:
                result = agent._get_default_unit(question_type)
                assert result == expected_unit
    
    def test_generate_fallback_answer(self):
        """Test fallback answer generation."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            test_cases = [
                ("kinematics_velocity", "m/s"),
                ("kinematics_acceleration", "m/s²"),
                ("force_newton_second", "N"),
                ("kinetic_energy", "J"),
                ("unknown_type", "units")
            ]
            
            for question_type, expected_unit in test_cases:
                result = agent._generate_fallback_answer(question_type)
                assert expected_unit in result
                assert any(char.isdigit() for char in result)
    
    def test_update_from_feedback(self):
        """Test feedback update."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Initial state
            assert agent.total_answers == 0
            assert agent.correct_answers == 0
            assert len(agent.reward_history) == 0
            assert len(agent.conversation_history) == 0
            
            # Update with correct answer
            agent.update_from_feedback(
                question="Test question",
                question_type="kinematics_velocity",
                user_answer="10.0 m/s",
                correct_answer="10.0 m/s",
                is_correct=True,
                reward=1.0
            )
            
            assert agent.total_answers == 1
            assert agent.correct_answers == 1
            assert len(agent.reward_history) == 1
            assert agent.reward_history[0] == 1.0
            assert len(agent.conversation_history) == 1
            
            # Update with incorrect answer
            agent.update_from_feedback(
                question="Test question 2",
                question_type="kinematics_velocity",
                user_answer="5.0 m/s",
                correct_answer="10.0 m/s",
                is_correct=False,
                reward=-0.1
            )
            
            assert agent.total_answers == 2
            assert agent.correct_answers == 1  # Still 1
            assert len(agent.reward_history) == 2
            assert agent.reward_history[1] == -0.1
            assert len(agent.conversation_history) == 2
    
    def test_get_accuracy(self):
        """Test accuracy calculation."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Initially no answers
            assert agent.get_accuracy() == 0.0
            
            # Add some answers
            agent.correct_answers = 3
            agent.total_answers = 5
            assert agent.get_accuracy() == 0.6
            
            # All correct
            agent.correct_answers = 5
            agent.total_answers = 5
            assert agent.get_accuracy() == 1.0
    
    def test_get_average_reward(self):
        """Test average reward calculation."""
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # No rewards
            assert agent.get_average_reward() == 0.0
            
            # Add some rewards
            agent.reward_history = [1.0, 0.5, -0.1, 0.8, 0.2]
            assert agent.get_average_reward() == 0.48  # (1.0 + 0.5 + -0.1 + 0.8 + 0.2) / 5
            
            # Test with window
            assert agent.get_average_reward(window=3) == 0.3  # (0.8 + 0.2) / 3, but only last 3


class TestLLMRLEnvironment:
    """Test cases for LLMRLEnvironment class."""
    
    def test_init(self):
        """Test environment initialization."""
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
        assert env.question_generator is generator
        assert env.current_question is None
        assert len(env.agent_attempts) == 0
    
    def test_reset(self):
        """Test environment reset."""
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
        question_data = env.reset()
        
        assert env.current_question is not None
        assert len(env.agent_attempts) == 0
        assert question_data is env.current_question
        
        # Check question structure
        assert 'question' in env.current_question
        assert 'answer' in env.current_question
        assert 'type' in env.current_question
    
    def test_step_correct_answer(self):
        """Test step with correct answer."""
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
        # Reset to get a question
        question_data = env.reset()
        correct_answer = env.current_question['answer']
        
        # Create mock agent
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Step with correct answer
            reward, done, info = env.step(correct_answer, agent)
            
            assert reward == 1.0
            assert done is True
            assert info['correct'] is True
            assert info['question_type'] == env.current_question['type']
            assert len(env.agent_attempts) == 1
            assert env.agent_attempts[0]['correct'] is True
    
    def test_step_incorrect_answer(self):
        """Test step with incorrect answer."""
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
        # Reset to get a question
        question_data = env.reset()
        wrong_answer = "999.99 m/s"  # Obviously wrong
        
        # Create mock agent
        with patch('llm_rl_agent.openai.OpenAI'):
            agent = LLMRLAgent(api_key="test_key")
            
            # Step with wrong answer
            reward, done, info = env.step(wrong_answer, agent)
            
            assert reward == -0.1
            assert done is True
            assert info['correct'] is False
            assert len(env.agent_attempts) == 1
            assert env.agent_attempts[0]['correct'] is False
    
    def test_check_answer_numerical(self):
        """Test answer checking with numerical values."""
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
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
        from physics_question_generator import PhysicsQuestionGenerator
        
        generator = PhysicsQuestionGenerator()
        env = LLMRLEnvironment(generator)
        
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
