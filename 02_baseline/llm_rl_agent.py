#!/usr/bin/env python3
"""
LLM-based Reinforcement Learning Agent for Physics Questions
Uses actual LLM APIs to generate answers and learns from rewards.
"""

import openai
import json
import time
import random
from typing import Dict, List, Tuple, Any, Optional
from rl_physics_agent import PhysicsRLEnvironment, QLearningAgent
from physics_question_generator import PhysicsQuestionGenerator


class LLMRLAgent:
    """RL agent that uses LLM APIs to generate answers to physics questions."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 learning_rate: float = 0.1, epsilon: float = 0.1):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        # Q-table for state-action values
        self.q_table = {}
        
        # Context memory for better answers
        self.conversation_history = []
        
        # Performance tracking
        self.correct_answers = 0
        self.total_answers = 0
        self.reward_history = []
        
    def generate_answer(self, question: str, question_type: str, 
                       previous_attempts: List[Dict] = None) -> str:
        """Generate answer using LLM API."""
        
        # Build context from previous attempts
        context = self._build_context(question_type, previous_attempts)
        
        # Create prompt for the LLM
        prompt = self._create_prompt(question, question_type, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1  # Low temperature for consistent answers
            )
            
            answer = response.choices[0].message.content.strip()
            return self._extract_numerical_answer(answer, question_type)
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return self._generate_fallback_answer(question_type)
    
    def _build_context(self, question_type: str, previous_attempts: List[Dict] = None) -> str:
        """Build context from previous attempts and question type."""
        context = f"Question type: {question_type.replace('_', ' ').title()}\n"
        
        if previous_attempts:
            context += "Previous attempts:\n"
            for i, attempt in enumerate(previous_attempts[-3:], 1):  # Last 3 attempts
                context += f"  {i}. Question: {attempt['question'][:100]}...\n"
                context += f"     Your answer: {attempt['user_answer']}\n"
                context += f"     Correct: {attempt['correct']}\n"
                context += f"     Correct answer: {attempt['correct_answer']}\n\n"
        
        return context
    
    def _create_prompt(self, question: str, question_type: str, context: str) -> str:
        """Create prompt for the LLM."""
        prompt = f"""You are a physics expert taking an exam. Answer the following question with just the numerical value and unit.

{context}

Question: {question}

Instructions:
1. Read the question carefully
2. Identify the physics concept (it's a {question_type.replace('_', ' ')} problem)
3. Apply the correct formula
4. Calculate the answer
5. Provide ONLY the final numerical answer with units (e.g., "15.5 m/s" or "42.3 N")

Answer:"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM."""
        return """You are an expert physics tutor helping a student with mechanics problems. 
You must provide accurate, step-by-step solutions and give only the final numerical answer when requested.
Be precise with units and calculations. If you make an error, learn from it for future questions."""
    
    def _extract_numerical_answer(self, llm_response: str, question_type: str) -> str:
        """Extract numerical answer from LLM response."""
        import re
        
        # Look for numerical values with units
        patterns = [
            r'(\d+\.?\d*)\s*(m/s|m/s²|N|J|m)',
            r'(\d+\.?\d*)\s*(meters per second|meters per second squared|newtons|joules|meters)',
            r'(\d+\.?\d*)\s*(ms⁻¹|ms⁻²)',
            r'(\d+\.?\d*)'  # Just number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                number = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else self._get_default_unit(question_type)
                return f"{number} {unit}"
        
        # Fallback: return first number found
        numbers = re.findall(r'-?\d+\.?\d*', llm_response)
        if numbers:
            return f"{numbers[0]} {self._get_default_unit(question_type)}"
        
        return self._generate_fallback_answer(question_type)
    
    def _get_default_unit(self, question_type: str) -> str:
        """Get default unit for question type."""
        unit_map = {
            'kinematics_velocity': 'm/s',
            'kinematics_acceleration': 'm/s²',
            'kinematics_displacement': 'm',
            'force_newton_second': 'N',
            'kinetic_energy': 'J',
            'potential_energy': 'J',
            'work_energy': 'J',
            'simple_collision': 'm/s'
        }
        return unit_map.get(question_type, 'units')
    
    def _generate_fallback_answer(self, question_type: str) -> str:
        """Generate fallback answer when LLM fails."""
        # Simple fallback based on question type
        fallback_values = {
            'kinematics_velocity': '10.0 m/s',
            'kinematics_acceleration': '5.0 m/s²',
            'kinematics_displacement': '50.0 m',
            'force_newton_second': '25.0 N',
            'kinetic_energy': '100.0 J',
            'potential_energy': '200.0 J',
            'work_energy': '150.0 J',
            'simple_collision': '8.0 m/s'
        }
        return fallback_values.get(question_type, '1.0 units')
    
    def update_from_feedback(self, question: str, question_type: str, 
                           user_answer: str, correct_answer: str, 
                           is_correct: bool, reward: float):
        """Update agent based on feedback."""
        self.total_answers += 1
        if is_correct:
            self.correct_answers += 1
        
        self.reward_history.append(reward)
        
        # Store attempt for context
        attempt = {
            'question': question,
            'question_type': question_type,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'correct': is_correct,
            'reward': reward
        }
        self.conversation_history.append(attempt)
        
        # Keep only recent history
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total_answers == 0:
            return 0.0
        return self.correct_answers / self.total_answers
    
    def get_average_reward(self, window: int = 10) -> float:
        """Get average reward over recent attempts."""
        if not self.reward_history:
            return 0.0
        recent_rewards = self.reward_history[-window:]
        return sum(recent_rewards) / len(recent_rewards)


class LLMRLEnvironment:
    """Environment for training LLM-based RL agent."""
    
    def __init__(self, question_generator: PhysicsQuestionGenerator):
        self.question_generator = question_generator
        self.current_question = None
        self.agent_attempts = []
    
    def reset(self) -> Dict:
        """Reset environment and return new question."""
        self.current_question = self.question_generator.generate_question()
        self.agent_attempts = []
        return self.current_question
    
    def step(self, agent_answer: str, agent: LLMRLAgent) -> Tuple[float, bool, Dict]:
        """Process agent's answer and return reward, done, info."""
        is_correct = self._check_answer(agent_answer)
        
        if is_correct:
            reward = 1.0
        else:
            reward = -0.1
        
        # Update agent with feedback
        agent.update_from_feedback(
            question=self.current_question['question'],
            question_type=self.current_question['type'],
            user_answer=agent_answer,
            correct_answer=self.current_question['answer'],
            is_correct=is_correct,
            reward=reward
        )
        
        # Store attempt
        attempt = {
            'question': self.current_question['question'],
            'user_answer': agent_answer,
            'correct_answer': self.current_question['answer'],
            'correct': is_correct,
            'reward': reward
        }
        self.agent_attempts.append(attempt)
        
        # Episode is done after each question
        done = True
        
        info = {
            'correct': is_correct,
            'question_type': self.current_question['type'],
            'accuracy': agent.get_accuracy(),
            'average_reward': agent.get_average_reward()
        }
        
        return reward, done, info
    
    def _check_answer(self, user_answer: str) -> bool:
        """Check if user answer matches correct answer."""
        correct_answer = self.current_question['answer']
        
        # Normalize answers for comparison
        def normalize_answer(answer):
            import re
            answer = answer.lower().strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return float(numbers[0])
            return answer
        
        try:
            user_num = normalize_answer(user_answer)
            correct_num = normalize_answer(correct_answer)
            
            if isinstance(user_num, float) and isinstance(correct_num, float):
                return abs(user_num - correct_num) < 0.01
            else:
                return user_num == correct_num
        except:
            return user_answer.strip().lower() == correct_answer.strip().lower()


def train_llm_agent(api_key: str, episodes: int = 50, model: str = "gpt-3.5-turbo"):
    """Train LLM-based RL agent."""
    print("Training LLM-based RL Agent")
    print("=" * 40)
    
    # Initialize components
    question_generator = PhysicsQuestionGenerator()
    environment = LLMRLEnvironment(question_generator)
    agent = LLMRLAgent(api_key=api_key, model=model)
    
    print(f"Using model: {model}")
    print(f"Training for {episodes} episodes...")
    
    for episode in range(episodes):
        # Reset environment
        question_data = environment.reset()
        
        # Generate answer using LLM
        agent_answer = agent.generate_answer(
            question=question_data['question'],
            question_type=question_data['type'],
            previous_attempts=agent.conversation_history
        )
        
        # Process answer and get reward
        reward, done, info = environment.step(agent_answer, agent)
        
        # Print progress
        if episode % 5 == 0:
            print(f"Episode {episode:3d} | Accuracy: {info['accuracy']:.3f} | "
                  f"Avg Reward: {info['average_reward']:.3f} | "
                  f"Answer: {agent_answer}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {agent.get_accuracy():.3f}")
    print(f"Total questions: {agent.total_answers}")
    print(f"Correct answers: {agent.correct_answers}")
    
    return agent, environment


def evaluate_llm_agent(agent: LLMRLAgent, num_questions: int = 20):
    """Evaluate the trained LLM agent."""
    print(f"\nEvaluating LLM agent on {num_questions} new questions...")
    print("=" * 50)
    
    question_generator = PhysicsQuestionGenerator()
    environment = LLMRLEnvironment(question_generator)
    
    # Reset agent stats
    agent.correct_answers = 0
    agent.total_answers = 0
    agent.reward_history = []
    
    for i in range(num_questions):
        question_data = environment.reset()
        
        # Generate answer
        agent_answer = agent.generate_answer(
            question=question_data['question'],
            question_type=question_data['type'],
            previous_attempts=agent.conversation_history
        )
        
        # Process answer
        reward, done, info = environment.step(agent_answer, agent)
        
        # Show first few questions
        if i < 3:
            print(f"\nQuestion {i+1}: {question_data['question']}")
            print(f"Correct Answer: {question_data['answer']}")
            print(f"Agent Answer: {agent_answer}")
            print(f"Correct: {info['correct']}")
        
        time.sleep(0.5)  # Rate limiting
    
    final_accuracy = agent.get_accuracy()
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {final_accuracy:.3f}")
    print(f"Correct: {agent.correct_answers}/{agent.total_answers}")
    
    return final_accuracy


def main():
    """Main function for LLM RL training."""
    print("LLM-based RL Physics Agent")
    print("=" * 30)
    
    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("API key required!")
        return
    
    # Choose model
    print("\nAvailable models:")
    print("1. gpt-3.5-turbo (faster, cheaper)")
    print("2. gpt-4 (more accurate, expensive)")
    
    model_choice = input("Choose model (1-2): ").strip()
    model = "gpt-3.5-turbo" if model_choice == "1" else "gpt-4"
    
    # Get number of episodes
    try:
        episodes = int(input("Number of training episodes (default 50): ") or "50")
    except ValueError:
        episodes = 50
    
    # Train agent
    agent, env = train_llm_agent(api_key, episodes, model)
    
    # Evaluate agent
    evaluate_llm_agent(agent, num_questions=20)


if __name__ == "__main__":
    main()
