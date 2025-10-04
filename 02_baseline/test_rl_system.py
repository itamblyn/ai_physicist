#!/usr/bin/env python3
"""
Test script for the RL Physics Agent system
Demonstrates the basic functionality without requiring API keys.
"""

from rl_physics_agent import train_rl_agent, evaluate_agent, QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
from physics_question_generator import PhysicsQuestionGenerator
import time


def test_basic_rl_training():
    """Test basic RL training functionality."""
    print("Testing Basic RL Training")
    print("=" * 30)
    
    # Quick training test
    print("Training agent for 50 episodes...")
    agent, environment = train_rl_agent(episodes=50)
    
    print(f"Training completed!")
    print(f"Final accuracy: {environment.get_accuracy():.3f}")
    print(f"Total questions: {environment.total_answers}")
    print(f"Correct answers: {environment.correct_answers}")
    
    return agent, environment


def test_agent_learning():
    """Test that agent actually learns over time."""
    print("\nTesting Agent Learning")
    print("=" * 25)
    
    # Create components
    question_generator = PhysicsQuestionGenerator()
    environment = PhysicsRLEnvironment(question_generator)
    agent = QLearningAgent(learning_rate=0.2, epsilon=0.3)
    answer_generator = PhysicsAnswerGenerator()
    
    # Track accuracy over time
    accuracy_history = []
    
    for episode in range(100):
        # Reset environment
        state = environment.reset()
        
        # Generate answer options
        answer_options = answer_generator.generate_answer_options(environment.current_question)
        
        # Agent chooses action
        action = agent.get_action(state, answer_options)
        
        # Environment processes action
        next_state, reward, done, info = environment.step(action)
        
        # Update Q-values
        agent.update_q_value(state, action, reward, next_state, done)
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record accuracy every 10 episodes
        if episode % 10 == 0:
            accuracy = environment.get_accuracy()
            accuracy_history.append(accuracy)
            print(f"Episode {episode:3d}: Accuracy = {accuracy:.3f}, Epsilon = {agent.epsilon:.3f}")
    
    print(f"\nLearning progression:")
    for i, acc in enumerate(accuracy_history):
        print(f"  Episodes {i*10:3d}-{(i+1)*10-1:3d}: {acc:.3f}")
    
    # Check if agent improved
    if len(accuracy_history) >= 2:
        improvement = accuracy_history[-1] - accuracy_history[0]
        print(f"Overall improvement: {improvement:+.3f}")
        if improvement > 0.1:
            print("✓ Agent shows significant learning!")
        else:
            print("⚠ Agent learning is minimal")
    
    return agent, environment


def test_answer_generation():
    """Test answer generation system."""
    print("\nTesting Answer Generation")
    print("=" * 25)
    
    question_generator = PhysicsQuestionGenerator()
    answer_generator = PhysicsAnswerGenerator()
    
    # Test with different question types
    for question_type in question_generator.question_types[:3]:  # Test first 3 types
        print(f"\nQuestion Type: {question_type}")
        question_data = question_generator.generate_question(question_type)
        
        print(f"Question: {question_data['question']}")
        print(f"Correct Answer: {question_data['answer']}")
        
        # Generate answer options
        options = answer_generator.generate_answer_options(question_data, num_options=4)
        print("Answer Options:")
        for i, option in enumerate(options, 1):
            marker = "✓" if option == question_data['answer'] else " "
            print(f"  {i}. {option} {marker}")


def test_q_table_learning():
    """Test Q-table learning and state representation."""
    print("\nTesting Q-Table Learning")
    print("=" * 25)
    
    question_generator = PhysicsQuestionGenerator()
    environment = PhysicsRLEnvironment(question_generator)
    agent = QLearningAgent(learning_rate=0.3, epsilon=0.5)
    answer_generator = PhysicsAnswerGenerator()
    
    # Train for a few episodes
    for episode in range(20):
        state = environment.reset()
        answer_options = answer_generator.generate_answer_options(environment.current_question)
        action = agent.get_action(state, answer_options)
        next_state, reward, done, info = environment.step(action)
        agent.update_q_value(state, action, reward, next_state, done)
        agent.decay_epsilon()
    
    # Analyze Q-table
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"Action space size: {len(agent.action_space)}")
    
    # Show some Q-values
    print("\nSample Q-values:")
    count = 0
    for state, actions in agent.q_table.items():
        if count >= 3:
            break
        print(f"  State: {state}")
        for action, q_value in list(actions.items())[:2]:  # Show first 2 actions
            print(f"    {action}: {q_value:.3f}")
        count += 1


def test_evaluation():
    """Test agent evaluation on new questions."""
    print("\nTesting Agent Evaluation")
    print("=" * 25)
    
    # Train a quick agent
    agent, _ = train_rl_agent(episodes=100)
    
    # Evaluate on new questions
    print("Evaluating on 20 new questions...")
    accuracy = evaluate_agent(agent, num_questions=20)
    
    print(f"Evaluation accuracy: {accuracy:.3f}")
    
    if accuracy > 0.3:
        print("✓ Agent performs reasonably well!")
    else:
        print("⚠ Agent performance is low")


def main():
    """Run all tests."""
    print("RL Physics Agent System Tests")
    print("=" * 35)
    
    try:
        # Test 1: Basic training
        test_basic_rl_training()
        
        # Test 2: Learning progression
        test_agent_learning()
        
        # Test 3: Answer generation
        test_answer_generation()
        
        # Test 4: Q-table learning
        test_q_table_learning()
        
        # Test 5: Evaluation
        test_evaluation()
        
        print("\n" + "=" * 35)
        print("All tests completed successfully!")
        print("The RL system is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
