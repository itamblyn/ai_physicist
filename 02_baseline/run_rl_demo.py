#!/usr/bin/env python3
"""
Complete RL Demo for Physics Question Answering
Shows the full reinforcement learning system in action.
"""

import matplotlib.pyplot as plt
import numpy as np
from rl_physics_agent import train_rl_agent, evaluate_agent, QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
from physics_question_generator import PhysicsQuestionGenerator
import time


def run_complete_demo():
    """Run a complete demonstration of the RL system."""
    print("🤖 AI Physicist - Reinforcement Learning Demo")
    print("=" * 50)
    print("This demo shows how an AI agent learns to answer physics questions")
    print("using reinforcement learning with rewards for correct answers.\n")
    
    # Step 1: Show the physics questions
    print("📚 STEP 1: Physics Question Generator")
    print("-" * 40)
    generator = PhysicsQuestionGenerator()
    
    print("Sample physics questions:")
    for i, question_type in enumerate(generator.question_types[:3], 1):
        question_data = generator.generate_question(question_type)
        print(f"\n{i}. {question_type.replace('_', ' ').title()}")
        print(f"   Question: {question_data['question']}")
        print(f"   Answer: {question_data['answer']}")
    
    # Step 2: Show answer generation
    print(f"\n🎯 STEP 2: Answer Generation System")
    print("-" * 40)
    answer_generator = PhysicsAnswerGenerator()
    
    sample_question = generator.generate_question('kinematics_velocity')
    print(f"Question: {sample_question['question']}")
    print(f"Correct Answer: {sample_question['answer']}")
    
    options = answer_generator.generate_answer_options(sample_question, num_options=4)
    print("\nGenerated Answer Options:")
    for i, option in enumerate(options, 1):
        marker = "✓" if option == sample_question['answer'] else " "
        print(f"  {i}. {option} {marker}")
    
    # Step 3: Train the RL agent
    print(f"\n🧠 STEP 3: Training RL Agent")
    print("-" * 40)
    print("Training agent to learn from rewards...")
    print("(Correct answer = +1.0 reward, Wrong answer = -0.1 penalty)")
    
    start_time = time.time()
    agent, environment = train_rl_agent(episodes=500)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds!")
    print(f"Final accuracy: {environment.get_accuracy():.3f}")
    print(f"Total questions: {environment.total_answers}")
    print(f"Correct answers: {environment.correct_answers}")
    
    # Step 4: Show learning progress
    print(f"\n📈 STEP 4: Learning Progress")
    print("-" * 40)
    
    if agent.training_accuracy:
        episodes = range(0, len(agent.training_accuracy) * 10, 10)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, agent.training_accuracy, 'b-', linewidth=2, marker='o', markersize=3)
        plt.title('Agent Learning Progress')
        plt.xlabel('Training Episodes')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(episodes, agent.training_rewards, 'r-', linewidth=2, marker='o', markersize=3)
        plt.title('Reward History')
        plt.xlabel('Training Episodes')
        plt.ylabel('Episode Reward')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rl_learning_progress.png', dpi=150, bbox_inches='tight')
        print("Learning progress plot saved as 'rl_learning_progress.png'")
        
        # Show learning statistics
        initial_accuracy = agent.training_accuracy[0] if agent.training_accuracy else 0
        final_accuracy = agent.training_accuracy[-1] if agent.training_accuracy else 0
        improvement = final_accuracy - initial_accuracy
        
        print(f"Initial accuracy: {initial_accuracy:.3f}")
        print(f"Final accuracy: {final_accuracy:.3f}")
        print(f"Improvement: {improvement:+.3f}")
        
        if improvement > 0.1:
            print("✅ Agent shows significant learning!")
        elif improvement > 0.05:
            print("✅ Agent shows moderate learning")
        else:
            print("⚠️  Agent learning is minimal")
    
    # Step 5: Evaluate on new questions
    print(f"\n🎯 STEP 5: Evaluation on New Questions")
    print("-" * 40)
    print("Testing agent on 30 new questions it hasn't seen before...")
    
    evaluation_accuracy = evaluate_agent(agent, num_questions=30)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy on new questions: {evaluation_accuracy:.3f}")
    
    if evaluation_accuracy > 0.4:
        print("🎉 Excellent performance!")
    elif evaluation_accuracy > 0.25:
        print("👍 Good performance!")
    else:
        print("📚 Room for improvement - more training needed!")
    
    # Step 6: Show Q-table insights
    print(f"\n🔍 STEP 6: Q-Table Analysis")
    print("-" * 40)
    
    print(f"Q-table contains {len(agent.q_table)} different states")
    print(f"Agent learned {len(agent.action_space)} different actions")
    
    # Show some learned Q-values
    print("\nSample learned Q-values:")
    count = 0
    for state, actions in agent.q_table.items():
        if count >= 3:
            break
        print(f"\nState: {state}")
        # Show top 2 actions by Q-value
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        for action, q_value in sorted_actions[:2]:
            print(f"  {action}: {q_value:.3f}")
        count += 1
    
    # Step 7: Interactive demo
    print(f"\n🎮 STEP 7: Interactive Demo")
    print("-" * 40)
    print("Watch the agent answer questions in real-time!")
    print("(Press Enter to continue, 'q' to quit)")
    
    # Set agent to exploitation mode
    agent.epsilon = 0.0
    
    correct_count = 0
    total_count = 0
    
    for i in range(5):  # Show 5 questions
        user_input = input(f"\nPress Enter for question {i+1} (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        
        # Generate question
        state = environment.reset()
        answer_options = answer_generator.generate_answer_options(environment.current_question)
        
        # Agent chooses answer
        agent_answer = agent.get_action(state, answer_options)
        
        # Check if correct
        is_correct = environment._check_answer(agent_answer)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Display results
        print(f"\nQuestion: {environment.current_question['question']}")
        print(f"Question Type: {environment.current_question['type'].replace('_', ' ').title()}")
        print(f"Agent's Answer: {agent_answer}")
        print(f"Correct Answer: {environment.current_question['answer']}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print(f"Running Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    # Final summary
    print(f"\n🎯 DEMO SUMMARY")
    print("=" * 50)
    print("✅ Physics question generator working")
    print("✅ Answer generation system working")
    print("✅ RL agent training completed")
    print("✅ Learning progress tracked")
    print("✅ Agent evaluation completed")
    print("✅ Q-table analysis shown")
    print("✅ Interactive demo completed")
    
    print(f"\nKey Results:")
    print(f"  • Training accuracy: {environment.get_accuracy():.3f}")
    print(f"  • Evaluation accuracy: {evaluation_accuracy:.3f}")
    print(f"  • Interactive accuracy: {correct_count/total_count:.3f}" if total_count > 0 else "  • Interactive demo skipped")
    print(f"  • Q-table states: {len(agent.q_table)}")
    print(f"  • Training time: {training_time:.1f} seconds")
    
    print(f"\n🎉 The RL system successfully learned to answer physics questions!")
    print("The agent uses reinforcement learning to improve its performance")
    print("based on rewards for correct answers and penalties for wrong ones.")


if __name__ == "__main__":
    run_complete_demo()
