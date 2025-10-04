#!/usr/bin/env python3
"""
Demo script for RL Physics Agent Training
Shows the reinforcement learning process in action with visual feedback.
"""

import matplotlib.pyplot as plt
import numpy as np
from rl_physics_agent import train_rl_agent, evaluate_agent, QLearningAgent, PhysicsRLEnvironment, PhysicsAnswerGenerator
from physics_question_generator import PhysicsQuestionGenerator


def plot_training_progress(agent: QLearningAgent):
    """Plot training progress over time."""
    if not agent.training_accuracy:
        print("No training data to plot.")
        return
    
    episodes = range(0, len(agent.training_accuracy) * 10, 10)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(episodes, agent.training_accuracy, 'b-', linewidth=2)
    plt.title('Training Accuracy Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(episodes, agent.training_rewards, 'r-', linewidth=2)
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


def interactive_demo():
    """Interactive demo of the RL agent answering questions."""
    print("Interactive RL Physics Agent Demo")
    print("=" * 40)
    
    # Load or create agent
    try:
        agent = QLearningAgent()
        agent.load_model("physics_rl_model_final.pkl")
        print("Loaded trained model!")
    except FileNotFoundError:
        print("No trained model found. Training a new agent...")
        agent, _ = train_rl_agent(episodes=500)
    
    # Setup components
    question_generator = PhysicsQuestionGenerator()
    environment = PhysicsRLEnvironment(question_generator)
    answer_generator = PhysicsAnswerGenerator()
    
    # Set agent to exploitation mode
    agent.epsilon = 0.0
    
    print("\nThe agent will now answer physics questions. Watch it learn!")
    print("Press Enter to continue, 'q' to quit")
    
    correct_count = 0
    total_count = 0
    
    while True:
        user_input = input("\nPress Enter for next question (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        
        # Generate question and answers
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
        print(f"\nAnswer Options:")
        for i, option in enumerate(answer_options, 1):
            marker = "✓" if option == environment.current_question['answer'] else " "
            print(f"  {i}. {option} {marker}")
        
        print(f"\nAgent's Answer: {agent_answer}")
        print(f"Correct Answer: {environment.current_question['answer']}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print(f"Current Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
        
        # Show Q-values for this state
        print(f"\nQ-values for this state:")
        for action in answer_options:
            q_value = agent.q_table[state][action]
            print(f"  {action}: {q_value:.3f}")


def compare_learning_rates():
    """Compare different learning rates."""
    print("Comparing Different Learning Rates")
    print("=" * 40)
    
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Create new agent with specific learning rate
        agent = QLearningAgent(learning_rate=lr, epsilon=0.2)
        question_generator = PhysicsQuestionGenerator()
        environment = PhysicsRLEnvironment(question_generator)
        answer_generator = PhysicsAnswerGenerator()
        
        # Train for fewer episodes for comparison
        for episode in range(200):
            state = environment.reset()
            answer_options = answer_generator.generate_answer_options(environment.current_question)
            action = agent.get_action(state, answer_options)
            next_state, reward, done, info = environment.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            agent.decay_epsilon()
        
        # Evaluate
        final_accuracy = environment.get_accuracy()
        results[lr] = final_accuracy
        print(f"Final accuracy: {final_accuracy:.3f}")
    
    print(f"\nLearning Rate Comparison Results:")
    for lr, accuracy in results.items():
        print(f"  LR {lr}: {accuracy:.3f}")


def analyze_q_table(agent: QLearningAgent):
    """Analyze the learned Q-table."""
    print("\nQ-Table Analysis")
    print("=" * 30)
    
    if not agent.q_table:
        print("No Q-table data available.")
        return
    
    # Count states and actions
    num_states = len(agent.q_table)
    all_actions = set()
    for state_actions in agent.q_table.values():
        all_actions.update(state_actions.keys())
    num_actions = len(all_actions)
    
    print(f"Number of states: {num_states}")
    print(f"Number of unique actions: {num_actions}")
    print(f"Total Q-values: {sum(len(actions) for actions in agent.q_table.values())}")
    
    # Show some example Q-values
    print(f"\nSample Q-values:")
    count = 0
    for state, actions in agent.q_table.items():
        if count >= 5:
            break
        print(f"  State: {state}")
        for action, q_value in actions.items():
            if count >= 5:
                break
            print(f"    {action}: {q_value:.3f}")
        count += 1


def main():
    """Main demo function."""
    print("RL Physics Agent Demo")
    print("=" * 30)
    print("1. Train new agent")
    print("2. Interactive demo")
    print("3. Compare learning rates")
    print("4. Analyze Q-table")
    print("5. Full training with plots")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        print("\nTraining new agent...")
        agent, env = train_rl_agent(episodes=1000)
        print("Training completed!")
        
    elif choice == '2':
        interactive_demo()
        
    elif choice == '3':
        compare_learning_rates()
        
    elif choice == '4':
        try:
            agent = QLearningAgent()
            agent.load_model("physics_rl_model_final.pkl")
            analyze_q_table(agent)
        except FileNotFoundError:
            print("No trained model found. Please train an agent first.")
            
    elif choice == '5':
        print("\nFull training with progress plots...")
        agent, env = train_rl_agent(episodes=2000)
        plot_training_progress(agent)
        evaluate_agent(agent, num_questions=100)
        
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
