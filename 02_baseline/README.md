# AI Physicist - Baseline RL System

This directory contains the baseline reinforcement learning system for training AI agents to answer physics questions. The system uses Q-learning and LLM integration to create intelligent physics tutoring agents.

## Overview

The baseline system demonstrates how reinforcement learning can be applied to educational AI, specifically for physics problem-solving. It includes both traditional Q-learning approaches and modern LLM-enhanced methods.

## Files

### Core Components

- **`rl_physics_agent.py`** - Main RL system with Q-learning agent and physics environment
- **`llm_rl_agent.py`** - LLM-enhanced RL agent using OpenAI API
- **`demo_rl_training.py`** - Interactive demo and training visualization
- **`run_rl_demo.py`** - Simple demo runner
- **`test_rl_system.py`** - Test suite for the RL components

### Utilities

- **`physics_question_generator.py`** - Symlink to question generator (from `../01_generate_questions/`)
- **`clean.sh`** - Cleanup script for temporary files

## Features

### 🤖 Q-Learning Agent
- **State Encoding**: Converts physics questions into learnable state representations
- **Action Space**: Dynamic answer generation with multiple choice options
- **Reward System**: +1.0 for correct answers, -0.1 for incorrect answers
- **Epsilon-Greedy Policy**: Balances exploration vs exploitation
- **Model Persistence**: Save/load trained models

### 🧠 LLM Integration
- **OpenAI API Support**: Uses GPT models for intelligent answer generation
- **Context Learning**: Maintains conversation history for better performance
- **Fallback System**: Graceful degradation when API is unavailable
- **Rate Limiting**: Prevents API overuse

### 📊 Training & Evaluation
- **Progress Tracking**: Real-time accuracy and reward monitoring
- **Visualization**: Training curves and performance plots
- **Interactive Demo**: Live question-answering sessions
- **Comparative Analysis**: Different learning rate experiments

## Quick Start

### Basic Q-Learning Training

```bash
# Train a basic Q-learning agent
python rl_physics_agent.py

# This will:
# - Train for 100,000 episodes
# - Save model as 'physics_rl_model_final.pkl'
# - Show final accuracy and statistics
```

### Interactive Demo

```bash
# Run the interactive demo
python demo_rl_training.py

# Choose from options:
# 1. Train new agent
# 2. Interactive demo (answer questions live)
# 3. Compare learning rates
# 4. Analyze Q-table
# 5. Full training with plots
```

### LLM-Enhanced Training

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Train with LLM integration
python llm_rl_agent.py
```

## System Architecture

```
Physics Question Generator
    ↓
RL Environment (PhysicsRLEnvironment)
    ↓
Agent (QLearningAgent or LLMRLAgent)
    ↓
Answer Generator (PhysicsAnswerGenerator)
    ↓
Reward System & Learning
```

### Key Classes

#### `PhysicsRLEnvironment`
- Manages physics questions and state transitions
- Encodes questions into state representations
- Evaluates answers and provides rewards
- Tracks accuracy and performance metrics

#### `QLearningAgent`
- Implements Q-learning algorithm
- Maintains Q-table for state-action values
- Uses epsilon-greedy exploration policy
- Supports model saving/loading

#### `PhysicsAnswerGenerator`
- Creates multiple choice answer options
- Generates plausible wrong answers
- Adapts to different question types
- Provides realistic distractors

## Training Process

### 1. Environment Setup
```python
question_generator = PhysicsQuestionGenerator()
environment = PhysicsRLEnvironment(question_generator)
agent = QLearningAgent()
answer_generator = PhysicsAnswerGenerator()
```

### 2. Training Loop
```python
for episode in range(episodes):
    state = environment.reset()
    answer_options = answer_generator.generate_answer_options(environment.current_question)
    action = agent.get_action(state, answer_options)
    next_state, reward, done, info = environment.step(action)
    agent.update_q_value(state, action, reward, next_state, done)
    agent.decay_epsilon()
```

### 3. Evaluation
```python
# Set to exploitation mode
agent.epsilon = 0.0

# Test on new questions
accuracy = evaluate_agent(agent, num_questions=100)
```

## Configuration

### Q-Learning Parameters

```python
agent = QLearningAgent(
    learning_rate=0.1,      # How fast to learn
    discount_factor=0.9,    # Future reward importance
    epsilon=0.1,            # Exploration rate
    epsilon_decay=0.995,    # Exploration decay
    epsilon_min=0.01        # Minimum exploration
)
```

### Training Parameters

```python
train_rl_agent(
    episodes=1000,          # Number of training episodes
    save_interval=100       # How often to save model
)
```

## Performance Metrics

The system tracks several key metrics:

- **Accuracy**: Percentage of questions answered correctly
- **Episode Rewards**: Cumulative reward per episode
- **Learning Progress**: Accuracy improvement over time
- **Q-Value Analysis**: State-action value insights
- **Exploration Rate**: Epsilon decay over time

## Question Types Supported

The system can learn to answer 8 types of physics questions:

1. **Kinematics - Velocity**: Distance and time calculations
2. **Kinematics - Acceleration**: Velocity change problems
3. **Kinematics - Displacement**: Motion with constant acceleration
4. **Force (Newton's 2nd Law)**: F = ma calculations
5. **Kinetic Energy**: KE = ½mv² problems
6. **Potential Energy**: PE = mgh calculations
7. **Work-Energy**: W = Fd problems
8. **Simple Collisions**: 1D elastic collision problems

## State Representation

Questions are encoded into states using:
- **Question Type**: The physics concept being tested
- **Key Terms**: Important physics vocabulary in the question
- **Numerical Context**: Relevant values and units

Example state: `"kinematics_velocity_distance_time"`

## Reward System

- **Correct Answer**: +1.0 reward
- **Incorrect Answer**: -0.1 penalty
- **Episode Completion**: Immediate feedback

This reward structure encourages learning while providing gentle penalties for mistakes.

## Advanced Features

### Model Persistence
```python
# Save trained model
agent.save_model("my_physics_agent.pkl")

# Load trained model
agent.load_model("my_physics_agent.pkl")
```

### Training Visualization
```python
# Plot training progress
plot_training_progress(agent)

# Shows accuracy and reward curves over time
```

### Interactive Analysis
```python
# Analyze learned Q-table
analyze_q_table(agent)

# Shows state-action value distributions
```

## Example Usage

### Train and Evaluate
```python
from rl_physics_agent import train_rl_agent, evaluate_agent

# Train agent
agent, env = train_rl_agent(episodes=5000)

# Evaluate performance
accuracy = evaluate_agent(agent, num_questions=50)
print(f"Final accuracy: {accuracy:.3f}")
```

### Interactive Session
```python
# Load trained agent
agent = QLearningAgent()
agent.load_model("physics_rl_model_final.pkl")

# Answer questions interactively
interactive_demo()
```

## Dependencies

```
numpy>=1.21.0
matplotlib>=3.5.0
openai>=1.0.0  # For LLM integration
```

Install with:
```bash
pip install -r ../requirements.txt
```

## Testing

Run the test suite:
```bash
python test_rl_system.py
```

Tests cover:
- Environment functionality
- Agent learning capabilities
- Answer generation
- Model persistence
- Integration with question generator

## Extending the System

### Adding New Question Types

1. **Update Question Generator**: Add new question types to the physics question generator
2. **Modify State Encoding**: Update `_encode_state()` to handle new question types
3. **Extend Answer Generation**: Add logic for generating wrong answers for new types
4. **Test Integration**: Ensure the RL system can learn the new question types

### Custom Reward Functions

```python
def custom_reward_function(self, is_correct, question_type):
    if is_correct:
        # Higher rewards for harder questions
        if question_type in ['simple_collision', 'work_energy']:
            return 2.0
        else:
            return 1.0
    else:
        return -0.1
```

### Alternative Learning Algorithms

The system is designed to be modular. You can replace the Q-learning agent with:
- Deep Q-Networks (DQN)
- Policy Gradient methods
- Actor-Critic algorithms
- Multi-agent systems

## Performance Expectations

With default parameters, you can expect:

- **Initial Accuracy**: ~25% (random guessing with 4 options)
- **After 1000 episodes**: ~60-70% accuracy
- **After 10000 episodes**: ~80-90% accuracy
- **Convergence**: Usually within 50000 episodes

Performance varies based on:
- Learning rate settings
- Exploration strategy
- Question difficulty distribution
- Answer generation quality

## Troubleshooting

### Common Issues

1. **Low Accuracy**: Increase training episodes or adjust learning rate
2. **No Learning**: Check reward system and state encoding
3. **API Errors**: Verify OpenAI API key for LLM integration
4. **Memory Issues**: Reduce episode count or implement experience replay

### Debug Mode

Enable verbose logging:
```python
# Add debug prints in training loop
if episode % 10 == 0:
    print(f"Episode {episode}: Accuracy {accuracy:.3f}, Epsilon {agent.epsilon:.3f}")
```

## Future Enhancements

Potential improvements include:

- **Deep Learning**: Replace Q-table with neural networks
- **Multi-Step Problems**: Support for complex, multi-part questions
- **Curriculum Learning**: Progressive difficulty increase
- **Meta-Learning**: Learning to learn new physics concepts quickly
- **Explanation Generation**: Provide step-by-step solution explanations
- **Student Modeling**: Adapt to individual learning patterns

## Contributing

When contributing to this baseline system:

1. Maintain compatibility with the existing question generator
2. Preserve the modular architecture
3. Add comprehensive tests for new features
4. Update documentation for any API changes
5. Consider performance implications of modifications

## License

This project is part of the AI Physicist educational toolkit and follows the same licensing terms as the parent project.
