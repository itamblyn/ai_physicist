# AI Physicist

Toy problem for making an AI physicist from an LLM

## Physics Question Generator

A Python program that generates simple mechanics physics questions with detailed solutions. Perfect for students learning introductory physics concepts.

### Features

- **8 Question Types**: Covers fundamental mechanics topics
- **Randomized Parameters**: Each question uses different values for variety
- **Step-by-Step Solutions**: Shows complete work with formulas and calculations
- **Interactive Interface**: Choose question types or generate random problems
- **Educational Focus**: Clear, simple problems suitable for learning

### Question Types

1. **Kinematics - Velocity**: Calculate velocity from distance and time
2. **Kinematics - Acceleration**: Find acceleration from velocity change
3. **Kinematics - Displacement**: Motion with constant acceleration
4. **Force (Newton's 2nd Law)**: Calculate force using F = ma
5. **Kinetic Energy**: Energy of moving objects (KE = ½mv²)
6. **Potential Energy**: Gravitational potential energy (PE = mgh)
7. **Work-Energy**: Work done by constant forces (W = Fd)
8. **Simple Collisions**: Elastic collision problems in 1D

### How It Works

#### Core Architecture

The program is built around the `PhysicsQuestionGenerator` class:

```python
class PhysicsQuestionGenerator:
    def __init__(self):
        self.question_types = [
            'kinematics_velocity',
            'kinematics_acceleration', 
            'kinematics_displacement',
            'force_newton_second',
            'kinetic_energy',
            'potential_energy',
            'work_energy',
            'simple_collision'
        ]
```

#### Question Generation Process

1. **Random Parameter Selection**: Each question type uses `random.randint()` to generate realistic physics values:
   ```python
   distance = random.randint(50, 500)  # meters
   time = random.randint(5, 30)  # seconds
   ```

2. **Question Formatting**: Creates clear, readable problem statements:
   ```python
   question = f"A car travels {distance} meters in {time} seconds at constant speed. What is the car's velocity?"
   ```

3. **Solution Calculation**: Applies the appropriate physics formula:
   ```python
   velocity = distance / time
   ```

4. **Step-by-Step Solution**: Generates detailed explanation showing all work:
   ```python
   solution = f"""
   Solution:
   Using the formula: velocity = distance / time
   v = {distance} m / {time} s = {velocity:.2f} m/s
   
   Answer: {velocity:.2f} m/s
   """
   ```

5. **Return Structured Data**: Each question returns a dictionary with:
   - `question`: The problem statement
   - `solution`: Step-by-step solution
   - `answer`: Final numerical answer
   - `type`: Question category

#### Method Structure

Each question type has its own generation method following the pattern:
- `generate_[question_type]()`: Creates question, calculates answer, formats solution
- Uses appropriate physics formulas for the specific topic
- Includes proper units and significant figures
- Provides educational context in solutions

#### Example: Kinetic Energy Generator

```python
def generate_kinetic_energy(self):
    # Generate random parameters
    mass = random.randint(2, 20)  # kg
    velocity = random.randint(5, 30)  # m/s
    
    # Create question
    question = f"A {mass} kg object is moving at {velocity} m/s. What is its kinetic energy?"
    
    # Calculate answer
    kinetic_energy = 0.5 * mass * velocity**2
    
    # Format solution with steps
    solution = f"""
    Solution:
    Using the formula: Kinetic Energy = ½ × mass × velocity²
    KE = ½ × {mass} kg × ({velocity} m/s)²
    KE = ½ × {mass} × {velocity**2}
    KE = {kinetic_energy:.2f} J
    
    Answer: {kinetic_energy:.2f} J
    """
    
    return {
        'question': question,
        'solution': solution,
        'answer': f"{kinetic_energy:.2f} J",
        'type': 'kinetic_energy'
    }
```

### Usage

#### Interactive Mode
```bash
python physics_question_generator.py
```

Provides a menu-driven interface where users can:
- Generate random questions
- Choose specific question types
- Generate multiple questions at once
- View solutions step-by-step

#### Demo Mode
```bash
python demo_physics_questions.py
```

Shows one example from each question type plus random samples.

### Code Structure

```
physics_question_generator.py
├── PhysicsQuestionGenerator class
│   ├── __init__(): Initialize question types
│   ├── generate_question(): Main generation method
│   ├── generate_kinematics_velocity()
│   ├── generate_kinematics_acceleration()
│   ├── generate_kinematics_displacement()
│   ├── generate_force_newton_second()
│   ├── generate_kinetic_energy()
│   ├── generate_potential_energy()
│   ├── generate_work_energy()
│   └── generate_simple_collision()
└── main(): Interactive command-line interface

demo_physics_questions.py
├── demo_all_question_types(): Show one of each type
└── demo_random_questions(): Show random samples
```

### Educational Design

The program is designed with learning in mind:

- **Realistic Values**: Uses practical numbers students might encounter
- **Clear Units**: Always includes proper units (m/s, N, J, etc.)
- **Step-by-Step**: Shows formula, substitution, and calculation
- **Variety**: Random parameters prevent memorization of specific answers
- **Progressive Difficulty**: Questions build from basic kinematics to more complex topics

### Extensibility

Adding new question types is straightforward:
1. Add the question type name to `self.question_types`
2. Create a `generate_[new_type]()` method
3. Follow the established pattern for parameter generation and solution formatting

This modular design makes it easy to expand the program with additional physics topics like momentum, rotational motion, or thermodynamics.

## Reinforcement Learning System

The project now includes a complete reinforcement learning system that can train an AI agent to answer physics questions correctly using rewards.

### RL Components

#### 1. **Physics RL Environment** (`rl_physics_agent.py`)
- **Environment**: Manages physics questions and evaluates answers
- **State Encoding**: Converts questions into state representations
- **Reward System**: +1.0 for correct answers, -0.1 for incorrect
- **Answer Validation**: Compares agent answers with correct solutions

#### 2. **Q-Learning Agent**
- **Q-Table**: Learns state-action values through experience
- **Epsilon-Greedy Policy**: Balances exploration vs exploitation
- **Learning Rate**: Adjustable learning speed (default: 0.1)
- **Epsilon Decay**: Reduces exploration over time

#### 3. **Answer Generation System**
- **Multiple Choice**: Generates plausible wrong answers
- **Question-Specific**: Wrong answers based on physics concepts
- **Realistic Distractors**: Uses common calculation errors

#### 4. **LLM Integration** (`llm_rl_agent.py`)
- **OpenAI API**: Uses GPT models for answer generation
- **Context Learning**: Learns from previous attempts
- **Error Handling**: Fallback answers when API fails
- **Rate Limiting**: Prevents API overuse

### RL Training Process

#### Basic Q-Learning Training
```python
from rl_physics_agent import train_rl_agent

# Train agent for 1000 episodes
agent, environment = train_rl_agent(episodes=1000)

# Evaluate performance
evaluate_agent(agent, num_questions=100)
```

#### LLM-Based Training
```python
from llm_rl_agent import train_llm_agent

# Train with OpenAI API
agent, environment = train_llm_agent(
    api_key="your-api-key",
    episodes=50,
    model="gpt-3.5-turbo"
)
```

### Key Features

#### **Reward System**
- **Correct Answer**: +1.0 reward
- **Incorrect Answer**: -0.1 penalty
- **Learning Signal**: Clear feedback for improvement

#### **State Representation**
- **Question Type**: Identifies physics concept
- **Key Terms**: Extracts relevant physics vocabulary
- **Context**: Maintains conversation history (LLM version)

#### **Action Space**
- **Dynamic**: Generates answer options per question
- **Realistic**: Includes plausible wrong answers
- **Educational**: Helps learn common mistakes

#### **Learning Algorithm**
- **Q-Learning**: Model-free reinforcement learning
- **Experience Replay**: Learns from past attempts
- **Exploration**: Balances learning vs performance

### Usage Examples

#### **Basic RL Training**
```bash
# Train a simple Q-learning agent
python rl_physics_agent.py

# Interactive demo
python demo_rl_training.py
```

#### **LLM-Based Training**
```bash
# Train with OpenAI API
python llm_rl_agent.py
```

#### **Demo and Analysis**
```bash
# Full demo with plots
python demo_rl_training.py
# Choose option 5 for full training with visualizations
```

### Performance Metrics

The RL system tracks several key metrics:

- **Accuracy**: Percentage of correct answers
- **Learning Curve**: Improvement over time
- **Q-Value Analysis**: State-action value insights
- **Reward History**: Training signal progression

### Architecture Overview

```
Physics Question Generator
    ↓
RL Environment
    ↓
Q-Learning Agent ←→ Answer Generator
    ↓
Reward System
    ↓
Performance Metrics
```

### Advanced Features

#### **Context Learning (LLM Version)**
- Remembers previous attempts
- Learns from mistakes
- Improves over time

#### **Adaptive Learning**
- Epsilon decay for exploration
- Learning rate adjustment
- Model persistence

#### **Evaluation Tools**
- Training progress plots
- Q-table analysis
- Performance comparison

### Educational Applications

This RL system demonstrates:

1. **Reinforcement Learning**: How agents learn from rewards
2. **Physics Education**: Interactive problem-solving
3. **AI Training**: Real-world RL applications
4. **Adaptive Learning**: Systems that improve over time

The system can be extended for:
- Different subjects (chemistry, math, etc.)
- More complex question types
- Multi-step problem solving
- Collaborative learning environments
