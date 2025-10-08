# AI Physicist

A comprehensive toolkit for training and evaluating AI systems on physics problem-solving tasks. This project provides multiple approaches to creating AI physics tutors, from basic question generation to advanced reinforcement learning systems and specialized datasets for training robust physics-solving models.

## 🎯 Project Overview

AI Physicist is designed as an educational and research platform that demonstrates various approaches to automated physics problem-solving:

- **Question Generation**: Automated creation of physics problems with step-by-step solutions
- **Reinforcement Learning**: Training agents to learn physics through trial and error
- **Dataset Creation**: Specialized datasets for training robust AI systems
- **Educational Applications**: Interactive physics tutoring systems

## 📁 Project Structure

```
ai_physicist/
├── 01_generate_questions/     # Physics question generator with solutions
├── 02_baseline/              # Reinforcement learning baseline system
├── 03_extraneous_info_dataset/  # Dataset with irrelevant information
├── 04_unsolvable/            # Inconsistent and unsolvable problems
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai_physicist

# Install dependencies
pip install -r requirements.txt

# For LLM integration (optional)
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# Generate physics questions
python 01_generate_questions/physics_question_generator.py

# Train a reinforcement learning agent
python 02_baseline/demo_rl_training.py

# Create specialized datasets
python 03_extraneous_info_dataset/generate_extraneous_dataset.py --count 100
python 04_unsolvable/generate_inconsistent_dataset.py --count 50
```

## 📚 Modules

### 1. Question Generator (`01_generate_questions/`)

**Purpose**: Generate educational physics problems with detailed solutions

**Features**:
- 8 fundamental physics question types (kinematics, forces, energy, etc.)
- Randomized parameters for variety
- Step-by-step solutions with proper units
- Interactive command-line interface

**Question Types**:
- Kinematics (velocity, acceleration, displacement)
- Newton's Laws (force calculations)
- Energy (kinetic and potential energy)
- Work-Energy theorem
- Simple collisions

**Usage**:
```bash
cd 01_generate_questions/
python physics_question_generator.py  # Interactive mode
python demo_physics_questions.py      # Demo all types
```

### 2. Baseline RL System (`02_baseline/`)

**Purpose**: Train AI agents to answer physics questions using reinforcement learning

**Features**:
- Q-learning agent with epsilon-greedy exploration
- LLM-enhanced agents using OpenAI API
- Interactive training demos with visualization
- Model persistence and evaluation tools
- Comprehensive test suite

**Key Components**:
- `PhysicsRLEnvironment`: Manages questions and rewards
- `QLearningAgent`: Traditional Q-learning implementation
- `LLMRLAgent`: LLM-enhanced learning with context
- `PhysicsAnswerGenerator`: Creates multiple choice options

**Usage**:
```bash
cd 02_baseline/
python rl_physics_agent.py           # Basic Q-learning
python llm_rl_agent.py              # LLM-enhanced training
python demo_rl_training.py          # Interactive demo
```

**Performance**: Achieves 80-90% accuracy after 10,000 training episodes

### 3. Extraneous Info Dataset (`03_extraneous_info_dataset/`)

**Purpose**: Create datasets with irrelevant information to train robust AI systems

**Features**:
- Physics problems with extraneous details (colors, names, ambient conditions)
- Supervised learning format (JSONL)
- Preference pairs for RLHF/DPO training
- Multiple physics categories (mechanics, circuits)

**Output Formats**:
- `supervised.jsonl`: Standard training format with solutions
- `preference.jsonl`: Preference pairs (chosen vs rejected responses)
- Schema files for data validation

**Usage**:
```bash
cd 03_extraneous_info_dataset/
python generate_extraneous_dataset.py --count 100 --outdir samples/
```

**Applications**:
- Fine-tuning language models to ignore irrelevant information
- Training reward models for RLHF
- Evaluating robustness to distractors

### 4. Unsolvable Problems (`04_unsolvable/`)

**Purpose**: Generate inconsistent and unsolvable physics problems for robustness testing

**Features**:
- **Inconsistent Problems**: Internal contradictions and unit conflicts
- **Solvability Labels**: Binary classification of problem solvability
- Length normalization to prevent trivial solutions
- Detailed inconsistency annotations

**Two Generators**:
1. **Inconsistent Dataset**: Problems with explicit contradictions
2. **Solvability Dataset**: Mix of solvable/unsolvable with binary labels

**Usage**:
```bash
cd 04_unsolvable/
python generate_inconsistent_dataset.py --count 50 --outdir samples/
python generate_solvability_dataset.py --count 200 --outdir samples/
```

**Applications**:
- Training models to detect inconsistencies
- Robustness evaluation
- Problem validation systems

## 🎓 Educational Applications

### For Students
- **Interactive Problem Solving**: Get instant feedback on physics problems
- **Step-by-Step Solutions**: Learn proper problem-solving techniques
- **Adaptive Learning**: RL agents that improve with practice

### For Educators
- **Question Generation**: Create unlimited practice problems
- **Assessment Tools**: Evaluate student understanding
- **Curriculum Design**: Progressive difficulty levels

### For Researchers
- **AI Training**: Multiple approaches to physics problem-solving
- **Dataset Creation**: Specialized datasets for robustness research
- **Benchmarking**: Standardized evaluation metrics

## 🔬 Research Applications

### Machine Learning Research
- **Reinforcement Learning**: Educational domain for RL algorithms
- **Robustness Testing**: Datasets with distractors and inconsistencies
- **Multi-Modal Learning**: Combining symbolic and neural approaches

### Physics Education Research
- **Automated Tutoring**: AI-powered physics instruction
- **Misconception Detection**: Identifying common student errors
- **Adaptive Assessment**: Personalized difficulty adjustment

### AI Safety Research
- **Consistency Checking**: Detecting contradictory information
- **Robustness Evaluation**: Performance under adversarial conditions
- **Alignment Research**: Training AI to focus on relevant information

## 📊 Performance Metrics

### Question Generator
- **Coverage**: 8 fundamental physics concepts
- **Accuracy**: Mathematically verified solutions
- **Variety**: Randomized parameters prevent memorization

### RL System
- **Learning Efficiency**: 80-90% accuracy in 10K episodes
- **Robustness**: Handles all question types
- **Scalability**: Modular design for easy extension

### Datasets
- **Scale**: Configurable from 50 to 10,000+ problems
- **Quality**: Verified physics with controlled inconsistencies
- **Format**: Standard JSONL for ML pipelines

## 🛠️ Technical Details

### Dependencies
```
numpy>=1.21.0      # Numerical computations
matplotlib>=3.5.0  # Visualization
openai>=1.0.0      # LLM integration (optional)
```

### System Requirements
- Python 3.8+
- 4GB RAM (for large dataset generation)
- OpenAI API key (for LLM features)

### Architecture
- **Modular Design**: Independent components for easy extension
- **Standard Formats**: JSONL output for ML compatibility
- **Configurable**: Command-line arguments for all parameters
- **Extensible**: Clear patterns for adding new question types

## 🔧 Extending the System

### Adding New Question Types
1. **Question Generator**: Add methods to `PhysicsQuestionGenerator`
2. **RL Environment**: Update state encoding for new types
3. **Answer Generation**: Create plausible wrong answers
4. **Dataset Generators**: Include in specialized datasets

### Custom Learning Algorithms
- Replace Q-learning with deep RL methods
- Implement curriculum learning
- Add multi-agent systems
- Integrate with modern LLM architectures

### New Dataset Types
- Multi-step problems
- Visual physics problems
- Cross-domain transfer
- Temporal reasoning tasks

## 📈 Future Enhancements

### Planned Features
- **Deep Learning Integration**: Neural network-based agents
- **Visual Problem Solving**: Diagram interpretation
- **Multi-Step Reasoning**: Complex, multi-part problems
- **Explanation Generation**: Detailed solution explanations

### Research Directions
- **Meta-Learning**: Learning to learn new physics concepts
- **Transfer Learning**: Cross-domain knowledge application
- **Collaborative Learning**: Multi-agent problem solving
- **Uncertainty Quantification**: Confidence in solutions

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

1. **New Question Types**: Expand physics coverage
2. **Advanced RL**: Modern deep RL algorithms
3. **Evaluation Metrics**: Better assessment methods
4. **Educational Tools**: User interface improvements
5. **Documentation**: Examples and tutorials

### Development Guidelines
- Maintain modular architecture
- Add comprehensive tests
- Update documentation
- Follow existing code patterns
- Consider educational applications

## 📄 License

This project is open-source and available under [appropriate license]. See LICENSE file for details.

## 🙏 Acknowledgments

This project demonstrates the intersection of AI and physics education, inspired by the need for intelligent tutoring systems and robust AI training methods.

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on the project repository.

---

**AI Physicist** - Where artificial intelligence meets physics education 🚀🔬
