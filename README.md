# AI Physicist 🤖⚛️

An educational AI system that generates physics problems and trains reinforcement learning agents to solve them. This project combines physics education with modern AI techniques to create an intelligent tutoring system.

## 🎯 **Project Overview**

The AI Physicist project explores how artificial intelligence can learn to solve physics problems through reinforcement learning, while also generating diverse datasets for training and evaluation. The system covers undergraduate-level mechanics problems and provides tools for creating specialized training datasets.

### **Key Features**

- 🔬 **Physics Problem Generation**: Automated creation of mechanics problems with step-by-step solutions
- 🧠 **Hybrid AI Agents**: Reinforcement learning agents enhanced with large language models
- 📊 **Specialized Datasets**: Tools for generating problems with extraneous information and unsolvable scenarios
- 🎓 **Educational Focus**: Designed with pedagogical principles for effective learning

## 📁 **Project Structure**

```
ai_physicist/
├── 01_generate_questions/     # Core physics problem generator
├── 02_baseline/              # RL agents and training system
├── 03_extraneous_info_dataset/  # Dataset with irrelevant information
├── 04_unsolvable/            # Inconsistent and unsolvable problems
├── requirements.txt          # Python dependencies
├── CODE_REVIEW.md           # Detailed code analysis and recommendations
└── README.md               # This file
```

### **Module Progression**

1. **Start Here**: `01_generate_questions/` - Learn how physics problems are generated
2. **Core System**: `02_baseline/` - Explore the RL training system
3. **Advanced Datasets**: `03_extraneous_info_dataset/` - Generate robust training data
4. **Research Tools**: `04_unsolvable/` - Create challenging evaluation scenarios

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- OpenAI API key (for LLM-enhanced agents)

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai_physicist.git
   cd ai_physicist
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API** (optional, for LLM features):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### **Basic Usage**

#### **Generate Physics Problems**
```bash
cd 01_generate_questions
python demo_physics_questions.py
```

#### **Train RL Agent**
```bash
cd 02_baseline
python demo_rl_training.py
```

#### **Generate Specialized Datasets**
```bash
cd 03_extraneous_info_dataset
python generate_extraneous_dataset.py --num_problems 100

cd 04_unsolvable
python generate_inconsistent_dataset.py --num_problems 50
```

## 🔬 **Physics Coverage**

The system currently supports these mechanics topics:

- **Kinematics**: Velocity, acceleration, motion equations
- **Dynamics**: Newton's laws, force calculations
- **Energy**: Kinetic energy, potential energy, work
- **Momentum**: Conservation of momentum, collisions
- **Circuits**: Basic electrical circuits (Ohm's law, power)

## 🤖 **AI Approaches**

### **Reinforcement Learning Agents**

1. **Traditional Q-Learning**: Tabular approach for discrete problem spaces
2. **LLM-Enhanced RL**: Hybrid system combining neural language models with RL
3. **Multi-Agent Training**: Comparative learning between different approaches

### **Training Features**

- Interactive learning environment
- Reward shaping for educational objectives
- Performance tracking and visualization
- Model persistence and evaluation

## 📊 **Dataset Generation**

### **Extraneous Information Dataset**
Creates solvable physics problems with intentionally irrelevant details to train models to focus on relevant information.

**Example**:
```
Original: "A 5kg object moves at 10 m/s. What is its kinetic energy?"
With Extraneous Info: "A red 5kg ball, owned by Sarah, moves at 10 m/s on a sunny Tuesday. What is its kinetic energy?"
```

### **Unsolvable Problems Dataset**
Generates two types of challenging problems:
- **Inconsistent**: Problems with contradictory information
- **Insufficient Information**: Problems missing key data for solution

## 📖 **Documentation**

Each module contains detailed documentation:

- **[01_generate_questions/README.md](01_generate_questions/README.md)**: Complete API documentation for problem generation
- **[02_baseline/README.md](02_baseline/README.md)**: Comprehensive RL system guide
- **[03_extraneous_info_dataset/README.md](03_extraneous_info_dataset/README.md)**: Dataset generation methodology
- **[04_unsolvable/README.md](04_unsolvable/README.md)**: Unsolvable problem creation guide
- **[CODE_REVIEW.md](CODE_REVIEW.md)**: Detailed code analysis and improvement roadmap

## 🛠️ **Development**

### **Current Status**

This project is in **active development** with a focus on educational applications and AI research. See [CODE_REVIEW.md](CODE_REVIEW.md) for detailed analysis and improvement recommendations.

### **Known Limitations**

- Limited to basic mechanics problems
- Q-table RL approach has scalability constraints
- Missing comprehensive testing framework
- No configuration management system

### **Roadmap**

- [ ] Implement comprehensive testing suite
- [ ] Add type hints and improve error handling
- [ ] Transition to deep reinforcement learning
- [ ] Expand physics coverage beyond mechanics
- [ ] Add configuration management system
- [ ] Create web-based interface

## 🤝 **Contributing**

We welcome contributions! Areas where help is needed:

1. **Testing**: Add unit tests and integration tests
2. **Physics**: Expand to new domains (thermodynamics, waves, optics)
3. **AI**: Improve RL algorithms and LLM integration
4. **Documentation**: Enhance tutorials and examples
5. **UI/UX**: Create web interface for easier interaction

### **Development Setup**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- Physics problem formulations based on standard undergraduate textbooks
- Reinforcement learning approaches inspired by educational AI research
- Dataset generation methodologies adapted from ML robustness literature

## 📞 **Contact & Support**

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check module-specific READMEs for detailed usage

---

**Note**: This project is designed for educational and research purposes. The AI agents are learning systems and may not always provide correct answers. Always verify physics solutions independently for critical applications.
