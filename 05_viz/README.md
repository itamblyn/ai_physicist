# AI Physicist Visualization Tools

This directory contains comprehensive visualization tools for analyzing the AI Physicist training datasets, RL model performance, and question characteristics.

## Overview

The visualization tools provide insights into:
- **Dataset Analysis**: Statistics and distributions across different physics question datasets
- **RL Training Progress**: Learning curves, model comparisons, and Q-table analysis
- **Question Analysis**: Difficulty assessment, physics concept usage, and complexity patterns
- **Performance Dashboards**: Unified views of training progress and system performance

## Quick Start

### Basic Usage

```bash
# Generate all visualizations
python main.py --all

# Create comprehensive dashboard
python main.py --dashboard comprehensive

# Analyze specific components
python main.py --datasets --rl --questions
```

### Focused Analysis

```bash
# Dataset analysis only
python main.py --focus datasets

# RL training analysis only
python main.py --focus rl --model physics_rl_model_final.pkl

# Question analysis only
python main.py --focus questions --difficulty --concepts
```

## Available Tools

### 1. Dataset Analyzer (`dataset_analyzer.py`)

Analyzes the physics question datasets and provides statistical insights.

**Features:**
- Dataset size and category distribution analysis
- Extraneous vs important facts analysis
- Preference learning data analysis
- Inconsistency and solvability analysis

**Usage:**
```bash
python main.py --datasets
python dataset_analyzer.py --output-dir ./analysis
```

### 2. RL Visualizer (`rl_visualizer.py`)

Visualizes reinforcement learning training progress and model performance.

**Features:**
- Training accuracy and reward curves
- Model comparison across different checkpoints
- Q-table analysis and visualization
- Learning curves with confidence intervals

**Usage:**
```bash
python main.py --rl --model physics_rl_model_final.pkl
python rl_visualizer.py --compare --q-analysis
```

### 3. Question Analyzer (`question_analyzer.py`)

Analyzes physics question content, difficulty, and patterns.

**Features:**
- Question difficulty assessment and distribution
- Physics concept usage analysis
- Text complexity and length analysis
- Category-specific difficulty patterns

**Usage:**
```bash
python main.py --questions --difficulty --concepts
python question_analyzer.py --all --output-dir ./analysis
```

### 4. Performance Dashboard (`performance_dashboard.py`)

Creates comprehensive dashboards combining multiple analysis views.

**Features:**
- Comprehensive overview dashboard
- Focused dashboards for specific analysis areas
- Real-time performance monitoring
- Summary statistics and reports

**Usage:**
```bash
python main.py --dashboard comprehensive
python performance_dashboard.py --dashboard datasets
```

## Command Line Options

### Main Script (`main.py`)

```bash
python main.py [OPTIONS]

Options:
  --data-dir DIR          Base directory containing dataset files
  --output-dir DIR        Output directory for plots and reports
  --datasets              Analyze and visualize datasets
  --rl                    Visualize RL training progress
  --questions             Analyze question content and difficulty
  --dashboard TYPE        Create specific dashboard (comprehensive|datasets|rl|questions)
  --model FILE            Specific RL model file to analyze
  --compare-models        Compare multiple RL models
  --q-analysis            Analyze Q-table from RL model
  --learning-curves       Generate learning curves with confidence intervals
  --difficulty            Generate difficulty distribution plots
  --concepts              Generate physics concept usage plots
  --complexity            Generate complexity heatmap
  --text-length           Generate text length analysis
  --all                   Generate all visualizations
  --focus AREA            Focus on specific analysis area (datasets|rl|questions)
  --report-only           Generate only text reports, no plots
```

### Individual Tools

Each tool can be run independently with its own command-line interface:

```bash
# Dataset analyzer
python dataset_analyzer.py --help

# RL visualizer
python rl_visualizer.py --help

# Question analyzer
python question_analyzer.py --help

# Performance dashboard
python performance_dashboard.py --help
```

## Output Files

The tools generate various output files in the specified output directory:

### Plots and Visualizations
- `dataset_overview.png` - Dataset size and category overview
- `category_comparison.png` - Category distribution comparison
- `rl_training_progress.png` - RL training accuracy and rewards
- `rl_model_comparison.png` - Comparison of multiple RL models
- `rl_learning_curves.png` - Learning curves with confidence intervals
- `rl_q_table_analysis.png` - Q-table analysis and visualization
- `question_difficulty_distribution.png` - Question difficulty distribution
- `physics_concept_usage.png` - Physics concept usage patterns
- `question_complexity_heatmap.png` - Complexity heatmap by category
- `question_text_length_analysis.png` - Text length analysis
- `comprehensive_dashboard.png` - Comprehensive performance dashboard
- `{focus}_dashboard.png` - Focused dashboards

### Reports
- `dataset_analysis_report.txt` - Detailed dataset statistics
- `rl_performance_report.txt` - RL model performance metrics
- `question_analysis_report.txt` - Question content analysis
- `dashboard_report.txt` - Dashboard summary report

## Examples

### Complete Analysis Workflow

```bash
# 1. Generate all visualizations
python main.py --all --output-dir ./complete_analysis

# 2. Create focused dashboards
python main.py --dashboard comprehensive --output-dir ./dashboards
python main.py --dashboard rl --output-dir ./rl_analysis
python main.py --dashboard questions --output-dir ./question_analysis

# 3. Generate reports only (no plots)
python main.py --all --report-only --output-dir ./reports
```

### RL Training Analysis

```bash
# Analyze specific model
python main.py --rl --model physics_rl_model_final.pkl --q-analysis --learning-curves

# Compare multiple models
python main.py --rl --compare-models

# Generate RL-focused dashboard
python main.py --dashboard rl
```

### Question Analysis

```bash
# Analyze question difficulty and concepts
python main.py --questions --difficulty --concepts --complexity

# Generate question-focused dashboard
python main.py --dashboard questions
```

### Dataset Analysis

```bash
# Analyze all datasets
python main.py --datasets

# Generate dataset-focused dashboard
python main.py --dashboard datasets
```

## Dependencies

The visualization tools require the following Python packages:

```
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
seaborn>=0.11.0
nltk>=3.7
```

Install with:
```bash
pip install -r ../requirements.txt
```

## Data Structure

The tools expect the following directory structure:

```
ai_physicist/
├── 02_baseline/           # RL models and training data
│   ├── *.pkl             # Trained RL models
│   └── ...
├── 03_extraneous_info_dataset/samples/  # Extraneous info datasets
│   ├── supervised.jsonl
│   ├── preference.jsonl
│   └── ...
├── 04_unsolvable/samples/  # Unsolvable/inconsistent datasets
│   ├── unsolvable.jsonl
│   ├── solvability.jsonl
│   └── ...
└── 05_viz/               # This visualization directory
    ├── main.py
    ├── dataset_analyzer.py
    ├── rl_visualizer.py
    ├── question_analyzer.py
    └── performance_dashboard.py
```

## Customization

### Adding New Visualizations

To add new visualization types:

1. Create a new analysis function in the appropriate module
2. Add command-line options in `main.py`
3. Update the dashboard if needed
4. Add documentation

### Custom Data Sources

To analyze data from different sources:

1. Modify the data loading functions in each analyzer
2. Update the file path configurations
3. Adjust the data structure parsing as needed

## Troubleshooting

### Common Issues

1. **No data found**: Ensure the data directory structure is correct
2. **Import errors**: Install required dependencies
3. **Plot generation errors**: Check matplotlib backend configuration
4. **Memory issues**: Reduce dataset size or use sampling

### Debug Mode

Enable verbose output:
```bash
python main.py --all --output-dir ./debug_output 2>&1 | tee debug.log
```

## Contributing

When adding new visualization features:

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings and type hints
3. Include command-line interface support
4. Update this README with new features
5. Test with sample data before committing

## License

This visualization toolkit is part of the AI Physicist project and follows the same licensing terms.
