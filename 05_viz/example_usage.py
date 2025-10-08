#!/usr/bin/env python3
"""
Example Usage of AI Physicist Visualization Tools

This script demonstrates how to use the visualization tools programmatically.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_analyzer import DatasetAnalyzer
from rl_visualizer import RLVisualizer
from question_analyzer import QuestionAnalyzer
from performance_dashboard import PerformanceDashboard


def example_dataset_analysis():
    """Example of dataset analysis."""
    print("=" * 60)
    print("EXAMPLE: Dataset Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Analyze extraneous dataset
    print("Analyzing extraneous information dataset...")
    extraneous_analysis = analyzer.analyze_extraneous_dataset()
    
    print(f"Supervised records: {extraneous_analysis['supervised']['total_records']}")
    print(f"Preference records: {extraneous_analysis['preference']['total_records']}")
    print(f"Categories: {extraneous_analysis['supervised']['categories']}")
    
    # Generate plots
    print("Generating dataset overview plot...")
    analyzer.plot_dataset_overview('example_dataset_overview.png')
    
    print("Generating category comparison plot...")
    analyzer.plot_category_comparison('example_category_comparison.png')
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_report('example_dataset_report.txt')
    print("Report generated successfully!")


def example_question_analysis():
    """Example of question analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Question Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = QuestionAnalyzer()
    
    # Analyze questions from different datasets
    datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset}...")
        analysis = analyzer.analyze_dataset_questions(dataset)
        
        if analysis:
            print(f"  Total questions: {analysis['total_questions']}")
            print(f"  Average difficulty: {analysis['average_difficulty_score']:.2f}")
            print(f"  Difficulty levels: {analysis['difficulty_distribution']}")
    
    # Generate plots
    print("\nGenerating difficulty distribution plot...")
    analyzer.plot_question_difficulty_distribution('example_difficulty_distribution.png')
    
    print("Generating physics concept usage plot...")
    analyzer.plot_physics_concept_usage('example_concept_usage.png')


def example_rl_analysis():
    """Example of RL analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE: RL Analysis")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = RLVisualizer()
    
    # Find available models
    model_files = visualizer.find_model_files()
    print(f"Available RL models: {model_files}")
    
    if model_files:
        # Analyze the latest model
        latest_model = model_files[-1]
        print(f"Analyzing model: {latest_model}")
        
        # Generate training progress plot
        print("Generating training progress plot...")
        visualizer.plot_training_progress(latest_model, 'example_rl_progress.png')
        
        # Generate performance report
        print("Generating performance report...")
        report = visualizer.generate_performance_report(latest_model, 'example_rl_report.txt')
        print("RL analysis complete!")
    else:
        print("No RL models found. Train a model first using the baseline system.")


def example_dashboard():
    """Example of dashboard creation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Performance Dashboard")
    print("=" * 60)
    
    # Initialize dashboard
    dashboard = PerformanceDashboard()
    
    # Create comprehensive dashboard
    print("Creating comprehensive dashboard...")
    dashboard.create_comprehensive_dashboard('example_comprehensive_dashboard.png')
    
    # Create focused dashboards
    print("Creating dataset-focused dashboard...")
    dashboard.create_focused_dashboard('datasets', 'example_datasets_dashboard.png')
    
    print("Creating question-focused dashboard...")
    dashboard.create_focused_dashboard('questions', 'example_questions_dashboard.png')
    
    # Generate dashboard report
    print("Generating dashboard report...")
    report = dashboard.generate_dashboard_report('example_dashboard_report.txt')
    print("Dashboard creation complete!")


def main():
    """Run all examples."""
    print("AI Physicist Visualization Tools - Example Usage")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('example_output', exist_ok=True)
    os.chdir('example_output')
    
    try:
        # Run examples
        example_dataset_analysis()
        example_question_analysis()
        example_rl_analysis()
        example_dashboard()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        
        # List generated files
        for file in os.listdir('.'):
            if file.endswith(('.png', '.txt')):
                print(f"  - {file}")
        
        print(f"\nCheck the 'example_output' directory for all generated files.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
