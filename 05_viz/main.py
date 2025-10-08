#!/usr/bin/env python3
"""
Main Visualization Script for AI Physicist Training

This script provides a unified interface for all visualization tools
in the AI Physicist project.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))

from dataset_analyzer import DatasetAnalyzer
from rl_visualizer import RLVisualizer
from question_analyzer import QuestionAnalyzer
from performance_dashboard import PerformanceDashboard


def main():
    """Main function for the visualization tools."""
    parser = argparse.ArgumentParser(
        description='AI Physicist Training Visualization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comprehensive dashboard
  python main.py --dashboard comprehensive

  # Analyze datasets only
  python main.py --datasets

  # Visualize RL training progress
  python main.py --rl --model physics_rl_model_final.pkl

  # Analyze question difficulty
  python main.py --questions --difficulty

  # Generate all visualizations
  python main.py --all

  # Focus on specific analysis
  python main.py --focus datasets --output-dir ./my_analysis
        """
    )
    
    # Main options
    parser.add_argument('--data-dir', type=str, 
                       help='Base directory containing dataset files (default: project structure)')
    parser.add_argument('--output-dir', type=str, default='./viz_output',
                       help='Output directory for plots and reports (default: ./viz_output)')
    
    # Analysis options
    parser.add_argument('--datasets', action='store_true',
                       help='Analyze and visualize datasets')
    parser.add_argument('--rl', action='store_true',
                       help='Visualize RL training progress and models')
    parser.add_argument('--questions', action='store_true',
                       help='Analyze question content and difficulty')
    parser.add_argument('--dashboard', choices=['comprehensive', 'datasets', 'rl', 'questions'],
                       help='Create specific dashboard type')
    
    # RL-specific options
    parser.add_argument('--model', type=str,
                       help='Specific RL model file to analyze')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare multiple RL models')
    parser.add_argument('--q-analysis', action='store_true',
                       help='Analyze Q-table from RL model')
    parser.add_argument('--learning-curves', action='store_true',
                       help='Generate learning curves with confidence intervals')
    
    # Question analysis options
    parser.add_argument('--difficulty', action='store_true',
                       help='Generate difficulty distribution plots')
    parser.add_argument('--concepts', action='store_true',
                       help='Generate physics concept usage plots')
    parser.add_argument('--complexity', action='store_true',
                       help='Generate complexity heatmap')
    parser.add_argument('--text-length', action='store_true',
                       help='Generate text length analysis')
    
    # Convenience options
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualizations')
    parser.add_argument('--focus', choices=['datasets', 'rl', 'questions'],
                       help='Focus on specific analysis area')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate only text reports, no plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize analyzers
    dataset_analyzer = DatasetAnalyzer(args.data_dir)
    rl_visualizer = RLVisualizer(args.data_dir)
    question_analyzer = QuestionAnalyzer(args.data_dir)
    dashboard = PerformanceDashboard(args.data_dir)
    
    # Determine what to run
    run_datasets = args.datasets or args.all or args.focus == 'datasets'
    run_rl = args.rl or args.all or args.focus == 'rl'
    run_questions = args.questions or args.all or args.focus == 'questions'
    run_dashboard = args.dashboard is not None
    
    # Run analyses
    if run_datasets:
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        if not args.report_only:
            print("Generating dataset overview...")
            dataset_analyzer.plot_dataset_overview(
                os.path.join(args.output_dir, 'dataset_overview.png')
            )
            
            print("Generating category comparison...")
            dataset_analyzer.plot_category_comparison(
                os.path.join(args.output_dir, 'category_comparison.png')
            )
        
        print("Generating dataset analysis report...")
        report = dataset_analyzer.generate_report(
            os.path.join(args.output_dir, 'dataset_analysis_report.txt')
        )
        print(report)
    
    if run_rl:
        print("\n" + "="*60)
        print("RL TRAINING VISUALIZATION")
        print("="*60)
        
        if not args.report_only:
            print("Generating RL training progress...")
            rl_visualizer.plot_training_progress(
                args.model,
                os.path.join(args.output_dir, 'rl_training_progress.png')
            )
            
            if args.compare_models:
                print("Comparing RL models...")
                rl_visualizer.compare_models(
                    save_path=os.path.join(args.output_dir, 'rl_model_comparison.png')
                )
            
            if args.learning_curves:
                print("Generating learning curves...")
                rl_visualizer.plot_learning_curves(
                    args.model,
                    save_path=os.path.join(args.output_dir, 'rl_learning_curves.png')
                )
            
            if args.q_analysis:
                print("Analyzing Q-table...")
                rl_visualizer.analyze_q_table(
                    args.model,
                    save_path=os.path.join(args.output_dir, 'rl_q_table_analysis.png')
                )
        
        print("Generating RL performance report...")
        report = rl_visualizer.generate_performance_report(
            args.model,
            os.path.join(args.output_dir, 'rl_performance_report.txt')
        )
        print(report)
    
    if run_questions:
        print("\n" + "="*60)
        print("QUESTION ANALYSIS")
        print("="*60)
        
        if not args.report_only:
            if args.difficulty or args.all:
                print("Generating difficulty distribution...")
                question_analyzer.plot_question_difficulty_distribution(
                    os.path.join(args.output_dir, 'question_difficulty_distribution.png')
                )
            
            if args.concepts or args.all:
                print("Generating physics concept usage...")
                question_analyzer.plot_physics_concept_usage(
                    os.path.join(args.output_dir, 'physics_concept_usage.png')
                )
            
            if args.complexity or args.all:
                print("Generating complexity heatmap...")
                question_analyzer.plot_question_complexity_heatmap(
                    os.path.join(args.output_dir, 'question_complexity_heatmap.png')
                )
            
            if args.text_length or args.all:
                print("Generating text length analysis...")
                question_analyzer.plot_text_length_analysis(
                    os.path.join(args.output_dir, 'question_text_length_analysis.png')
                )
        
        print("Generating question analysis report...")
        report = question_analyzer.generate_question_analysis_report(
            os.path.join(args.output_dir, 'question_analysis_report.txt')
        )
        print(report)
    
    if run_dashboard:
        print("\n" + "="*60)
        print("PERFORMANCE DASHBOARD")
        print("="*60)
        
        if not args.report_only:
            print(f"Creating {args.dashboard} dashboard...")
            if args.dashboard == 'comprehensive':
                dashboard.create_comprehensive_dashboard(
                    os.path.join(args.output_dir, 'comprehensive_dashboard.png')
                )
            else:
                dashboard.create_focused_dashboard(
                    args.dashboard,
                    os.path.join(args.output_dir, f'{args.dashboard}_dashboard.png')
                )
        
        print("Generating dashboard report...")
        report = dashboard.generate_dashboard_report(
            os.path.join(args.output_dir, 'dashboard_report.txt')
        )
        print(report)
    
    # If no specific options were given, show help
    if not any([run_datasets, run_rl, run_questions, run_dashboard]):
        print("No analysis options specified. Use --help for available options.")
        print("\nQuick start examples:")
        print("  python main.py --all                    # Generate all visualizations")
        print("  python main.py --dashboard comprehensive # Create comprehensive dashboard")
        print("  python main.py --datasets --rl          # Analyze datasets and RL training")
        print("  python main.py --questions --difficulty # Analyze question difficulty")
    
    print(f"\nAnalysis complete! Check the output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
