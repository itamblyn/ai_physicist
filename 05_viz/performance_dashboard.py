#!/usr/bin/env python3
"""
Performance Dashboard for AI Physicist Training

This module provides a comprehensive dashboard for visualizing training performance,
model comparisons, and dataset statistics in a unified interface.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from datetime import datetime
import json

from dataset_analyzer import DatasetAnalyzer
from rl_visualizer import RLVisualizer
from question_analyzer import QuestionAnalyzer


class PerformanceDashboard:
    """Comprehensive performance dashboard for AI Physicist training."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the performance dashboard.
        
        Args:
            data_dir: Base directory containing all data files. If None, uses project structure.
        """
        self.data_dir = data_dir
        self.dataset_analyzer = DatasetAnalyzer(data_dir)
        self.rl_visualizer = RLVisualizer(data_dir)
        self.question_analyzer = QuestionAnalyzer(data_dir)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, save_path: str = None):
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('AI Physicist Training Performance Dashboard', fontsize=20, fontweight='bold')
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Dataset Overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_dataset_overview(ax1)
        
        # 2. RL Training Progress (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_rl_progress(ax2)
        
        # 3. Question Difficulty Distribution (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_difficulty_distribution(ax3)
        
        # 4. Physics Concept Usage (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_concept_usage(ax4)
        
        # 5. Model Performance Comparison (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_model_comparison(ax5)
        
        # 6. Question Complexity Heatmap (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_complexity_heatmap(ax6)
        
        # 7. Training Statistics Summary (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_training_summary(ax7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_dataset_overview(self, ax):
        """Plot dataset overview statistics."""
        try:
            extraneous_analysis = self.dataset_analyzer.analyze_extraneous_dataset()
            unsolvable_analysis = self.dataset_analyzer.analyze_unsolvable_dataset()
            
            # Prepare data
            datasets = ['Extraneous\nSupervised', 'Extraneous\nPreference', 'Unsolvable\nInconsistent', 'Solvability\nClassification']
            counts = [
                extraneous_analysis.get('supervised', {}).get('total_records', 0),
                extraneous_analysis.get('preference', {}).get('total_records', 0),
                unsolvable_analysis.get('unsolvable', {}).get('total_records', 0),
                unsolvable_analysis.get('solvability', {}).get('total_records', 0)
            ]
            
            bars = ax.bar(datasets, counts, alpha=0.8, color=['skyblue', 'lightblue', 'lightcoral', 'lightpink'])
            ax.set_title('Dataset Size Overview', fontweight='bold')
            ax.set_ylabel('Number of Records')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading dataset overview:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dataset Overview (Error)', fontweight='bold')
    
    def _plot_rl_progress(self, ax):
        """Plot RL training progress."""
        try:
            model_files = self.rl_visualizer.find_model_files()
            if not model_files:
                ax.text(0.5, 0.5, 'No RL models found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('RL Training Progress (No Data)', fontweight='bold')
                return
            
            # Use the most recent model
            model_data = self.rl_visualizer.load_training_data(model_files[-1])
            training_accuracy = getattr(model_data, 'training_accuracy', [])
            
            if training_accuracy:
                episodes = range(0, len(training_accuracy) * 10, 10)
                ax.plot(episodes, training_accuracy, 'b-', linewidth=2, alpha=0.8, label='Accuracy')
                
                # Add moving average
                if len(training_accuracy) > 10:
                    window_size = min(50, len(training_accuracy) // 10)
                    moving_avg = pd.Series(training_accuracy).rolling(window=window_size).mean()
                    ax.plot(episodes, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Avg')
                
                ax.set_title('RL Training Progress', fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Accuracy')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('RL Training Progress (No Data)', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading RL data:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('RL Training Progress (Error)', fontweight='bold')
    
    def _plot_difficulty_distribution(self, ax):
        """Plot question difficulty distribution."""
        try:
            datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
            difficulty_data = {}
            
            for dataset in datasets:
                analysis = self.question_analyzer.analyze_dataset_questions(dataset)
                if analysis and 'difficulty_distribution' in analysis:
                    difficulty_data[dataset] = analysis['difficulty_distribution']
            
            if difficulty_data:
                # Create stacked bar chart
                difficulty_levels = ['Easy', 'Medium', 'Hard', 'Very Hard']
                colors = ['lightgreen', 'yellow', 'orange', 'red']
                
                bottom = np.zeros(len(datasets))
                for i, level in enumerate(difficulty_levels):
                    values = [difficulty_data.get(dataset, {}).get(level, 0) for dataset in datasets]
                    ax.bar(range(len(datasets)), values, bottom=bottom, 
                          label=level, color=colors[i], alpha=0.8)
                    bottom += values
                
                ax.set_title('Question Difficulty Distribution', fontweight='bold')
                ax.set_ylabel('Number of Questions')
                ax.set_xticks(range(len(datasets)))
                ax.set_xticklabels([d.replace('_', '\n').title() for d in datasets])
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No difficulty data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Difficulty Distribution (No Data)', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading difficulty data:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Difficulty Distribution (Error)', fontweight='bold')
    
    def _plot_concept_usage(self, ax):
        """Plot physics concept usage."""
        try:
            datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
            concepts = ['kinematics', 'newton', 'energy', 'momentum', 'circuits']
            
            # Collect concept usage data
            concept_usage = {}
            for concept in concepts:
                concept_usage[concept] = []
                for dataset in datasets:
                    analysis = self.question_analyzer.analyze_dataset_questions(dataset)
                    if analysis and 'physics_terms_stats' in analysis:
                        # This is a simplified version - in practice, you'd need to load raw data
                        concept_usage[concept].append(analysis['physics_terms_stats']['mean'])
                    else:
                        concept_usage[concept].append(0)
            
            # Create grouped bar chart
            x = np.arange(len(concepts))
            width = 0.2
            
            for i, dataset in enumerate(datasets):
                values = [concept_usage[concept][i] for concept in concepts]
                ax.bar(x + i * width, values, width, label=dataset.replace('_', ' ').title(), alpha=0.8)
            
            ax.set_title('Physics Concept Usage', fontweight='bold')
            ax.set_ylabel('Average Usage per Question')
            ax.set_xlabel('Physics Concepts')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(concepts)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading concept data:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Concept Usage (Error)', fontweight='bold')
    
    def _plot_model_comparison(self, ax):
        """Plot model performance comparison."""
        try:
            model_files = self.rl_visualizer.find_model_files()
            if len(model_files) < 2:
                ax.text(0.5, 0.5, 'Need at least 2 models to compare', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Comparison (Insufficient Data)', fontweight='bold')
                return
            
            # Load final accuracies
            final_accuracies = []
            model_names = []
            
            for model_file in model_files[-5:]:  # Compare last 5 models
                model_data = self.rl_visualizer.load_training_data(model_file)
                training_accuracy = getattr(model_data, 'training_accuracy', [])
                if training_accuracy:
                    final_accuracies.append(training_accuracy[-1])
                    model_names.append(model_file.replace('.pkl', '').replace('physics_rl_model_', ''))
            
            if final_accuracies:
                bars = ax.bar(model_names, final_accuracies, alpha=0.8, color='lightblue')
                ax.set_title('Model Performance Comparison', fontweight='bold')
                ax.set_ylabel('Final Accuracy')
                ax.set_xlabel('Model')
                
                # Add value labels
                for bar, acc in zip(bars, final_accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Comparison (No Data)', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading model data:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Comparison (Error)', fontweight='bold')
    
    def _plot_complexity_heatmap(self, ax):
        """Plot question complexity heatmap."""
        try:
            # This is a simplified version - in practice, you'd load detailed data
            datasets = ['Extraneous\nSupervised', 'Extraneous\nPreference', 'Unsolvable', 'Solvability']
            categories = ['Kinematics', 'Newton', 'Energy', 'Momentum', 'Circuits']
            
            # Create mock complexity matrix (replace with real data)
            complexity_matrix = np.random.uniform(2, 8, (len(datasets), len(categories)))
            
            im = ax.imshow(complexity_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(categories)))
            ax.set_yticks(range(len(datasets)))
            ax.set_xticklabels(categories)
            ax.set_yticklabels(datasets)
            ax.set_title('Question Complexity Heatmap', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Complexity Score')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Complexity Heatmap (Error)', fontweight='bold')
    
    def _plot_training_summary(self, ax):
        """Plot training summary statistics."""
        try:
            # Get summary statistics
            extraneous_analysis = self.dataset_analyzer.analyze_extraneous_dataset()
            unsolvable_analysis = self.dataset_analyzer.analyze_unsolvable_dataset()
            
            # Calculate totals
            total_extraneous = extraneous_analysis.get('combined', {}).get('total_records', 0)
            total_unsolvable = unsolvable_analysis.get('combined', {}).get('total_records', 0)
            total_records = total_extraneous + total_unsolvable
            
            # Get RL model info
            model_files = self.rl_visualizer.find_model_files()
            num_models = len(model_files)
            
            # Create summary text
            summary_text = f"""
            TRAINING SUMMARY
            ================
            
            Total Training Records: {total_records:,}
            ├─ Extraneous Dataset: {total_extraneous:,}
            └─ Unsolvable Dataset: {total_unsolvable:,}
            
            RL Models Available: {num_models}
            
            Dataset Categories:
            ├─ Kinematics: Physics of motion
            ├─ Newton: Force and acceleration
            ├─ Energy: Work, power, and energy
            ├─ Momentum: Collisions and conservation
            └─ Circuits: Electrical circuits
            
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Training Summary', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error generating summary:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Summary (Error)', fontweight='bold')
    
    def create_focused_dashboard(self, focus: str, save_path: str = None):
        """Create a focused dashboard for specific analysis."""
        if focus == 'datasets':
            self._create_dataset_focused_dashboard(save_path)
        elif focus == 'rl':
            self._create_rl_focused_dashboard(save_path)
        elif focus == 'questions':
            self._create_question_focused_dashboard(save_path)
        else:
            print(f"Unknown focus: {focus}. Available options: datasets, rl, questions")
    
    def _create_dataset_focused_dashboard(self, save_path: str = None):
        """Create a dataset-focused dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Dataset overview
        self._plot_dataset_overview(axes[0, 0])
        
        # Category comparison
        try:
            extraneous_analysis = self.dataset_analyzer.analyze_extraneous_dataset()
            unsolvable_analysis = self.dataset_analyzer.analyze_unsolvable_dataset()
            
            all_categories = set()
            if extraneous_analysis.get('combined', {}).get('categories'):
                all_categories.update(extraneous_analysis['combined']['categories'].keys())
            if unsolvable_analysis.get('combined', {}).get('categories'):
                all_categories.update(unsolvable_analysis['combined']['categories'].keys())
            
            all_categories = sorted(list(all_categories))
            extraneous_counts = [extraneous_analysis.get('combined', {}).get('categories', {}).get(cat, 0) for cat in all_categories]
            unsolvable_counts = [unsolvable_analysis.get('combined', {}).get('categories', {}).get(cat, 0) for cat in all_categories]
            
            x = np.arange(len(all_categories))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, extraneous_counts, width, label='Extraneous', alpha=0.7, color='skyblue')
            axes[0, 1].bar(x + width/2, unsolvable_counts, width, label='Unsolvable', alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Category Distribution Comparison')
            axes[0, 1].set_ylabel('Number of Problems')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(all_categories, rotation=45)
            axes[0, 1].legend()
            
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Difficulty distribution
        self._plot_difficulty_distribution(axes[1, 0])
        
        # Concept usage
        self._plot_concept_usage(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dataset-focused dashboard saved to {save_path}")
        
        plt.show()
    
    def _create_rl_focused_dashboard(self, save_path: str = None):
        """Create an RL-focused dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL Training Dashboard', fontsize=16, fontweight='bold')
        
        # Training progress
        self._plot_rl_progress(axes[0, 0])
        
        # Model comparison
        self._plot_model_comparison(axes[0, 1])
        
        # Learning curves
        try:
            model_files = self.rl_visualizer.find_model_files()
            if model_files:
                model_data = self.rl_visualizer.load_training_data(model_files[-1])
                training_accuracy = getattr(model_data, 'training_accuracy', [])
                training_rewards = getattr(model_data, 'training_rewards', [])
                
                if training_accuracy:
                    episodes = range(0, len(training_accuracy) * 10, 10)
                    axes[1, 0].plot(episodes, training_accuracy, 'b-', linewidth=2, alpha=0.8, label='Accuracy')
                    
                    # Add moving average
                    if len(training_accuracy) > 10:
                        window_size = min(50, len(training_accuracy) // 10)
                        moving_avg = pd.Series(training_accuracy).rolling(window=window_size).mean()
                        axes[1, 0].plot(episodes, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Avg')
                    
                    axes[1, 0].set_title('Learning Curves')
                    axes[1, 0].set_xlabel('Episode')
                    axes[1, 0].set_ylabel('Accuracy')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend()
                
                if training_rewards:
                    episodes = range(0, len(training_rewards) * 10, 10)
                    axes[1, 1].plot(episodes, training_rewards, 'g-', linewidth=2, alpha=0.8, label='Rewards')
                    axes[1, 1].set_title('Reward Learning Curve')
                    axes[1, 1].set_xlabel('Episode')
                    axes[1, 1].set_ylabel('Episode Reward')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].legend()
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RL-focused dashboard saved to {save_path}")
        
        plt.show()
    
    def _create_question_focused_dashboard(self, save_path: str = None):
        """Create a question-focused dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Question Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Difficulty distribution
        self._plot_difficulty_distribution(axes[0, 0])
        
        # Concept usage
        self._plot_concept_usage(axes[0, 1])
        
        # Complexity heatmap
        self._plot_complexity_heatmap(axes[1, 0])
        
        # Text length analysis
        try:
            datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
            word_counts_by_dataset = {}
            
            for dataset in datasets:
                analysis = self.question_analyzer.analyze_dataset_questions(dataset)
                if analysis and 'word_count_stats' in analysis:
                    word_counts_by_dataset[dataset] = analysis['word_count_stats']['mean']
            
            if word_counts_by_dataset:
                datasets_clean = [d.replace('_', '\n').title() for d in word_counts_by_dataset.keys()]
                counts = list(word_counts_by_dataset.values())
                
                bars = axes[1, 1].bar(datasets_clean, counts, alpha=0.8, color='lightblue')
                axes[1, 1].set_title('Average Word Count by Dataset')
                axes[1, 1].set_ylabel('Average Word Count')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                   f'{count:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'No word count data available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Word Count Analysis (No Data)')
                
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Word Count Analysis (Error)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Question-focused dashboard saved to {save_path}")
        
        plt.show()
    
    def generate_dashboard_report(self, output_file: str = None) -> str:
        """Generate a comprehensive dashboard report."""
        report = []
        report.append("=" * 80)
        report.append("AI PHYSICIST TRAINING DASHBOARD REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset analysis
        try:
            extraneous_analysis = self.dataset_analyzer.analyze_extraneous_dataset()
            unsolvable_analysis = self.dataset_analyzer.analyze_unsolvable_dataset()
            
            report.append("DATASET OVERVIEW")
            report.append("-" * 50)
            report.append(f"Extraneous Dataset Records: {extraneous_analysis.get('combined', {}).get('total_records', 0)}")
            report.append(f"Unsolvable Dataset Records: {unsolvable_analysis.get('combined', {}).get('total_records', 0)}")
            report.append(f"Total Training Records: {extraneous_analysis.get('combined', {}).get('total_records', 0) + unsolvable_analysis.get('combined', {}).get('total_records', 0)}")
            report.append("")
        except Exception as e:
            report.append(f"Error analyzing datasets: {str(e)}")
            report.append("")
        
        # RL model analysis
        try:
            model_files = self.rl_visualizer.find_model_files()
            report.append("RL MODEL STATUS")
            report.append("-" * 50)
            report.append(f"Available Models: {len(model_files)}")
            if model_files:
                report.append(f"Latest Model: {model_files[-1]}")
                
                # Get performance data
                model_data = self.rl_visualizer.load_training_data(model_files[-1])
                training_accuracy = getattr(model_data, 'training_accuracy', [])
                if training_accuracy:
                    report.append(f"Final Accuracy: {training_accuracy[-1]:.4f}")
                    report.append(f"Best Accuracy: {max(training_accuracy):.4f}")
                    report.append(f"Training Episodes: {len(training_accuracy) * 10}")
            report.append("")
        except Exception as e:
            report.append(f"Error analyzing RL models: {str(e)}")
            report.append("")
        
        # Question analysis
        try:
            datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
            total_questions = 0
            
            for dataset in datasets:
                analysis = self.question_analyzer.analyze_dataset_questions(dataset)
                if analysis:
                    total_questions += analysis.get('total_questions', 0)
            
            report.append("QUESTION ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total Questions Analyzed: {total_questions}")
            report.append("")
        except Exception as e:
            report.append(f"Error analyzing questions: {str(e)}")
            report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 50)
        report.append("The AI Physicist training system includes:")
        report.append("• Multiple physics question datasets with different characteristics")
        report.append("• Reinforcement learning models for question answering")
        report.append("• Comprehensive analysis tools for performance monitoring")
        report.append("• Visualization tools for understanding training progress")
        report.append("")
        report.append("Use the dashboard tools to monitor training progress and identify areas for improvement.")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Dashboard report saved to {output_file}")
        
        return report_text


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create performance dashboards for AI Physicist training')
    parser.add_argument('--data-dir', type=str, help='Base directory containing all data files')
    parser.add_argument('--output-dir', type=str, default='./viz_output', help='Output directory for plots and reports')
    parser.add_argument('--dashboard', choices=['comprehensive', 'datasets', 'rl', 'questions'], 
                       default='comprehensive', help='Type of dashboard to create')
    parser.add_argument('--save', action='store_true', help='Save dashboard plots')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(args.data_dir)
    
    # Create dashboard
    if args.dashboard == 'comprehensive':
        save_path = os.path.join(args.output_dir, 'comprehensive_dashboard.png') if args.save else None
        dashboard.create_comprehensive_dashboard(save_path)
    else:
        save_path = os.path.join(args.output_dir, f'{args.dashboard}_dashboard.png') if args.save else None
        dashboard.create_focused_dashboard(args.dashboard, save_path)
    
    # Generate report
    report = dashboard.generate_dashboard_report(os.path.join(args.output_dir, 'dashboard_report.txt'))
    print(report)


if __name__ == "__main__":
    main()
