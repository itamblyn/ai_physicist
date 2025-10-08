#!/usr/bin/env python3
"""
Reinforcement Learning Training Visualization Tools

This module provides visualization tools for analyzing RL training progress,
model performance, and learning dynamics in the AI Physicist project.
"""

import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
from collections import defaultdict


class RLVisualizer:
    """Visualizes RL training progress and model performance."""
    
    def __init__(self, baseline_dir: str = None):
        """
        Initialize the RL visualizer.
        
        Args:
            baseline_dir: Directory containing RL models and training data. If None, uses project structure.
        """
        if baseline_dir is None:
            self.baseline_dir = Path(__file__).parent.parent / "02_baseline"
        else:
            self.baseline_dir = Path(baseline_dir)
    
    def load_training_data(self, model_file: str) -> Dict[str, Any]:
        """Load training data from a saved model file."""
        model_path = self.baseline_dir / model_file
        
        if not model_path.exists():
            print(f"Warning: Model file {model_path} not found")
            return {}
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return {}
    
    def find_model_files(self) -> List[str]:
        """Find all available model files."""
        model_files = []
        for file_path in self.baseline_dir.glob("*.pkl"):
            if "model" in file_path.name:
                model_files.append(file_path.name)
        return sorted(model_files)
    
    def plot_training_progress(self, model_file: str = None, save_path: str = None, show_plot: bool = False):
        """Plot training progress for a specific model."""
        if model_file is None:
            model_files = self.find_model_files()
            if not model_files:
                print("No model files found")
                return
            model_file = model_files[-1]  # Use the last (most recent) model
        
        model_data = self.load_training_data(model_file)
        if not model_data:
            return
        
        # Extract training data
        training_accuracy = getattr(model_data, 'training_accuracy', [])
        training_rewards = getattr(model_data, 'training_rewards', [])
        
        if not training_accuracy and not training_rewards:
            print(f"No training data found in {model_file}")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'RL Training Progress - {model_file}', fontsize=16, fontweight='bold')
        
        # Plot accuracy over time
        if training_accuracy:
            episodes = range(0, len(training_accuracy) * 10, 10)
            axes[0, 0].plot(episodes, training_accuracy, 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].set_title('Training Accuracy Over Time')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average
            if len(training_accuracy) > 10:
                window_size = min(50, len(training_accuracy) // 10)
                moving_avg = pd.Series(training_accuracy).rolling(window=window_size).mean()
                axes[0, 0].plot(episodes, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Avg (window={window_size})')
                axes[0, 0].legend()
        
        # Plot rewards over time
        if training_rewards:
            episodes = range(0, len(training_rewards) * 10, 10)
            axes[0, 1].plot(episodes, training_rewards, 'g-', linewidth=2, alpha=0.8)
            axes[0, 1].set_title('Training Rewards Over Time')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Episode Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add moving average
            if len(training_rewards) > 10:
                window_size = min(50, len(training_rewards) // 10)
                moving_avg = pd.Series(training_rewards).rolling(window=window_size).mean()
                axes[0, 1].plot(episodes, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Avg (window={window_size})')
                axes[0, 1].legend()
        
        # Plot accuracy distribution
        if training_accuracy:
            axes[1, 0].hist(training_accuracy, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('Accuracy Distribution')
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(training_accuracy), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(training_accuracy):.3f}')
            axes[1, 0].legend()
        
        # Plot reward distribution
        if training_rewards:
            axes[1, 1].hist(training_rewards, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Episode Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(np.mean(training_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(training_rewards):.3f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def compare_models(self, model_files: List[str] = None, save_path: str = None, show_plot: bool = False):
        """Compare training progress across multiple models."""
        if model_files is None:
            model_files = self.find_model_files()
        
        if len(model_files) < 2:
            print("Need at least 2 models to compare")
            return
        
        # Load data from all models
        model_data = {}
        for model_file in model_files:
            data = self.load_training_data(model_file)
            if data:
                model_data[model_file] = data
        
        if len(model_data) < 2:
            print("Not enough valid models to compare")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Model Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
        
        # Plot accuracy comparison
        for i, (model_name, data) in enumerate(model_data.items()):
            training_accuracy = getattr(data, 'training_accuracy', [])
            if training_accuracy:
                episodes = range(0, len(training_accuracy) * 10, 10)
                axes[0, 0].plot(episodes, training_accuracy, color=colors[i], linewidth=2, alpha=0.8, label=model_name)
        
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot reward comparison
        for i, (model_name, data) in enumerate(model_data.items()):
            training_rewards = getattr(data, 'training_rewards', [])
            if training_rewards:
                episodes = range(0, len(training_rewards) * 10, 10)
                axes[0, 1].plot(episodes, training_rewards, color=colors[i], linewidth=2, alpha=0.8, label=model_name)
        
        axes[0, 1].set_title('Reward Comparison')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot final performance comparison
        final_accuracies = []
        final_rewards = []
        model_names = []
        
        for model_name, data in model_data.items():
            training_accuracy = getattr(data, 'training_accuracy', [])
            training_rewards = getattr(data, 'training_rewards', [])
            
            if training_accuracy:
                final_accuracies.append(training_accuracy[-1])
                model_names.append(model_name)
            
            if training_rewards:
                final_rewards.append(training_rewards[-1])
        
        if final_accuracies:
            axes[1, 0].bar(model_names, final_accuracies, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Final Accuracy Comparison')
            axes[1, 0].set_ylabel('Final Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        if final_rewards:
            axes[1, 1].bar(model_names, final_rewards, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Final Reward Comparison')
            axes[1, 1].set_ylabel('Final Reward')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def plot_learning_curves(self, model_file: str = None, window_size: int = 50, save_path: str = None, show_plot: bool = False):
        """Plot smoothed learning curves with confidence intervals."""
        if model_file is None:
            model_files = self.find_model_files()
            if not model_files:
                print("No model files found")
                return
            model_file = model_files[-1]
        
        model_data = self.load_training_data(model_file)
        if not model_data:
            return
        
        training_accuracy = getattr(model_data, 'training_accuracy', [])
        training_rewards = getattr(model_data, 'training_rewards', [])
        
        if not training_accuracy and not training_rewards:
            print(f"No training data found in {model_file}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Learning Curves - {model_file}', fontsize=16, fontweight='bold')
        
        # Plot accuracy learning curve
        if training_accuracy:
            episodes = np.arange(len(training_accuracy)) * 10
            
            # Calculate moving average and confidence interval
            df = pd.DataFrame({'accuracy': training_accuracy, 'episode': episodes})
            rolling_mean = df['accuracy'].rolling(window=window_size, center=True).mean()
            rolling_std = df['accuracy'].rolling(window=window_size, center=True).std()
            
            axes[0].plot(episodes, training_accuracy, alpha=0.3, color='blue', label='Raw Data')
            axes[0].plot(episodes, rolling_mean, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
            
            # Add confidence interval
            upper_bound = rolling_mean + 1.96 * rolling_std
            lower_bound = rolling_mean - 1.96 * rolling_std
            axes[0].fill_between(episodes, lower_bound, upper_bound, alpha=0.2, color='red', label='95% Confidence Interval')
            
            axes[0].set_title('Accuracy Learning Curve')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Accuracy')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Plot reward learning curve
        if training_rewards:
            episodes = np.arange(len(training_rewards)) * 10
            
            # Calculate moving average and confidence interval
            df = pd.DataFrame({'reward': training_rewards, 'episode': episodes})
            rolling_mean = df['reward'].rolling(window=window_size, center=True).mean()
            rolling_std = df['reward'].rolling(window=window_size, center=True).std()
            
            axes[1].plot(episodes, training_rewards, alpha=0.3, color='green', label='Raw Data')
            axes[1].plot(episodes, rolling_mean, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
            
            # Add confidence interval
            upper_bound = rolling_mean + 1.96 * rolling_std
            lower_bound = rolling_mean - 1.96 * rolling_std
            axes[1].fill_between(episodes, lower_bound, upper_bound, alpha=0.2, color='red', label='95% Confidence Interval')
            
            axes[1].set_title('Reward Learning Curve')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Episode Reward')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def analyze_q_table(self, model_file: str = None, save_path: str = None, show_plot: bool = False):
        """Analyze and visualize the Q-table from a trained model."""
        if model_file is None:
            model_files = self.find_model_files()
            if not model_files:
                print("No model files found")
                return
            model_file = model_files[-1]
        
        model_data = self.load_training_data(model_file)
        if not model_data:
            return
        
        q_table = getattr(model_data, 'q_table', {})
        if not q_table:
            print(f"No Q-table found in {model_file}")
            return
        
        # Convert Q-table to DataFrame for analysis
        q_data = []
        for state, actions in q_table.items():
            for action, value in actions.items():
                q_data.append({
                    'state': state,
                    'action': action,
                    'q_value': value
                })
        
        if not q_data:
            print("Q-table is empty")
            return
        
        df = pd.DataFrame(q_data)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Q-Table Analysis - {model_file}', fontsize=16, fontweight='bold')
        
        # Q-value distribution
        axes[0, 0].hist(df['q_value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Q-Value Distribution')
        axes[0, 0].set_xlabel('Q-Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['q_value'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["q_value"].mean():.3f}')
        axes[0, 0].legend()
        
        # Q-values by state (top 20 states by average Q-value)
        state_avg_q = df.groupby('state')['q_value'].mean().sort_values(ascending=False)
        top_states = state_avg_q.head(20)
        
        axes[0, 1].bar(range(len(top_states)), top_states.values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Top 20 States by Average Q-Value')
        axes[0, 1].set_xlabel('State Rank')
        axes[0, 1].set_ylabel('Average Q-Value')
        axes[0, 1].set_xticks(range(0, len(top_states), 2))
        axes[0, 1].set_xticklabels([f'State {i+1}' for i in range(0, len(top_states), 2)], rotation=45)
        
        # Q-values by action
        action_avg_q = df.groupby('action')['q_value'].mean().sort_values(ascending=False)
        
        axes[1, 0].bar(range(len(action_avg_q)), action_avg_q.values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Average Q-Value by Action')
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('Average Q-Value')
        axes[1, 0].set_xticks(range(len(action_avg_q)))
        axes[1, 0].set_xticklabels(action_avg_q.index, rotation=45)
        
        # Q-value heatmap (sample of states and actions)
        if len(df) > 0:
            # Sample states and actions for heatmap
            sample_states = df['state'].value_counts().head(10).index
            sample_actions = df['action'].value_counts().head(10).index
            
            heatmap_data = df[df['state'].isin(sample_states) & df['action'].isin(sample_actions)]
            if not heatmap_data.empty:
                pivot_table = heatmap_data.pivot_table(values='q_value', index='state', columns='action', fill_value=0)
                
                sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 1])
                axes[1, 1].set_title('Q-Value Heatmap (Sample)')
                axes[1, 1].set_xlabel('Action')
                axes[1, 1].set_ylabel('State')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Q-table analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def generate_performance_report(self, model_file: str = None, output_file: str = None) -> str:
        """Generate a comprehensive performance report for a model."""
        if model_file is None:
            model_files = self.find_model_files()
            if not model_files:
                return "No model files found"
            model_file = model_files[-1]
        
        model_data = self.load_training_data(model_file)
        if not model_data:
            return f"Could not load model {model_file}"
        
        training_accuracy = getattr(model_data, 'training_accuracy', [])
        training_rewards = getattr(model_data, 'training_rewards', [])
        q_table = getattr(model_data, 'q_table', {})
        
        report = []
        report.append("=" * 60)
        report.append(f"RL MODEL PERFORMANCE REPORT - {model_file}")
        report.append("=" * 60)
        report.append("")
        
        # Training statistics
        if training_accuracy:
            report.append("TRAINING ACCURACY STATISTICS")
            report.append("-" * 40)
            report.append(f"Total episodes: {len(training_accuracy) * 10}")
            report.append(f"Initial accuracy: {training_accuracy[0]:.4f}")
            report.append(f"Final accuracy: {training_accuracy[-1]:.4f}")
            report.append(f"Best accuracy: {max(training_accuracy):.4f}")
            report.append(f"Mean accuracy: {np.mean(training_accuracy):.4f}")
            report.append(f"Std accuracy: {np.std(training_accuracy):.4f}")
            report.append("")
        
        if training_rewards:
            report.append("TRAINING REWARD STATISTICS")
            report.append("-" * 40)
            report.append(f"Total episodes: {len(training_rewards) * 10}")
            report.append(f"Initial reward: {training_rewards[0]:.4f}")
            report.append(f"Final reward: {training_rewards[-1]:.4f}")
            report.append(f"Best reward: {max(training_rewards):.4f}")
            report.append(f"Mean reward: {np.mean(training_rewards):.4f}")
            report.append(f"Std reward: {np.std(training_rewards):.4f}")
            report.append("")
        
        # Q-table statistics
        if q_table:
            report.append("Q-TABLE STATISTICS")
            report.append("-" * 40)
            
            # Convert Q-table to DataFrame for analysis
            q_data = []
            for state, actions in q_table.items():
                for action, value in actions.items():
                    q_data.append(value)
            
            if q_data:
                report.append(f"Total state-action pairs: {len(q_data)}")
                report.append(f"Unique states: {len(set(state for state, _ in q_table.items()))}")
                report.append(f"Mean Q-value: {np.mean(q_data):.4f}")
                report.append(f"Std Q-value: {np.std(q_data):.4f}")
                report.append(f"Min Q-value: {np.min(q_data):.4f}")
                report.append(f"Max Q-value: {np.max(q_data):.4f}")
                
                # Learning progress analysis
                if training_accuracy:
                    # Calculate learning rate (improvement per episode)
                    accuracy_improvement = training_accuracy[-1] - training_accuracy[0]
                    episodes = len(training_accuracy) * 10
                    learning_rate = accuracy_improvement / episodes
                    report.append(f"Learning rate (accuracy/episode): {learning_rate:.6f}")
            
            report.append("")
        
        # Model parameters
        report.append("MODEL PARAMETERS")
        report.append("-" * 40)
        for attr in ['learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay', 'epsilon_min']:
            value = getattr(model_data, attr, 'N/A')
            report.append(f"{attr}: {value}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Performance report saved to {output_file}")
        
        return report_text


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RL training progress and model performance')
    parser.add_argument('--baseline-dir', type=str, help='Directory containing RL models')
    parser.add_argument('--output-dir', type=str, default='./viz_output', help='Output directory for plots and reports')
    parser.add_argument('--model', type=str, help='Specific model file to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--q-analysis', action='store_true', help='Analyze Q-table')
    parser.add_argument('--learning-curves', action='store_true', help='Generate learning curves')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RLVisualizer(args.baseline_dir)
    
    if args.compare:
        print("Comparing models...")
        visualizer.compare_models(save_path=os.path.join(args.output_dir, 'model_comparison.png'))
    
    if args.learning_curves:
        print("Generating learning curves...")
        visualizer.plot_learning_curves(args.model, save_path=os.path.join(args.output_dir, 'learning_curves.png'))
    
    if args.q_analysis:
        print("Analyzing Q-table...")
        visualizer.analyze_q_table(args.model, save_path=os.path.join(args.output_dir, 'q_table_analysis.png'))
    
    # Always generate training progress and performance report
    print("Generating training progress...")
    visualizer.plot_training_progress(args.model, save_path=os.path.join(args.output_dir, 'training_progress.png'))
    
    print("Generating performance report...")
    report = visualizer.generate_performance_report(args.model, os.path.join(args.output_dir, 'performance_report.txt'))
    print(report)


if __name__ == "__main__":
    main()
