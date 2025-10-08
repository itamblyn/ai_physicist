#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Tools

This module provides comprehensive analysis and visualization tools for
the various physics training datasets in the AI Physicist project.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from pathlib import Path


class DatasetAnalyzer:
    """Analyzes and visualizes physics training datasets."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the dataset analyzer.
        
        Args:
            data_dir: Base directory containing dataset files. If None, uses project structure.
        """
        if data_dir is None:
            # Use project structure
            self.base_dir = Path(__file__).parent.parent
            self.extraneous_dir = self.base_dir / "03_extraneous_info_dataset" / "samples"
            self.unsolvable_dir = self.base_dir / "04_unsolvable" / "samples"
            self.baseline_dir = self.base_dir / "02_baseline"
        else:
            self.base_dir = Path(data_dir)
            self.extraneous_dir = self.base_dir / "03_extraneous_info_dataset" / "samples"
            self.unsolvable_dir = self.base_dir / "04_unsolvable" / "samples"
            self.baseline_dir = self.base_dir / "02_baseline"
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load a JSONL file and return list of records."""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing JSON in {file_path}: {e}")
        return records
    
    def analyze_extraneous_dataset(self) -> Dict[str, Any]:
        """Analyze the extraneous information dataset."""
        print("Analyzing extraneous information dataset...")
        
        # Load datasets
        supervised_data = self.load_jsonl(str(self.extraneous_dir / "supervised.jsonl"))
        preference_data = self.load_jsonl(str(self.extraneous_dir / "preference.jsonl"))
        
        analysis = {
            'supervised': self._analyze_supervised_data(supervised_data),
            'preference': self._analyze_preference_data(preference_data),
            'combined': self._analyze_combined_extraneous(supervised_data, preference_data)
        }
        
        return analysis
    
    def analyze_unsolvable_dataset(self) -> Dict[str, Any]:
        """Analyze the unsolvable/inconsistent dataset."""
        print("Analyzing unsolvable/inconsistent dataset...")
        
        # Load datasets
        unsolvable_data = self.load_jsonl(str(self.unsolvable_dir / "unsolvable.jsonl"))
        solvability_data = self.load_jsonl(str(self.unsolvable_dir / "solvability.jsonl"))
        
        analysis = {
            'unsolvable': self._analyze_unsolvable_data(unsolvable_data),
            'solvability': self._analyze_solvability_data(solvability_data),
            'combined': self._analyze_combined_unsolvable(unsolvable_data, solvability_data)
        }
        
        return analysis
    
    def _analyze_supervised_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze supervised learning data."""
        if not data:
            return {}
        
        categories = [record['category'] for record in data]
        category_counts = Counter(categories)
        
        # Analyze extraneous facts
        extraneous_counts = []
        important_counts = []
        
        for record in data:
            extraneous_counts.append(len(record.get('extraneous_facts', [])))
            important_counts.append(len(record.get('important_facts', [])))
        
        # Analyze solution steps
        solution_lengths = [len(record.get('solution_steps', [])) for record in data]
        
        return {
            'total_records': len(data),
            'categories': dict(category_counts),
            'extraneous_facts': {
                'mean': np.mean(extraneous_counts),
                'std': np.std(extraneous_counts),
                'min': np.min(extraneous_counts),
                'max': np.max(extraneous_counts)
            },
            'important_facts': {
                'mean': np.mean(important_counts),
                'std': np.std(important_counts),
                'min': np.min(important_counts),
                'max': np.max(important_counts)
            },
            'solution_steps': {
                'mean': np.mean(solution_lengths),
                'std': np.std(solution_lengths),
                'min': np.min(solution_lengths),
                'max': np.max(solution_lengths)
            }
        }
    
    def _analyze_preference_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze preference learning data."""
        if not data:
            return {}
        
        categories = [record['category'] for record in data]
        category_counts = Counter(categories)
        
        # Analyze chosen vs rejected
        chosen_correct = sum(1 for record in data if record['chosen']['correct'])
        rejected_correct = sum(1 for record in data if record['rejected']['correct'])
        
        chosen_uses_extraneous = sum(1 for record in data if record['chosen']['uses_extraneous'])
        rejected_uses_extraneous = sum(1 for record in data if record['rejected']['uses_extraneous'])
        
        return {
            'total_records': len(data),
            'categories': dict(category_counts),
            'chosen': {
                'correct_count': chosen_correct,
                'correct_rate': chosen_correct / len(data),
                'uses_extraneous_count': chosen_uses_extraneous,
                'uses_extraneous_rate': chosen_uses_extraneous / len(data)
            },
            'rejected': {
                'correct_count': rejected_correct,
                'correct_rate': rejected_correct / len(data),
                'uses_extraneous_count': rejected_uses_extraneous,
                'uses_extraneous_rate': rejected_uses_extraneous / len(data)
            }
        }
    
    def _analyze_unsolvable_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze unsolvable/inconsistent data."""
        if not data:
            return {}
        
        categories = [record['category'] for record in data]
        category_counts = Counter(categories)
        
        # Analyze inconsistency types
        inconsistency_types = []
        for record in data:
            for inconsistency in record.get('inconsistencies', []):
                inconsistency_types.append(inconsistency['type'])
        
        inconsistency_type_counts = Counter(inconsistency_types)
        
        # Analyze inconsistency counts per problem
        inconsistency_counts = [len(record.get('inconsistencies', [])) for record in data]
        
        return {
            'total_records': len(data),
            'categories': dict(category_counts),
            'inconsistency_types': dict(inconsistency_type_counts),
            'inconsistencies_per_problem': {
                'mean': np.mean(inconsistency_counts),
                'std': np.std(inconsistency_counts),
                'min': np.min(inconsistency_counts),
                'max': np.max(inconsistency_counts)
            }
        }
    
    def _analyze_solvability_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze solvability classification data."""
        if not data:
            return {}
        
        categories = [record['category'] for record in data]
        category_counts = Counter(categories)
        
        solvable_count = sum(1 for record in data if record['solvable'])
        unsolvable_count = len(data) - solvable_count
        
        # Analyze by category
        category_solvability = defaultdict(lambda: {'solvable': 0, 'unsolvable': 0})
        for record in data:
            if record['solvable']:
                category_solvability[record['category']]['solvable'] += 1
            else:
                category_solvability[record['category']]['unsolvable'] += 1
        
        return {
            'total_records': len(data),
            'categories': dict(category_counts),
            'overall_solvability': {
                'solvable_count': solvable_count,
                'unsolvable_count': unsolvable_count,
                'solvable_rate': solvable_count / len(data)
            },
            'category_solvability': dict(category_solvability)
        }
    
    def _analyze_combined_extraneous(self, supervised_data: List[Dict], preference_data: List[Dict]) -> Dict[str, Any]:
        """Analyze combined extraneous dataset."""
        all_categories = []
        all_categories.extend([record['category'] for record in supervised_data])
        all_categories.extend([record['category'] for record in preference_data])
        
        return {
            'total_records': len(supervised_data) + len(preference_data),
            'categories': dict(Counter(all_categories)),
            'supervised_ratio': len(supervised_data) / (len(supervised_data) + len(preference_data))
        }
    
    def _analyze_combined_unsolvable(self, unsolvable_data: List[Dict], solvability_data: List[Dict]) -> Dict[str, Any]:
        """Analyze combined unsolvable dataset."""
        all_categories = []
        all_categories.extend([record['category'] for record in unsolvable_data])
        all_categories.extend([record['category'] for record in solvability_data])
        
        return {
            'total_records': len(unsolvable_data) + len(solvability_data),
            'categories': dict(Counter(all_categories)),
            'unsolvable_ratio': len(unsolvable_data) / (len(unsolvable_data) + len(solvability_data))
        }
    
    def plot_dataset_overview(self, save_path: str = None):
        """Create an overview plot of all datasets."""
        extraneous_analysis = self.analyze_extraneous_dataset()
        unsolvable_analysis = self.analyze_unsolvable_dataset()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI Physicist Training Dataset Overview', fontsize=16, fontweight='bold')
        
        # Extraneous dataset - category distribution
        if extraneous_analysis.get('combined', {}).get('categories'):
            categories = list(extraneous_analysis['combined']['categories'].keys())
            counts = list(extraneous_analysis['combined']['categories'].values())
            axes[0, 0].bar(categories, counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Extraneous Dataset - Categories')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Extraneous dataset - preference analysis
        if extraneous_analysis.get('preference', {}).get('chosen'):
            chosen_data = extraneous_analysis['preference']['chosen']
            rejected_data = extraneous_analysis['preference']['rejected']
            
            metrics = ['correct_rate', 'uses_extraneous_rate']
            chosen_values = [chosen_data.get(m, 0) for m in metrics]
            rejected_values = [rejected_data.get(m, 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, chosen_values, width, label='Chosen', alpha=0.7)
            axes[0, 1].bar(x + width/2, rejected_values, width, label='Rejected', alpha=0.7)
            axes[0, 1].set_title('Preference Dataset - Chosen vs Rejected')
            axes[0, 1].set_ylabel('Rate')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(['Correct', 'Uses Extraneous'])
            axes[0, 1].legend()
        
        # Extraneous dataset - facts distribution
        if extraneous_analysis.get('supervised', {}).get('extraneous_facts'):
            extraneous_stats = extraneous_analysis['supervised']['extraneous_facts']
            important_stats = extraneous_analysis['supervised']['important_facts']
            
            stats = ['mean', 'std', 'min', 'max']
            extraneous_values = [extraneous_stats.get(s, 0) for s in stats]
            important_values = [important_stats.get(s, 0) for s in stats]
            
            x = np.arange(len(stats))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, extraneous_values, width, label='Extraneous', alpha=0.7)
            axes[0, 2].bar(x + width/2, important_values, width, label='Important', alpha=0.7)
            axes[0, 2].set_title('Facts Distribution')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(stats)
            axes[0, 2].legend()
        
        # Unsolvable dataset - category distribution
        if unsolvable_analysis.get('combined', {}).get('categories'):
            categories = list(unsolvable_analysis['combined']['categories'].keys())
            counts = list(unsolvable_analysis['combined']['categories'].values())
            axes[1, 0].bar(categories, counts, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Unsolvable Dataset - Categories')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Unsolvable dataset - inconsistency types
        if unsolvable_analysis.get('unsolvable', {}).get('inconsistency_types'):
            inconsistency_types = list(unsolvable_analysis['unsolvable']['inconsistency_types'].keys())
            counts = list(unsolvable_analysis['unsolvable']['inconsistency_types'].values())
            axes[1, 1].pie(counts, labels=inconsistency_types, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Inconsistency Types')
        
        # Solvability dataset - solvability rates
        if unsolvable_analysis.get('solvability', {}).get('category_solvability'):
            category_data = unsolvable_analysis['solvability']['category_solvability']
            categories = list(category_data.keys())
            solvable_counts = [category_data[cat]['solvable'] for cat in categories]
            unsolvable_counts = [category_data[cat]['unsolvable'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[1, 2].bar(x - width/2, solvable_counts, width, label='Solvable', alpha=0.7)
            axes[1, 2].bar(x + width/2, unsolvable_counts, width, label='Unsolvable', alpha=0.7)
            axes[1, 2].set_title('Solvability by Category')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(categories, rotation=45)
            axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overview plot saved to {save_path}")
        
        plt.show()
    
    def plot_category_comparison(self, save_path: str = None):
        """Compare categories across all datasets."""
        extraneous_analysis = self.analyze_extraneous_dataset()
        unsolvable_analysis = self.analyze_unsolvable_dataset()
        
        # Get all unique categories
        all_categories = set()
        if extraneous_analysis.get('combined', {}).get('categories'):
            all_categories.update(extraneous_analysis['combined']['categories'].keys())
        if unsolvable_analysis.get('combined', {}).get('categories'):
            all_categories.update(unsolvable_analysis['combined']['categories'].keys())
        
        all_categories = sorted(list(all_categories))
        
        # Prepare data
        extraneous_counts = [extraneous_analysis.get('combined', {}).get('categories', {}).get(cat, 0) for cat in all_categories]
        unsolvable_counts = [unsolvable_analysis.get('combined', {}).get('categories', {}).get(cat, 0) for cat in all_categories]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        ax.bar(x - width/2, extraneous_counts, width, label='Extraneous Dataset', alpha=0.7, color='skyblue')
        ax.bar(x + width/2, unsolvable_counts, width, label='Unsolvable Dataset', alpha=0.7, color='lightcoral')
        
        ax.set_xlabel('Physics Category')
        ax.set_ylabel('Number of Problems')
        ax.set_title('Problem Distribution by Category Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Category comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive text report of dataset analysis."""
        extraneous_analysis = self.analyze_extraneous_dataset()
        unsolvable_analysis = self.analyze_unsolvable_dataset()
        
        report = []
        report.append("=" * 60)
        report.append("AI PHYSICIST TRAINING DATASET ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Extraneous dataset report
        report.append("EXTRANEOUS INFORMATION DATASET")
        report.append("-" * 40)
        
        if extraneous_analysis.get('supervised'):
            supervised = extraneous_analysis['supervised']
            report.append(f"Supervised Learning Data:")
            report.append(f"  Total records: {supervised.get('total_records', 0)}")
            report.append(f"  Categories: {supervised.get('categories', {})}")
            report.append(f"  Extraneous facts per problem: {supervised.get('extraneous_facts', {})}")
            report.append(f"  Important facts per problem: {supervised.get('important_facts', {})}")
            report.append(f"  Solution steps per problem: {supervised.get('solution_steps', {})}")
            report.append("")
        
        if extraneous_analysis.get('preference'):
            preference = extraneous_analysis['preference']
            report.append(f"Preference Learning Data:")
            report.append(f"  Total records: {preference.get('total_records', 0)}")
            report.append(f"  Categories: {preference.get('categories', {})}")
            report.append(f"  Chosen responses: {preference.get('chosen', {})}")
            report.append(f"  Rejected responses: {preference.get('rejected', {})}")
            report.append("")
        
        # Unsolvable dataset report
        report.append("UNSOLVABLE/INCONSISTENT DATASET")
        report.append("-" * 40)
        
        if unsolvable_analysis.get('unsolvable'):
            unsolvable = unsolvable_analysis['unsolvable']
            report.append(f"Inconsistent Problems:")
            report.append(f"  Total records: {unsolvable.get('total_records', 0)}")
            report.append(f"  Categories: {unsolvable.get('categories', {})}")
            report.append(f"  Inconsistency types: {unsolvable.get('inconsistency_types', {})}")
            report.append(f"  Inconsistencies per problem: {unsolvable.get('inconsistencies_per_problem', {})}")
            report.append("")
        
        if unsolvable_analysis.get('solvability'):
            solvability = unsolvable_analysis['solvability']
            report.append(f"Solvability Classification:")
            report.append(f"  Total records: {solvability.get('total_records', 0)}")
            report.append(f"  Categories: {solvability.get('categories', {})}")
            report.append(f"  Overall solvability: {solvability.get('overall_solvability', {})}")
            report.append(f"  Category solvability: {solvability.get('category_solvability', {})}")
            report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        total_extraneous = extraneous_analysis.get('combined', {}).get('total_records', 0)
        total_unsolvable = unsolvable_analysis.get('combined', {}).get('total_records', 0)
        report.append(f"Total extraneous dataset records: {total_extraneous}")
        report.append(f"Total unsolvable dataset records: {total_unsolvable}")
        report.append(f"Total training records: {total_extraneous + total_unsolvable}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze AI Physicist training datasets')
    parser.add_argument('--data-dir', type=str, help='Base directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='./viz_output', help='Output directory for plots and reports')
    parser.add_argument('--report-only', action='store_true', help='Generate only text report, no plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.data_dir)
    
    if not args.report_only:
        # Generate plots
        print("Generating dataset overview...")
        analyzer.plot_dataset_overview(os.path.join(args.output_dir, 'dataset_overview.png'))
        
        print("Generating category comparison...")
        analyzer.plot_category_comparison(os.path.join(args.output_dir, 'category_comparison.png'))
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_report(os.path.join(args.output_dir, 'dataset_analysis_report.txt'))
    print(report)


if __name__ == "__main__":
    main()
