#!/usr/bin/env python3
"""
Physics Question Analysis and Visualization Tools

This module provides tools for analyzing the content, difficulty, and patterns
in physics questions across different datasets.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class QuestionAnalyzer:
    """Analyzes physics questions for content, difficulty, and patterns."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the question analyzer.
        
        Args:
            data_dir: Base directory containing dataset files. If None, uses project structure.
        """
        if data_dir is None:
            self.base_dir = Path(__file__).parent.parent
            self.extraneous_dir = self.base_dir / "03_extraneous_info_dataset" / "samples"
            self.unsolvable_dir = self.base_dir / "04_unsolvable" / "samples"
        else:
            self.base_dir = Path(data_dir)
            self.extraneous_dir = self.base_dir / "03_extraneous_info_dataset" / "samples"
            self.unsolvable_dir = self.base_dir / "04_unsolvable" / "samples"
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Physics-specific terms
        self.physics_terms = {
            'kinematics': ['velocity', 'acceleration', 'displacement', 'time', 'distance', 'speed'],
            'newton': ['force', 'mass', 'newton', 'friction', 'gravity', 'weight'],
            'energy': ['energy', 'kinetic', 'potential', 'work', 'power', 'joule'],
            'momentum': ['momentum', 'collision', 'elastic', 'inelastic', 'conservation'],
            'circuits': ['voltage', 'current', 'resistance', 'ohm', 'circuit', 'resistor']
        }
    
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
    
    def extract_question_features(self, question_text: str) -> Dict[str, Any]:
        """Extract features from a physics question."""
        features = {}
        
        # Basic text features
        features['length'] = len(question_text)
        features['word_count'] = len(word_tokenize(question_text))
        features['sentence_count'] = len(sent_tokenize(question_text))
        
        # Physics term analysis
        question_lower = question_text.lower()
        physics_term_counts = {}
        for category, terms in self.physics_terms.items():
            count = sum(1 for term in terms if term in question_lower)
            physics_term_counts[category] = count
        
        features['physics_terms'] = physics_term_counts
        features['total_physics_terms'] = sum(physics_term_counts.values())
        
        # Numerical content analysis
        numbers = re.findall(r'-?\d+\.?\d*', question_text)
        features['number_count'] = len(numbers)
        features['has_numbers'] = len(numbers) > 0
        
        # Unit analysis
        units = re.findall(r'\b(m/s|m/s²|kg|N|J|W|V|A|Ω|s|m|°C|°F)\b', question_text)
        features['unit_count'] = len(units)
        features['has_units'] = len(units) > 0
        
        # Question type indicators
        question_words = ['what', 'how', 'find', 'calculate', 'determine', 'compute']
        features['question_word_count'] = sum(1 for word in question_words if word in question_lower)
        
        # Complexity indicators
        features['has_equation'] = any(char in question_text for char in ['=', '+', '-', '*', '/', '^'])
        features['has_condition'] = any(word in question_lower for word in ['if', 'given', 'assuming', 'suppose'])
        
        return features
    
    def analyze_question_difficulty(self, question_data: Dict) -> Dict[str, Any]:
        """Analyze the difficulty of a physics question."""
        prompt = question_data.get('prompt', '')
        features = self.extract_question_features(prompt)
        
        difficulty_score = 0
        
        # Length-based difficulty
        if features['word_count'] > 50:
            difficulty_score += 1
        if features['sentence_count'] > 3:
            difficulty_score += 1
        
        # Physics concept complexity
        if features['total_physics_terms'] > 3:
            difficulty_score += 1
        
        # Numerical complexity
        if features['number_count'] > 5:
            difficulty_score += 1
        if features['unit_count'] > 3:
            difficulty_score += 1
        
        # Problem structure complexity
        if features['has_equation']:
            difficulty_score += 1
        if features['has_condition']:
            difficulty_score += 1
        
        # Category-specific difficulty
        category = question_data.get('category', '')
        if category in ['momentum', 'energy']:
            difficulty_score += 1
        elif category in ['circuits']:
            difficulty_score += 1
        
        # Extraneous information complexity
        if 'extraneous_facts' in question_data:
            extraneous_count = len(question_data['extraneous_facts'])
            if extraneous_count > 2:
                difficulty_score += 1
        
        # Inconsistency complexity
        if 'inconsistencies' in question_data:
            inconsistency_count = len(question_data['inconsistencies'])
            difficulty_score += inconsistency_count
        
        features['difficulty_score'] = difficulty_score
        features['difficulty_level'] = self._categorize_difficulty(difficulty_score)
        
        return features
    
    def _categorize_difficulty(self, score: int) -> str:
        """Categorize difficulty score into levels."""
        if score <= 2:
            return 'Easy'
        elif score <= 4:
            return 'Medium'
        elif score <= 6:
            return 'Hard'
        else:
            return 'Very Hard'
    
    def analyze_dataset_questions(self, dataset_name: str) -> Dict[str, Any]:
        """Analyze questions from a specific dataset."""
        if dataset_name == 'extraneous_supervised':
            data = self.load_jsonl(str(self.extraneous_dir / "supervised.jsonl"))
        elif dataset_name == 'extraneous_preference':
            data = self.load_jsonl(str(self.extraneous_dir / "preference.jsonl"))
        elif dataset_name == 'unsolvable':
            data = self.load_jsonl(str(self.unsolvable_dir / "unsolvable.jsonl"))
        elif dataset_name == 'solvability':
            data = self.load_jsonl(str(self.unsolvable_dir / "solvability.jsonl"))
        else:
            print(f"Unknown dataset: {dataset_name}")
            return {}
        
        if not data:
            return {}
        
        # Analyze each question
        question_features = []
        for record in data:
            if 'prompt' in record:
                features = self.analyze_question_difficulty(record)
                features['category'] = record.get('category', 'unknown')
                features['dataset'] = dataset_name
                question_features.append(features)
        
        if not question_features:
            return {}
        
        # Aggregate statistics
        df = pd.DataFrame(question_features)
        
        analysis = {
            'total_questions': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'difficulty_distribution': df['difficulty_level'].value_counts().to_dict(),
            'average_difficulty_score': df['difficulty_score'].mean(),
            'difficulty_std': df['difficulty_score'].std(),
            'word_count_stats': {
                'mean': df['word_count'].mean(),
                'std': df['word_count'].std(),
                'min': df['word_count'].min(),
                'max': df['word_count'].max()
            },
            'physics_terms_stats': {
                'mean': df['total_physics_terms'].mean(),
                'std': df['total_physics_terms'].std(),
                'min': df['total_physics_terms'].min(),
                'max': df['total_physics_terms'].max()
            },
            'numerical_content': {
                'questions_with_numbers': df['has_numbers'].sum(),
                'questions_with_units': df['has_units'].sum(),
                'average_numbers_per_question': df['number_count'].mean(),
                'average_units_per_question': df['unit_count'].mean()
            }
        }
        
        return analysis
    
    def plot_question_difficulty_distribution(self, save_path: str = None):
        """Plot difficulty distribution across all datasets."""
        datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Question Difficulty Analysis Across Datasets', fontsize=16, fontweight='bold')
        
        for i, dataset in enumerate(datasets):
            analysis = self.analyze_dataset_questions(dataset)
            if not analysis:
                continue
            
            row, col = i // 2, i % 2
            
            # Difficulty level distribution
            difficulty_dist = analysis.get('difficulty_distribution', {})
            if difficulty_dist:
                levels = list(difficulty_dist.keys())
                counts = list(difficulty_dist.values())
                
                axes[row, col].bar(levels, counts, alpha=0.7, color=plt.cm.Set3(i))
                axes[row, col].set_title(f'{dataset.replace("_", " ").title()}')
                axes[row, col].set_ylabel('Number of Questions')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Difficulty distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_physics_concept_usage(self, save_path: str = None):
        """Plot usage of physics concepts across datasets."""
        datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
        
        # Collect physics term usage data
        concept_usage = defaultdict(lambda: defaultdict(int))
        
        for dataset in datasets:
            analysis = self.analyze_dataset_questions(dataset)
            if not analysis:
                continue
            
            # Load raw data to get detailed physics term usage
            if dataset.startswith('extraneous'):
                if 'supervised' in dataset:
                    data = self.load_jsonl(str(self.extraneous_dir / "supervised.jsonl"))
                else:
                    data = self.load_jsonl(str(self.extraneous_dir / "preference.jsonl"))
            else:
                if 'unsolvable' in dataset:
                    data = self.load_jsonl(str(self.unsolvable_dir / "unsolvable.jsonl"))
                else:
                    data = self.load_jsonl(str(self.unsolvable_dir / "solvability.jsonl"))
            
            for record in data:
                if 'prompt' in record:
                    features = self.extract_question_features(record['prompt'])
                    for concept, count in features['physics_terms'].items():
                        concept_usage[concept][dataset] += count
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        concepts = list(concept_usage.keys())
        datasets_clean = [d.replace('_', ' ').title() for d in datasets]
        
        x = np.arange(len(concepts))
        width = 0.2
        
        for i, dataset in enumerate(datasets):
            counts = [concept_usage[concept][dataset] for concept in concepts]
            ax.bar(x + i * width, counts, width, label=datasets_clean[i], alpha=0.8)
        
        ax.set_xlabel('Physics Concepts')
        ax.set_ylabel('Usage Count')
        ax.set_title('Physics Concept Usage Across Datasets')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(concepts, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Physics concept usage plot saved to {save_path}")
        
        plt.show()
    
    def plot_question_complexity_heatmap(self, save_path: str = None):
        """Create a heatmap showing question complexity by category and dataset."""
        datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
        categories = ['kinematics', 'newton', 'energy', 'momentum', 'circuits']
        
        # Create complexity matrix
        complexity_matrix = np.zeros((len(datasets), len(categories)))
        
        for i, dataset in enumerate(datasets):
            analysis = self.analyze_dataset_questions(dataset)
            if not analysis:
                continue
            
            # Load raw data for detailed analysis
            if dataset.startswith('extraneous'):
                if 'supervised' in dataset:
                    data = self.load_jsonl(str(self.extraneous_dir / "supervised.jsonl"))
                else:
                    data = self.load_jsonl(str(self.extraneous_dir / "preference.jsonl"))
            else:
                if 'unsolvable' in dataset:
                    data = self.load_jsonl(str(self.unsolvable_dir / "unsolvable.jsonl"))
                else:
                    data = self.load_jsonl(str(self.unsolvable_dir / "solvability.jsonl"))
            
            category_difficulties = defaultdict(list)
            for record in data:
                if 'prompt' in record and 'category' in record:
                    features = self.analyze_question_difficulty(record)
                    category_difficulties[record['category']].append(features['difficulty_score'])
            
            for j, category in enumerate(categories):
                if category in category_difficulties:
                    complexity_matrix[i, j] = np.mean(category_difficulties[category])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(complexity_matrix, 
                   xticklabels=categories, 
                   yticklabels=[d.replace('_', ' ').title() for d in datasets],
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   ax=ax)
        
        ax.set_title('Question Complexity Heatmap\n(Average Difficulty Score by Category and Dataset)')
        ax.set_xlabel('Physics Category')
        ax.set_ylabel('Dataset')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Complexity heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_text_length_analysis(self, save_path: str = None):
        """Analyze and plot text length patterns across datasets."""
        datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Question Text Length Analysis', fontsize=16, fontweight='bold')
        
        for i, dataset in enumerate(datasets):
            analysis = self.analyze_dataset_questions(dataset)
            if not analysis:
                continue
            
            row, col = i // 2, i % 2
            
            # Load raw data for detailed analysis
            if dataset.startswith('extraneous'):
                if 'supervised' in dataset:
                    data = self.load_jsonl(str(self.extraneous_dir / "supervised.jsonl"))
                else:
                    data = self.load_jsonl(str(self.extraneous_dir / "preference.jsonl"))
            else:
                if 'unsolvable' in dataset:
                    data = self.load_jsonl(str(self.unsolvable_dir / "unsolvable.jsonl"))
                else:
                    data = self.load_jsonl(str(self.unsolvable_dir / "solvability.jsonl"))
            
            word_counts = []
            for record in data:
                if 'prompt' in record:
                    features = self.extract_question_features(record['prompt'])
                    word_counts.append(features['word_count'])
            
            if word_counts:
                axes[row, col].hist(word_counts, bins=20, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
                axes[row, col].set_title(f'{dataset.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Word Count')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].axvline(np.mean(word_counts), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(word_counts):.1f}')
                axes[row, col].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Text length analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_question_analysis_report(self, output_file: str = None) -> str:
        """Generate a comprehensive report of question analysis."""
        datasets = ['extraneous_supervised', 'extraneous_preference', 'unsolvable', 'solvability']
        
        report = []
        report.append("=" * 60)
        report.append("PHYSICS QUESTION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        for dataset in datasets:
            analysis = self.analyze_dataset_questions(dataset)
            if not analysis:
                continue
            
            report.append(f"{dataset.replace('_', ' ').upper()}")
            report.append("-" * 40)
            report.append(f"Total questions: {analysis['total_questions']}")
            report.append(f"Categories: {analysis['categories']}")
            report.append(f"Difficulty distribution: {analysis['difficulty_distribution']}")
            report.append(f"Average difficulty score: {analysis['average_difficulty_score']:.2f}")
            report.append(f"Difficulty std: {analysis['difficulty_std']:.2f}")
            report.append(f"Word count stats: {analysis['word_count_stats']}")
            report.append(f"Physics terms stats: {analysis['physics_terms_stats']}")
            report.append(f"Numerical content: {analysis['numerical_content']}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Question analysis report saved to {output_file}")
        
        return report_text


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze physics questions across datasets')
    parser.add_argument('--data-dir', type=str, help='Base directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='./viz_output', help='Output directory for plots and reports')
    parser.add_argument('--difficulty', action='store_true', help='Generate difficulty distribution plots')
    parser.add_argument('--concepts', action='store_true', help='Generate physics concept usage plots')
    parser.add_argument('--complexity', action='store_true', help='Generate complexity heatmap')
    parser.add_argument('--text-length', action='store_true', help='Generate text length analysis')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = QuestionAnalyzer(args.data_dir)
    
    if args.all or args.difficulty:
        print("Generating difficulty distribution plots...")
        analyzer.plot_question_difficulty_distribution(os.path.join(args.output_dir, 'difficulty_distribution.png'))
    
    if args.all or args.concepts:
        print("Generating physics concept usage plots...")
        analyzer.plot_physics_concept_usage(os.path.join(args.output_dir, 'concept_usage.png'))
    
    if args.all or args.complexity:
        print("Generating complexity heatmap...")
        analyzer.plot_question_complexity_heatmap(os.path.join(args.output_dir, 'complexity_heatmap.png'))
    
    if args.all or args.text_length:
        print("Generating text length analysis...")
        analyzer.plot_text_length_analysis(os.path.join(args.output_dir, 'text_length_analysis.png'))
    
    # Always generate report
    print("Generating question analysis report...")
    report = analyzer.generate_question_analysis_report(os.path.join(args.output_dir, 'question_analysis_report.txt'))
    print(report)


if __name__ == "__main__":
    main()
