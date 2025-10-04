#!/usr/bin/env python3
"""
Demo of Physics Question Generator
Shows sample questions from each category.
"""

from physics_question_generator import PhysicsQuestionGenerator


def demo_all_question_types():
    """Generate and display one question of each type."""
    generator = PhysicsQuestionGenerator()
    
    print("Physics Question Generator - Demo")
    print("=" * 60)
    print("Sample questions from each category:\n")
    
    for question_type in generator.question_types:
        print(f"{'='*60}")
        print(f"CATEGORY: {question_type.replace('_', ' ').upper()}")
        print(f"{'='*60}")
        
        question_data = generator.generate_question(question_type)
        print(f"\nQuestion: {question_data['question']}")
        print(f"\nAnswer: {question_data['answer']}")
        print(question_data['solution'])
        print()


def demo_random_questions(num_questions=5):
    """Generate and display random questions."""
    generator = PhysicsQuestionGenerator()
    
    print(f"\n{'='*60}")
    print(f"RANDOM QUESTIONS ({num_questions} samples)")
    print(f"{'='*60}")
    
    for i in range(num_questions):
        question_data = generator.generate_question()
        print(f"\n--- Question {i+1} ---")
        print(f"Type: {question_data['type'].replace('_', ' ').title()}")
        print(f"Question: {question_data['question']}")
        print(f"Answer: {question_data['answer']}")


if __name__ == "__main__":
    # Show one question from each category
    demo_all_question_types()
    
    # Show some random questions
    demo_random_questions()
