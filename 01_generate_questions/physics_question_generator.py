#!/usr/bin/env python3
"""
Physics Question Generator
A program that generates simple mechanics physics questions with solutions.
"""

import random
import math


class PhysicsQuestionGenerator:
    """Generates simple mechanics physics questions with solutions."""
    
    def __init__(self):
        self.question_types = [
            'kinematics_velocity',
            'kinematics_acceleration', 
            'kinematics_displacement',
            'force_newton_second',
            'kinetic_energy',
            'potential_energy',
            'work_energy',
            'simple_collision'
        ]
    
    def generate_question(self, question_type=None):
        """Generate a random physics question or a specific type."""
        if question_type is None:
            question_type = random.choice(self.question_types)
        
        method_name = f"generate_{question_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    def generate_kinematics_velocity(self):
        """Generate a velocity calculation question."""
        distance = random.randint(50, 500)  # meters
        time = random.randint(5, 30)  # seconds
        
        question = f"A car travels {distance} meters in {time} seconds at constant speed. What is the car's velocity?"
        
        velocity = distance / time
        solution = f"""
Solution:
Using the formula: velocity = distance / time
v = {distance} m / {time} s = {velocity:.2f} m/s

Answer: {velocity:.2f} m/s
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{velocity:.2f} m/s",
            'type': 'kinematics_velocity'
        }
    
    def generate_kinematics_acceleration(self):
        """Generate an acceleration calculation question."""
        initial_velocity = random.randint(0, 20)  # m/s
        final_velocity = random.randint(initial_velocity + 10, initial_velocity + 50)  # m/s
        time = random.randint(3, 15)  # seconds
        
        question = f"A bicycle accelerates from {initial_velocity} m/s to {final_velocity} m/s in {time} seconds. What is the acceleration?"
        
        acceleration = (final_velocity - initial_velocity) / time
        solution = f"""
Solution:
Using the formula: acceleration = (final velocity - initial velocity) / time
a = ({final_velocity} - {initial_velocity}) m/s / {time} s = {acceleration:.2f} m/s²

Answer: {acceleration:.2f} m/s²
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{acceleration:.2f} m/s²",
            'type': 'kinematics_acceleration'
        }
    
    def generate_kinematics_displacement(self):
        """Generate a displacement calculation question."""
        initial_velocity = random.randint(5, 25)  # m/s
        time = random.randint(4, 12)  # seconds
        acceleration = random.randint(1, 5)  # m/s²
        
        question = f"An object starts with an initial velocity of {initial_velocity} m/s and accelerates at {acceleration} m/s² for {time} seconds. What is the displacement?"
        
        displacement = initial_velocity * time + 0.5 * acceleration * time**2
        solution = f"""
Solution:
Using the formula: displacement = initial_velocity × time + ½ × acceleration × time²
s = {initial_velocity} × {time} + ½ × {acceleration} × {time}²
s = {initial_velocity * time} + ½ × {acceleration} × {time**2}
s = {initial_velocity * time} + {0.5 * acceleration * time**2}
s = {displacement:.2f} m

Answer: {displacement:.2f} m
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{displacement:.2f} m",
            'type': 'kinematics_displacement'
        }
    
    def generate_force_newton_second(self):
        """Generate a force calculation question using Newton's second law."""
        mass = random.randint(5, 50)  # kg
        acceleration = random.randint(2, 15)  # m/s²
        
        question = f"A {mass} kg object accelerates at {acceleration} m/s². What is the net force acting on the object?"
        
        force = mass * acceleration
        solution = f"""
Solution:
Using Newton's Second Law: Force = mass × acceleration
F = {mass} kg × {acceleration} m/s² = {force} N

Answer: {force} N
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{force} N",
            'type': 'force_newton_second'
        }
    
    def generate_kinetic_energy(self):
        """Generate a kinetic energy calculation question."""
        mass = random.randint(2, 20)  # kg
        velocity = random.randint(5, 30)  # m/s
        
        question = f"A {mass} kg object is moving at {velocity} m/s. What is its kinetic energy?"
        
        kinetic_energy = 0.5 * mass * velocity**2
        solution = f"""
Solution:
Using the formula: Kinetic Energy = ½ × mass × velocity²
KE = ½ × {mass} kg × ({velocity} m/s)²
KE = ½ × {mass} × {velocity**2}
KE = {kinetic_energy:.2f} J

Answer: {kinetic_energy:.2f} J
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{kinetic_energy:.2f} J",
            'type': 'kinetic_energy'
        }
    
    def generate_potential_energy(self):
        """Generate a gravitational potential energy calculation question."""
        mass = random.randint(1, 15)  # kg
        height = random.randint(2, 25)  # m
        g = 9.8  # m/s²
        
        question = f"A {mass} kg object is at a height of {height} m above the ground. What is its gravitational potential energy? (Use g = 9.8 m/s²)"
        
        potential_energy = mass * g * height
        solution = f"""
Solution:
Using the formula: Potential Energy = mass × gravity × height
PE = {mass} kg × 9.8 m/s² × {height} m
PE = {potential_energy:.2f} J

Answer: {potential_energy:.2f} J
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{potential_energy:.2f} J",
            'type': 'potential_energy'
        }
    
    def generate_work_energy(self):
        """Generate a work calculation question."""
        force = random.randint(10, 100)  # N
        distance = random.randint(3, 20)  # m
        
        question = f"A constant force of {force} N is applied to move an object {distance} m in the direction of the force. How much work is done?"
        
        work = force * distance
        solution = f"""
Solution:
Using the formula: Work = Force × Distance (when force is in the direction of motion)
W = {force} N × {distance} m = {work} J

Answer: {work} J
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{work} J",
            'type': 'work_energy'
        }
    
    def generate_simple_collision(self):
        """Generate a simple elastic collision question (1D)."""
        m1 = random.randint(2, 10)  # kg
        m2 = random.randint(2, 10)  # kg
        v1_initial = random.randint(5, 15)  # m/s
        v2_initial = 0  # stationary target
        
        question = f"A {m1} kg object moving at {v1_initial} m/s collides elastically with a stationary {m2} kg object. What is the velocity of the first object after collision?"
        
        # Elastic collision formulas
        v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
        v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)
        
        solution = f"""
Solution:
For elastic collision, using conservation of momentum and energy:
v1_final = ((m1 - m2) × v1_initial + 2 × m2 × v2_initial) / (m1 + m2)
v1_final = (({m1} - {m2}) × {v1_initial} + 2 × {m2} × 0) / ({m1} + {m2})
v1_final = ({m1 - m2} × {v1_initial}) / {m1 + m2}
v1_final = {(m1 - m2) * v1_initial} / {m1 + m2} = {v1_final:.2f} m/s

Answer: {v1_final:.2f} m/s
"""
        
        return {
            'question': question,
            'solution': solution,
            'answer': f"{v1_final:.2f} m/s",
            'type': 'simple_collision'
        }


def main():
    """Main function to demonstrate the question generator."""
    generator = PhysicsQuestionGenerator()
    
    print("Physics Question Generator")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Generate random question")
        print("2. Choose question type")
        print("3. Generate multiple questions")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            question_data = generator.generate_question()
            print(f"\nQuestion Type: {question_data['type'].replace('_', ' ').title()}")
            print(f"\nQuestion: {question_data['question']}")
            
            show_solution = input("\nShow solution? (y/n): ").strip().lower()
            if show_solution == 'y':
                print(question_data['solution'])
        
        elif choice == '2':
            print("\nAvailable question types:")
            for i, q_type in enumerate(generator.question_types, 1):
                print(f"{i}. {q_type.replace('_', ' ').title()}")
            
            try:
                type_choice = int(input("\nEnter question type number: ")) - 1
                if 0 <= type_choice < len(generator.question_types):
                    question_data = generator.generate_question(generator.question_types[type_choice])
                    print(f"\nQuestion: {question_data['question']}")
                    
                    show_solution = input("\nShow solution? (y/n): ").strip().lower()
                    if show_solution == 'y':
                        print(question_data['solution'])
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == '3':
            try:
                num_questions = int(input("How many questions? "))
                for i in range(num_questions):
                    question_data = generator.generate_question()
                    print(f"\n--- Question {i+1} ---")
                    print(f"Type: {question_data['type'].replace('_', ' ').title()}")
                    print(f"Question: {question_data['question']}")
                    print(f"Answer: {question_data['answer']}")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")


if __name__ == "__main__":
    main()
