import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PhysicsProblem:
    category: str
    prompt: str
    important_facts: List[str]
    extraneous_facts: List[str]
    solution_steps: List[str]
    final_answer: str
    metadata: Dict[str, str]


def format_number(value: float, digits: int = 3) -> str:
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-2 or magnitude >= 1e4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def generate_kinematics_rng(rng: random.Random) -> PhysicsProblem:
    # Uniform acceleration displacement problem, extraneous: weather, shoe size, color, etc.
    u = rng.uniform(0.0, 10.0)  # initial velocity m/s
    a = rng.uniform(0.5, 3.0) * (-1 if rng.random() < 0.3 else 1)  # m/s^2 (sometimes negative)
    t = rng.uniform(2.0, 12.0)  # seconds
    s = u * t + 0.5 * a * t * t

    important = [
        f"Initial speed u = {format_number(u)} m/s",
        f"Acceleration a = {format_number(a)} m/s^2",
        f"Time t = {format_number(t)} s",
        "Use s = ut + 1/2 a t^2",
    ]
    extraneous = [
        f"The day is {rng.choice(['sunny','cloudy','rainy'])}.",
        f"The runner's shoe size is {rng.choice([8,9,10,11,12])}.",
        f"The track is painted {rng.choice(['blue','green','red'])}.",
        f"There are {rng.randint(50,200)} spectators at the field.",
    ]
    steps = [
        "Identify given values u, a, t and the equation s = ut + 1/2 a t^2.",
        f"Compute ut = {format_number(u)} * {format_number(t)} = {format_number(u*t)} m.",
        f"Compute 1/2 a t^2 = 0.5 * {format_number(a)} * {format_number(t)}^2 = {format_number(0.5*a*t*t)} m.",
        f"Sum to get s = {format_number(s)} m.",
    ]
    answer = f"{format_number(s)} m"
    prompt = (
        "A runner moves in a straight line with uniform acceleration. "
        f"{extraneous[0]} {extraneous[1]} {extraneous[2]} "
        f"Initially, their speed is {format_number(u)} m/s. "
        f"Their acceleration is {format_number(a)} m/s^2 for {format_number(t)} s. "
        f"{extraneous[3]} What is the displacement over this time?"
    )
    return PhysicsProblem(
        category="kinematics",
        prompt=prompt,
        important_facts=important,
        extraneous_facts=extraneous,
        solution_steps=steps,
        final_answer=answer,
        metadata={"equation": "s = ut + 1/2 a t^2"},
    )


def generate_newton_rng(rng: random.Random) -> PhysicsProblem:
    # F = ma; find acceleration or force, with extraneous mass of packaging, brand, etc.
    m = rng.uniform(1.0, 20.0)  # kg
    F = rng.uniform(5.0, 100.0)  # N
    a = F / m
    important = [
        f"Mass m = {format_number(m)} kg",
        f"Net force F = {format_number(F)} N",
        "Use F = ma",
    ]
    extraneous = [
        f"The box brand is {rng.choice(['Acme','Globex','Initech'])}.",
        f"The box is {rng.choice(['red','brown','white'])} with a logo.",
        f"Packaging weighs {format_number(rng.uniform(0.05,0.5))} kg but is removed before measurement.",
    ]
    steps = [
        "From F = ma, solve a = F/m.",
        f"Compute a = {format_number(F)} / {format_number(m)} = {format_number(a)} m/s^2.",
    ]
    answer = f"{format_number(a)} m/s^2"
    prompt = (
        "A horizontal net force is applied to a box on a frictionless surface. "
        f"{extraneous[0]} {extraneous[1]} {extraneous[2]} "
        f"If the net force is {format_number(F)} N and the mass is {format_number(m)} kg, what is the acceleration?"
    )
    return PhysicsProblem(
        category="newton",
        prompt=prompt,
        important_facts=important,
        extraneous_facts=extraneous,
        solution_steps=steps,
        final_answer=answer,
        metadata={"equation": "F = ma"},
    )


def generate_energy_rng(rng: random.Random) -> PhysicsProblem:
    # Energy conservation for a dropped object ignoring air resistance; extraneous height of building name, temperature, etc.
    h = rng.uniform(5.0, 60.0)  # m
    m = rng.uniform(0.2, 10.0)  # kg
    g = 9.8
    v = math.sqrt(2 * g * h)
    important = [
        f"Height h = {format_number(h)} m",
        f"g = {g} m/s^2",
        "Neglect air resistance",
        "Use mgh = 1/2 m v^2",
    ]
    extraneous = [
        f"The object is {rng.choice(['a book','a mug','a small statue'])}.",
        f"The building is named '{rng.choice(['Orion Tower','Maple Hall','Sunrise Complex'])}'.",
        f"Ambient temperature is {rng.randint(15, 30)} °C.",
    ]
    steps = [
        "Set mgh = 1/2 m v^2, mass cancels.",
        f"Solve v = sqrt(2 g h) = sqrt(2*{g}*{format_number(h)}) = {format_number(v)} m/s.",
    ]
    answer = f"{format_number(v)} m/s"
    prompt = (
        "An object is dropped from rest from a height h. Ignore air resistance. "
        f"{extraneous[0]} {extraneous[1]} {extraneous[2]} "
        f"If h = {format_number(h)} m, what is the speed just before impact?"
    )
    return PhysicsProblem(
        category="energy",
        prompt=prompt,
        important_facts=important,
        extraneous_facts=extraneous,
        solution_steps=steps,
        final_answer=answer,
        metadata={"equation": "mgh = 1/2 m v^2"},
    )


def generate_momentum_rng(rng: random.Random) -> PhysicsProblem:
    # Elastic collision in 1D with one object initially at rest; extraneous: colors, music playing, etc.
    m1 = rng.uniform(0.5, 5.0)
    m2 = rng.uniform(0.5, 5.0)
    u1 = rng.uniform(1.0, 8.0)
    u2 = 0.0
    # Elastic collision formulas (1D):
    v1 = (m1 - m2) / (m1 + m2) * u1
    v2 = (2 * m1) / (m1 + m2) * u1
    important = [
        f"Masses m1 = {format_number(m1)} kg, m2 = {format_number(m2)} kg",
        f"Initial velocities u1 = {format_number(u1)} m/s, u2 = 0 m/s",
        "Perfectly elastic collision, 1D",
    ]
    extraneous = [
        f"The carts are {rng.choice(['blue and yellow','black and white','green and orange'])}.",
        f"Background music plays at {rng.randint(60, 140)} BPM.",
        f"The lab floor is {rng.choice(['wood','tile','concrete'])}.",
    ]
    steps = [
        "Use elastic collision results for 1D with u2=0:",
        "v1 = ((m1 - m2)/(m1 + m2)) u1, v2 = (2 m1/(m1 + m2)) u1.",
        f"Compute v1 = {format_number(v1)} m/s and v2 = {format_number(v2)} m/s.",
    ]
    answer = f"v1 = {format_number(v1)} m/s, v2 = {format_number(v2)} m/s"
    prompt = (
        "Two carts collide elastically on a frictionless track. One cart is initially at rest. "
        f"{extraneous[0]} {extraneous[1]} {extraneous[2]} "
        f"Given m1 = {format_number(m1)} kg moving at {format_number(u1)} m/s, and m2 = {format_number(m2)} kg at rest, find the final speeds of both carts."
    )
    return PhysicsProblem(
        category="momentum",
        prompt=prompt,
        important_facts=important,
        extraneous_facts=extraneous,
        solution_steps=steps,
        final_answer=answer,
        metadata={"equations": "elastic 1D collision formulas"},
    )


def generate_circuit_rng(rng: random.Random) -> PhysicsProblem:
    # Simple series circuit: V = IR; extraneous LED color, wire material, etc.
    V = rng.uniform(3.0, 24.0)
    R = rng.uniform(10.0, 200.0)
    I = V / R
    important = [
        f"Voltage V = {format_number(V)} V",
        f"Resistance R = {format_number(R)} Ω",
        "Use Ohm's law V = IR",
    ]
    extraneous = [
        f"The LED color is {rng.choice(['red','green','blue'])}.",
        f"The wires are {rng.choice(['copper','aluminum'])} with plastic insulation.",
        f"The breadboard has {rng.randint(300, 830)} tie points.",
    ]
    steps = [
        "From V = IR, solve I = V/R.",
        f"Compute I = {format_number(V)} / {format_number(R)} = {format_number(I)} A.",
    ]
    answer = f"{format_number(I)} A"
    prompt = (
        "In a simple series circuit, a resistor is connected to a DC source. "
        f"{extraneous[0]} {extraneous[1]} {extraneous[2]} "
        f"If V = {format_number(V)} V across a resistor R = {format_number(R)} Ω, find the current."
    )
    return PhysicsProblem(
        category="circuits",
        prompt=prompt,
        important_facts=important,
        extraneous_facts=extraneous,
        solution_steps=steps,
        final_answer=answer,
        metadata={"equation": "V = IR"},
    )


GENERATOR_FUNCS = [
    generate_kinematics_rng,
    generate_newton_rng,
    generate_energy_rng,
    generate_momentum_rng,
    generate_circuit_rng,
]


def build_supervised_record(p: PhysicsProblem) -> Dict:
    return {
        "category": p.category,
        "prompt": p.prompt,
        "important_facts": p.important_facts,
        "extraneous_facts": p.extraneous_facts,
        "solution_steps": p.solution_steps,
        "final_answer": p.final_answer,
        "metadata": p.metadata,
    }


def build_preference_pair(p: PhysicsProblem, rng: random.Random) -> Dict:
    # Create a high-quality solution and a low-quality one that leans on extraneous info
    chosen = {
        "reasoning": "\n".join(p.solution_steps),
        "answer": p.final_answer,
        "uses_extraneous": False,
        "correct": True,
    }

    # Bad response: mention extraneous facts, compute a wrong numeric via perturbation
    wrong_factor = rng.uniform(1.15, 1.6)
    # Extract numeric from final_answer if possible and perturb; otherwise append note
    bad_answer = p.final_answer
    try:
        # naive parse: first token numeric
        num_str = p.final_answer.split()[0]
        if 'e' in num_str or 'E' in num_str:
            num_val = float(num_str)
        else:
            num_val = float(num_str)
        bad_val = num_val * wrong_factor
        bad_answer = p.final_answer.replace(num_str, format_number(bad_val))
    except Exception:
        bad_answer = p.final_answer + " (likely incorrect)"

    rejected = {
        "reasoning": (
            "The problem mentions details like "
            + ", ".join(p.extraneous_facts[:2])
            + ". These suggest certain adjustments. "
            + "Using the colors and ambient conditions, I estimate the result without core equations."
        ),
        "answer": bad_answer,
        "uses_extraneous": True,
        "correct": False,
    }

    return {
        "category": p.category,
        "prompt": p.prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": p.metadata,
    }


def generate_dataset(n: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    supervised: List[Dict] = []
    preference: List[Dict] = []

    for i in range(n):
        gen = GENERATOR_FUNCS[i % len(GENERATOR_FUNCS)]
        problem = gen(rng)
        supervised.append(build_supervised_record(problem))
        preference.append(build_preference_pair(problem, rng))

    return supervised, preference


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physics problems with extraneous info for RL fine-tuning.")
    parser.add_argument("--count", type=int, default=50, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "samples"), help="Output directory")
    args = parser.parse_args()

    supervised, preference = generate_dataset(args.count, args.seed)
    outdir = Path(args.outdir)
    save_jsonl(supervised, outdir / "supervised.jsonl")
    save_jsonl(preference, outdir / "preference.jsonl")
    # Minimal schema files for reference
    schema_supervised = {
        "category": "str",
        "prompt": "str",
        "important_facts": "List[str]",
        "extraneous_facts": "List[str]",
        "solution_steps": "List[str]",
        "final_answer": "str",
        "metadata": "Dict[str, Any]",
    }
    schema_preference = {
        "category": "str",
        "prompt": "str",
        "chosen": {"reasoning": "str", "answer": "str", "uses_extraneous": "bool", "correct": "bool"},
        "rejected": {"reasoning": "str", "answer": "str", "uses_extraneous": "bool", "correct": "bool"},
        "metadata": "Dict[str, Any]",
    }
    save_jsonl([schema_supervised], outdir / "schema_supervised.jsonl")
    save_jsonl([schema_preference], outdir / "schema_preference.jsonl")

    print(f"Wrote {len(supervised)} supervised records to {outdir / 'supervised.jsonl'}")
    print(f"Wrote {len(preference)} preference pairs to {outdir / 'preference.jsonl'}")


if __name__ == "__main__":
    main()


