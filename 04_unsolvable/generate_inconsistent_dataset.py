import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class InconsistentProblem:
    category: str
    prompt: str
    important_facts: List[str]
    inconsistent_facts: List[str]
    inconsistencies: List[Dict[str, str]]
    rationale: str
    label: str
    metadata: Dict[str, str]


def format_number(value: float, digits: int = 3) -> str:
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-2 or magnitude >= 1e4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def generate_kinematics_inconsistent(rng: random.Random) -> InconsistentProblem:
    # Base solvable: s = ut + 1/2 a t^2
    u = rng.uniform(1.0, 9.0)
    a = rng.uniform(-2.0, 3.0)
    t = rng.uniform(2.0, 12.0)
    s = u * t + 0.5 * a * t * t

    # Introduce inconsistency: contradictory time or acceleration stated twice
    t_conflict = t * rng.uniform(1.5, 2.2)
    a_conflict = a + rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)

    important = [
        f"Initial speed u = {format_number(u)} m/s",
        f"Acceleration a = {format_number(a)} m/s^2 (first statement)",
        f"Time t = {format_number(t)} s (first statement)",
        "Use s = ut + 1/2 a t^2",
    ]
    inconsistent_facts = [
        f"Later, the acceleration is stated as {format_number(a_conflict)} m/s^2 (contradicts first).",
        f"Elsewhere, the time is stated as {format_number(t_conflict)} s (contradicts first).",
    ]

    inconsistencies = [
        {
            "type": "contradictory_value",
            "field": "acceleration",
            "value_a": f"{format_number(a)} m/s^2",
            "value_b": f"{format_number(a_conflict)} m/s^2",
            "units": "m/s^2",
            "description": "Acceleration given two different numeric values."
        },
        {
            "type": "contradictory_value",
            "field": "time",
            "value_a": f"{format_number(t)} s",
            "value_b": f"{format_number(t_conflict)} s",
            "units": "s",
            "description": "Time given two different numeric values."
        },
    ]

    prompt = (
        "A runner moves in a straight line with uniform acceleration. "
        f"Initially, their speed is {format_number(u)} m/s. "
        f"Their acceleration is {format_number(a)} m/s^2 for {format_number(t)} s. "
        f"Later, the description says the acceleration is {format_number(a_conflict)} m/s^2 "
        f"and the time interval is {format_number(t_conflict)} s. "
        "What is the displacement over this time?"
    )

    rationale = (
        "The problem is inconsistent: acceleration and time are both reported with two different values. "
        "Without a single consistent set of parameters, the displacement cannot be uniquely determined."
    )

    return InconsistentProblem(
        category="kinematics",
        prompt=prompt,
        important_facts=important,
        inconsistent_facts=inconsistent_facts,
        inconsistencies=inconsistencies,
        rationale=rationale,
        label="inconsistent",
        metadata={"equation": "s = ut + 1/2 a t^2"},
    )


def generate_newton_inconsistent(rng: random.Random) -> InconsistentProblem:
    # Base: F = ma, introduce mass contradiction and unit mismatch on force
    m = rng.uniform(2.0, 25.0)
    a = rng.uniform(0.5, 5.0)
    F = m * a
    m_conflict = m * rng.uniform(1.6, 2.4)
    F_unit_wrong = F * 1e3  # mistakenly in mN but labeled as N

    important = [
        f"Mass m = {format_number(m)} kg (first statement)",
        f"Acceleration a = {format_number(a)} m/s^2",
        f"Net force F = {format_number(F)} N (from F = ma)",
        "Use F = ma",
    ]
    inconsistent_facts = [
        f"Elsewhere, mass is stated as {format_number(m_conflict)} kg.",
        f"In another line, the applied force is {format_number(F_unit_wrong)} N (likely mN mislabeled).",
    ]
    inconsistencies = [
        {
            "type": "contradictory_value",
            "field": "mass",
            "value_a": f"{format_number(m)} kg",
            "value_b": f"{format_number(m_conflict)} kg",
            "units": "kg",
            "description": "Mass given two different values."
        },
        {
            "type": "unit_mismatch",
            "field": "force",
            "value_a": f"{format_number(F)} N",
            "value_b": f"{format_number(F_unit_wrong)} N",
            "units": "N vs mN",
            "description": "Force magnitude implies millinewtons but labeled as newtons."
        },
    ]

    prompt = (
        "A net horizontal force accelerates a crate on a frictionless surface. "
        f"The mass is {format_number(m)} kg and the acceleration is {format_number(a)} m/s^2. "
        f"However, an equipment table lists the mass as {format_number(m_conflict)} kg, "
        f"and another note reports the force as {format_number(F_unit_wrong)} N. "
        "What is the acceleration of the crate?"
    )
    rationale = (
        "Inconsistent mass values and a likely unit mismatch for force prevent a unique solution under F = ma."
    )
    return InconsistentProblem(
        category="newton",
        prompt=prompt,
        important_facts=important,
        inconsistent_facts=inconsistent_facts,
        inconsistencies=inconsistencies,
        rationale=rationale,
        label="inconsistent",
        metadata={"equation": "F = ma"},
    )


def generate_energy_inconsistent(rng: random.Random) -> InconsistentProblem:
    # Base: mgh = 1/2 m v^2; introduce height contradiction and gravity mismatch
    h = rng.uniform(5.0, 60.0)
    g = 9.8
    h_conflict = h * rng.uniform(0.4, 0.7)
    g_conflict = 3.7  # Mars gravity listed elsewhere

    important = [
        f"Height h = {format_number(h)} m (first statement)",
        f"g = {g} m/s^2",
        "Neglect air resistance",
        "Use mgh = 1/2 m v^2",
    ]
    inconsistent_facts = [
        f"Elsewhere, height is stated as {format_number(h_conflict)} m.",
        f"Another source lists g = {g_conflict} m/s^2 (planet mismatch).",
    ]
    inconsistencies = [
        {
            "type": "contradictory_value",
            "field": "height",
            "value_a": f"{format_number(h)} m",
            "value_b": f"{format_number(h_conflict)} m",
            "units": "m",
            "description": "Height given two different values."
        },
        {
            "type": "equation_conflict",
            "field": "gravity",
            "value_a": f"{g} m/s^2",
            "value_b": f"{g_conflict} m/s^2",
            "units": "m/s^2",
            "description": "Two different gravitational accelerations referenced."
        },
    ]

    prompt = (
        "An object is dropped from rest from a height h. Ignore air resistance. "
        f"First, take h = {format_number(h)} m and g = {g} m/s^2. "
        f"Elsewhere the description lists h = {format_number(h_conflict)} m and g = {g_conflict} m/s^2. "
        "What is the impact speed?"
    )
    rationale = (
        "The problem specifies conflicting values for height and gravitational acceleration, so the impact speed is not uniquely defined."
    )
    return InconsistentProblem(
        category="energy",
        prompt=prompt,
        important_facts=important,
        inconsistent_facts=inconsistent_facts,
        inconsistencies=inconsistencies,
        rationale=rationale,
        label="inconsistent",
        metadata={"equation": "mgh = 1/2 m v^2"},
    )


def generate_momentum_inconsistent(rng: random.Random) -> InconsistentProblem:
    # Base: 1D elastic collision; introduce mass swap contradiction and inelastic/external loss note
    m1 = rng.uniform(0.5, 5.0)
    m2 = rng.uniform(0.5, 5.0)
    u1 = rng.uniform(1.0, 8.0)
    swap = rng.random() < 0.5

    important = [
        f"Masses m1 = {format_number(m1)} kg, m2 = {format_number(m2)} kg (first statement)",
        f"Initial velocities u1 = {format_number(u1)} m/s, u2 = 0 m/s",
        "Perfectly elastic collision, 1D",
    ]
    inconsistent_facts = [
        (f"A later line swaps the identifiers: m1 = {format_number(m2)} kg, m2 = {format_number(m1)} kg"
         if swap else
         f"A later line lists m1 = {format_number(m1)} kg, m2 = {format_number(m2)} kg but calls the collision inelastic with 30% energy loss."),
    ]
    inconsistencies = [
        ({
            "type": "contradictory_value",
            "field": "labels",
            "value_a": f"m1 = {format_number(m1)} kg, m2 = {format_number(m2)} kg",
            "value_b": f"m1 = {format_number(m2)} kg, m2 = {format_number(m1)} kg",
            "units": "kg",
            "description": "Mass labels swapped later, causing ambiguity."
        } if swap else {
            "type": "equation_conflict",
            "field": "elasticity",
            "value_a": "perfectly elastic",
            "value_b": "30% kinetic energy loss",
            "units": "",
            "description": "Elastic vs inelastic statements conflict."
        })
    ]

    prompt = (
        "Two carts collide on a frictionless track. One cart is initially at rest. "
        f"Initially, m1 = {format_number(m1)} kg moves at {format_number(u1)} m/s, m2 = {format_number(m2)} kg is at rest. "
        + (
            f"Elsewhere, labels are swapped: m1 = {format_number(m2)} kg and m2 = {format_number(m1)} kg. "
            if swap
            else "Elsewhere, the collision is described as inelastic with 30% kinetic energy lost. "
        )
        + "Find the final speeds of the carts."
    )
    rationale = (
        "Conflicting assumptions about either mass labels or collision type prevent a unique solution."
    )
    return InconsistentProblem(
        category="momentum",
        prompt=prompt,
        important_facts=important,
        inconsistent_facts=inconsistent_facts,
        inconsistencies=inconsistencies,
        rationale=rationale,
        label="inconsistent",
        metadata={"equations": "elastic 1D collision formulas"},
    )


def generate_circuit_inconsistent(rng: random.Random) -> InconsistentProblem:
    # Base: V = IR; introduce resistor value contradiction and voltage unit mismatch
    V = rng.uniform(3.0, 24.0)
    R = rng.uniform(10.0, 200.0)
    R_conflict = R * rng.uniform(1.8, 3.2)
    V_conflict_mV = V * 1e3

    important = [
        f"Voltage V = {format_number(V)} V (first statement)",
        f"Resistance R = {format_number(R)} Ω (first statement)",
        "Use Ohm's law V = IR",
    ]
    inconsistent_facts = [
        f"Elsewhere, the resistor value is listed as {format_number(R_conflict)} Ω.",
        f"A measurement sheet lists the voltage as {format_number(V_conflict_mV)} V (likely in mV).",
    ]
    inconsistencies = [
        {
            "type": "contradictory_value",
            "field": "resistance",
            "value_a": f"{format_number(R)} Ω",
            "value_b": f"{format_number(R_conflict)} Ω",
            "units": "Ω",
            "description": "Resistor value differs in two places."
        },
        {
            "type": "unit_mismatch",
            "field": "voltage",
            "value_a": f"{format_number(V)} V",
            "value_b": f"{format_number(V_conflict_mV)} V",
            "units": "V vs mV",
            "description": "Voltage magnitude indicates millivolts but labeled volts."
        },
    ]

    prompt = (
        "In a simple series circuit, a resistor is connected to a DC source. "
        f"The source is {format_number(V)} V and the resistor is {format_number(R)} Ω. "
        f"Elsewhere, the resistor is listed as {format_number(R_conflict)} Ω and the voltage measurement is {format_number(V_conflict_mV)} V. "
        "Find the current."
    )
    rationale = (
        "Conflicting resistor and voltage values (with unit mismatch) make the current undefined without clarifying which values are correct."
    )
    return InconsistentProblem(
        category="circuits",
        prompt=prompt,
        important_facts=important,
        inconsistent_facts=inconsistent_facts,
        inconsistencies=inconsistencies,
        rationale=rationale,
        label="inconsistent",
        metadata={"equation": "V = IR"},
    )


GENERATOR_FUNCS = [
    generate_kinematics_inconsistent,
    generate_newton_inconsistent,
    generate_energy_inconsistent,
    generate_momentum_inconsistent,
    generate_circuit_inconsistent,
]


def build_unsolvable_record(p: InconsistentProblem) -> Dict:
    return {
        "category": p.category,
        "prompt": p.prompt,
        "important_facts": p.important_facts,
        "inconsistent_facts": p.inconsistent_facts,
        "inconsistencies": p.inconsistencies,
        "rationale": p.rationale,
        "label": p.label,
        "metadata": p.metadata,
    }


def generate_dataset(n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    records: List[Dict] = []
    for i in range(n):
        gen = GENERATOR_FUNCS[i % len(GENERATOR_FUNCS)]
        p = gen(rng)
        records.append(build_unsolvable_record(p))
    return records


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physics problems with internal inconsistencies for detection tasks.")
    parser.add_argument("--count", type=int, default=50, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "samples"), help="Output directory")
    args = parser.parse_args()

    records = generate_dataset(args.count, args.seed)
    outdir = Path(args.outdir)
    save_jsonl(records, outdir / "unsolvable.jsonl")
    # Minimal schema
    schema = {
        "category": "str",
        "prompt": "str",
        "important_facts": "List[str]",
        "inconsistent_facts": "List[str]",
        "inconsistencies": [
            {
                "type": "str",
                "field": "str",
                "value_a": "str",
                "value_b": "str",
                "units": "str",
                "description": "str",
            }
        ],
        "rationale": "str",
        "label": "inconsistent",
        "metadata": "Dict[str, Any]",
    }
    save_jsonl([schema], outdir / "schema_unsolvable.jsonl")

    print(f"Wrote {len(records)} records to {outdir / 'unsolvable.jsonl'}")


if __name__ == "__main__":
    main()


