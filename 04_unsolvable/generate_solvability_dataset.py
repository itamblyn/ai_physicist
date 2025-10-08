import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Item:
    category: str
    prompt: str
    solvable: bool
    metadata: Dict[str, str]


FILLER_SENTENCES: List[str] = [
    "Assume standard Earth conditions unless stated otherwise.",
    "All measurements are taken with calibrated equipment.",
    "Neglect minor losses and idealize where conventional.",
    "Report the final value rounded sensibly to significant figures.",
    "Use consistent SI units throughout the calculation.",
    "Treat the system as isolated unless interactions are specified.",
]


def clamp_length(text: str, target_chars: int) -> str:
    if len(text) == target_chars:
        return text
    if len(text) > target_chars:
        return text[:target_chars]
    # pad with neutral characters (spaces and a period) to reach exact length
    padding = target_chars - len(text)
    return text + (" ") * max(0, padding - 1) + ("." if padding > 0 else "")


def balance_and_normalize(items: List[Item], target_chars: int, jitter: int = 0) -> List[Item]:
    normalized: List[Item] = []
    for it in items:
        # small optional jitter to avoid identical lengths; default 0
        effective_target = target_chars + (random.randint(-jitter, jitter) if jitter > 0 else 0)
        normalized.append(
            Item(
                category=it.category,
                prompt=clamp_length(it.prompt, effective_target),
                solvable=it.solvable,
                metadata=it.metadata,
            )
        )
    return normalized


def scenario_kinematics(seed: int) -> Tuple[Item, Item]:
    rnd = random.Random(seed)
    v0 = rnd.randint(2, 18)  # m/s
    a = rnd.choice([1, 2, 3])  # m/s^2
    t = rnd.randint(2, 9)  # s
    # Solvable: displacement with v0, a, t
    solvable_prompt = (
        f"A cart moves in a straight line with initial speed {v0} m/s and constant acceleration {a} m/s^2 for {t} s. "
        f"What is the displacement during this time? "
        f"Ignore air resistance. "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    # Unsolvable: remove time, or introduce contradiction
    if rnd.random() < 0.5:
        unsolvable_prompt = (
            f"A cart moves in a straight line with initial speed {v0} m/s and constant acceleration {a} m/s^2. "
            f"What is the displacement during the motion? The duration is not specified. "
            f"Ignore air resistance. "
            + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
        )
    else:
        unsolvable_prompt = (
            f"A cart moves in a straight line with initial speed {v0} m/s and constant acceleration {a} m/s^2 for {t} s. "
            f"During the same {t} s it also maintains constant speed with zero acceleration. "
            f"What is the displacement? "
            + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
        )
    return (
        Item("kinematics", solvable_prompt, True, {"type": "displacement_vat"}),
        Item("kinematics", unsolvable_prompt, False, {"type": "missing_or_contradictory_time"}),
    )


def scenario_newton(seed: int) -> Tuple[Item, Item]:
    rnd = random.Random(seed)
    m = rnd.randint(2, 10)  # kg
    f = rnd.randint(10, 40)  # N
    mu = rnd.choice([0.1, 0.2, 0.3])
    solvable_prompt = (
        f"A block of mass {m} kg on a horizontal surface is pushed by a {f} N force. "
        f"The kinetic friction coefficient is {mu}. What is the block's acceleration? "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    unsolvable_prompt = (
        f"A block of mass {m} kg on a horizontal surface is pushed by an unknown force. "
        f"The kinetic friction coefficient is {mu}. What is the block's acceleration? "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    return (
        Item("newton", solvable_prompt, True, {"type": "f_net_acceleration"}),
        Item("newton", unsolvable_prompt, False, {"type": "missing_force"}),
    )


def scenario_energy(seed: int) -> Tuple[Item, Item]:
    rnd = random.Random(seed)
    m = rnd.randint(1, 5)
    h = rnd.randint(2, 15)
    solvable_prompt = (
        f"A {m} kg object is released from rest at a height of {h} m above the ground. "
        f"Ignoring air resistance, what is its speed just before impact? "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    unsolvable_prompt = (
        f"An object is released from rest above the ground. "
        f"Ignoring air resistance, what is its speed just before impact? The height is unspecified. "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    return (
        Item("energy", solvable_prompt, True, {"type": "gh_to_v"}),
        Item("energy", unsolvable_prompt, False, {"type": "missing_height"}),
    )


def scenario_momentum(seed: int) -> Tuple[Item, Item]:
    rnd = random.Random(seed)
    m1, m2 = rnd.randint(1, 4), rnd.randint(2, 6)
    v1 = rnd.randint(2, 8)
    solvable_prompt = (
        f"On a frictionless track, a {m1} kg cart moving at {v1} m/s collides and sticks to a stationary {m2} kg cart. "
        f"What is their common speed after the collision? "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    unsolvable_prompt = (
        f"On a frictionless track, a cart collides with another and they stick together. "
        f"What is their common speed after the collision? The initial speed is not given. "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    return (
        Item("momentum", solvable_prompt, True, {"type": "inelastic_post_speed"}),
        Item("momentum", unsolvable_prompt, False, {"type": "missing_initial_speed"}),
    )


def scenario_circuits(seed: int) -> Tuple[Item, Item]:
    rnd = random.Random(seed)
    v = rnd.choice([6, 9, 12])
    r = rnd.choice([2, 3, 4, 6])
    solvable_prompt = (
        f"A DC source of {v} V is connected across a resistor of {r} Ω. "
        f"What current flows through the resistor? "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    unsolvable_prompt = (
        f"A DC source is connected across a resistor. "
        f"What current flows through the resistor? The voltage is not stated. "
        + " ".join(rnd.sample(FILLER_SENTENCES, k=3))
    )
    return (
        Item("circuits", solvable_prompt, True, {"type": "ohms_law"}),
        Item("circuits", unsolvable_prompt, False, {"type": "missing_voltage"}),
    )


SCENARIOS = [scenario_kinematics, scenario_newton, scenario_energy, scenario_momentum, scenario_circuits]


def generate_items(count: int, seed: int, target_chars: int, jitter: int) -> List[Item]:
    rnd = random.Random(seed)
    items: List[Item] = []
    while len(items) < count:
        sc = rnd.choice(SCENARIOS)
        pair_seed = rnd.randint(0, 10_000_000)
        solvable_item, unsolvable_item = sc(pair_seed)
        items.extend([solvable_item, unsolvable_item])
    items = items[:count]
    # normalize length to mitigate length-based leakage
    items = balance_and_normalize(items, target_chars=target_chars, jitter=jitter)
    return items


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple physics problems labeled by solvability with length-normalized prompts.")
    parser.add_argument("--count", type=int, default=200, help="Total number of items to generate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--target_chars", type=int, default=280, help="Target character length for prompts")
    parser.add_argument("--jitter", type=int, default=0, help="Optional +- jitter on target length")
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "samples"), help="Output directory")
    args = parser.parse_args()

    items = generate_items(args.count, args.seed, args.target_chars, args.jitter)
    # Ensure class balance exactly 50/50 if possible
    solvable = [it for it in items if it.solvable]
    unsolvable = [it for it in items if not it.solvable]
    n = min(len(solvable), len(unsolvable))
    balanced = (solvable[:n] + unsolvable[:n])[: args.count]
    random.Random(args.seed).shuffle(balanced)

    supervised: List[Dict] = [
        {
            "category": it.category,
            "prompt": it.prompt,
            "solvable": it.solvable,
            "metadata": it.metadata,
        }
        for it in balanced
    ]

    outdir = Path(args.outdir)
    save_jsonl(supervised, outdir / "solvability.jsonl")
    # Minimal schema file
    schema = {
        "category": "str",
        "prompt": "str (length-normalized)",
        "solvable": "bool",
        "metadata": "Dict[str, Any]",
    }
    save_jsonl([schema], outdir / "schema_solvability.jsonl")
    print(f"Wrote {len(supervised)} records to {outdir / 'solvability.jsonl'}")


if __name__ == "__main__":
    main()


