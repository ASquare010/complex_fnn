"""Deterministic retrieval/composition controls; no external model writes targets."""

import random


def relation_example(rng, hops, renamed=False):
    pool = range(100, 200) if renamed else range(100)
    names = [f"e{x:03d}" for x in rng.sample(list(pool), hops + 9)]
    path = names[: hops + 1]
    edges = list(zip(path[:-1], path[1:]))
    edges += list(zip(names[hops + 1 :: 2], names[hops + 2 :: 2]))
    rng.shuffle(edges)
    facts = ";".join(f"{a}>{b}" for a, b in edges)
    return {
        "prompt": f"facts:{facts}\nfollow:{path[0]},{hops}\nanswer:",
        "answer": path[-1],
        "task": "lookup" if hops == 1 else "chain",
        "difficulty": hops,
        "oracle": {"edges": edges, "start": path[0], "hops": hops},
    }


def arithmetic_example(rng, operations):
    expression, value = str(rng.randrange(10)), None
    value = int(expression)
    for _ in range(operations):
        right = rng.randrange(10)
        operation = rng.choice(("+", "*"))
        expression = f"({expression}{operation}{right})"
        value = (value + right if operation == "+" else value * right) % 97
    return {
        "prompt": f"calculate modulo 97:{expression}\nanswer:",
        "answer": str(value),
        "task": "arithmetic",
        "difficulty": operations,
        "oracle": {"expression": expression},
    }


def generate(seed, count, ood=False):
    if count < 1:
        raise ValueError("Each split needs at least one example")
    rng = random.Random(seed)
    examples, seen = [], set()
    while len(examples) < count:
        index = len(examples) % 3
        if index == 0:
            row = relation_example(rng, 1, renamed=ood)
        elif index == 1:
            row = relation_example(rng, rng.randint(4, 6) if ood else rng.randint(2, 3), ood)
        else:
            row = arithmetic_example(rng, rng.randint(4, 6) if ood else rng.randint(1, 3))
        # Identity ignores fact presentation order, preventing reordered duplicates.
        if "edges" in row["oracle"]:
            key = (
                tuple(sorted(map(tuple, row["oracle"]["edges"]))),
                row["oracle"]["start"],
                row["difficulty"],
            )
        else:
            key = row["prompt"]
        if key not in seen:
            seen.add(key)
            examples.append(row)
    return examples
