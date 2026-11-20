# Make random graph-operation sequences; reference/java_orders_run_tetrad.py applies them in Tetrad.
# (Merged result: reference/fixtures/java_orders.json, used by tests/test_fci.py.)
import json, numpy as np
cases = []
for p, seed in [(10, 0), (10, 1), (30, 2), (30, 3), (100, 4)]:
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(p) for j in range(i+1, p) if rng.random() < min(1, 3.0/(p-1))]
    ops = []
    for _ in range(4 * len(edges)):
        r = rng.random()
        i, j = edges[rng.integers(len(edges))]
        if rng.random() < 0.5: i, j = j, i
        if r < 0.8:
            ops.append(["set", int(i), int(j), ["ARROW", "TAIL", "CIRCLE"][rng.integers(3)]])
        elif r < 0.85:
            ops.append(["reorient", "CIRCLE"])
        else:
            ops.append(["remove_add", int(i), int(j)])
    cases.append({"p": p, "edges": [[int(a), int(b)] for a, b in edges], "ops": ops})
json.dump(cases, open("reference/fixtures/java_orders_ops.json", "w"))
print(len(cases), "cases")
