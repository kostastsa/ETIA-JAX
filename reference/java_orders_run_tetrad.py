import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / "PROJECTS" / "ETIA"))
from make_fixtures import start_jvm
jp = start_jvm()
GN = jp.JClass("edu.cmu.tetrad.graph.GraphNode"); ELG = jp.JClass("edu.cmu.tetrad.graph.EdgeListGraph"); EP = jp.JClass("edu.cmu.tetrad.graph.Endpoint")
AL = jp.JClass("java.util.ArrayList")
out = []
for case in json.load(open("reference/fixtures/java_orders_ops.json")):
    p = case["p"]; nodes = [GN(f"X{i+1}") for i in range(p)]; idx = {f"X{i+1}": i for i in range(p)}
    lst = AL()
    for n in nodes: lst.add(n)
    g = ELG(lst)
    for a, b in case["edges"]: g.addUndirectedEdge(nodes[a], nodes[b])
    for op in case["ops"]:
        if op[0] == "set": g.setEndpoint(nodes[op[1]], nodes[op[2]], getattr(EP, op[3]))
        elif op[0] == "reorient": g.reorientAllWith(EP.CIRCLE)
        else:
            e = g.getEdge(nodes[op[1]], nodes[op[2]]); g.removeEdge(e); g.addEdge(e)
    nm = lambda n: idx[str(n.getName())]
    res = {"edges": [[nm(e.getNode1()), nm(e.getNode2()), str(e.getEndpoint1()), str(e.getEndpoint2())] for e in g.getEdges()],
           "adj": [[nm(x) for x in g.getAdjacentNodes(n)] for n in nodes],
           "parents": [[nm(x) for x in g.getParents(n)] for n in nodes],
           "into": [[nm(x) for x in g.getNodesInTo(n, EP.ARROW)] for n in nodes]}
    out.append(res)
json.dump(out, open("reference/fixtures/java_orders_tetrad.json", "w")); print("ok")
