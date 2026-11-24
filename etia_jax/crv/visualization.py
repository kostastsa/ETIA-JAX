"""Graph drawings without Cytoscape (increment 10, ETIA ``CRV/visualization``).

ETIA's ``Visualization`` sends the graph to a running Cytoscape program
(py4cytoscape). ETIA-JAX writes an SVG file instead, with the same
information:

* ``matrix_to_edges`` is ETIA's ``matrix_to_cyto``: one row per edge with
  ``source``, ``target`` and ``interaction_type`` ("<mark at source>-<mark
  at target>", for example ``Tail-Arrow`` for source -> target);
* the end of each edge shows its mark: an arrowhead, a circle or nothing
  (tail), as ETIA's style;
* ``plot_edge_confidence`` maps a confidence to the width (0, 0.5, 0.7, 1
  -> 0.4, 1, 2, 3 px, linear between the points, as Cytoscape's continuous
  mapping) and to the opacity, and ``hide_edges`` hides the edges with a
  confidence below a threshold. In ETIA these two methods read the columns
  ``edge_confidence`` and ``confidence``, which the network does not have.
* ``set_layout("force-directed")``: the layout of Fruchterman and Reingold
  (1991), seeded; ``"circle"`` puts the nodes on a circle.
"""
import html

import numpy as np
import pandas as pd

_MARK = {1: "Circle", 2: "Arrow", 3: "Tail"}


def matrix_to_edges(M, names=None):
    """ETIA's matrix_to_cyto."""
    if isinstance(M, pd.DataFrame):
        names = M.columns.to_list()
        M = M.to_numpy()
    M = np.asarray(M)
    names = names if names is not None else [f"X{i}" for i in range(len(M))]
    rows = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if M[i, j] != 0 and M[j, i] != 0:
                if M[i, j] not in _MARK or M[j, i] not in _MARK:
                    raise ValueError("wrong notation on input matrix of the graph")
                rows.append([names[i], names[j], _MARK[int(M[j, i])] + "-" + _MARK[int(M[i, j])]])
    return pd.DataFrame(rows, columns=["source", "target", "interaction_type"])


def force_layout(M, seed=0, iters=300):
    """Fruchterman-Reingold layout in the unit square."""
    A = (np.asarray(M) != 0) | (np.asarray(M).T != 0)
    n = len(A)
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 1, (n, 2))
    k = np.sqrt(1.0 / max(n, 1))
    t = 0.1
    for _ in range(iters):
        D = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((D ** 2).sum(-1)) + 1e-9
        rep = (k * k / dist ** 2)[..., None] * D
        att = (A * dist / k)[..., None] * D
        disp = rep.sum(1) - att.sum(1)
        ln = np.sqrt((disp ** 2).sum(-1, keepdims=True)) + 1e-9
        P = P + disp / ln * np.minimum(ln, t)
        t *= 0.985
    P = P - P.min(0)
    return P / np.maximum(P.max(0), 1e-9)


def circle_layout(n):
    a = 2 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)], 1)


def _interp(x, xs, ys):
    return float(np.interp(x, xs, ys))


class Visualization:
    """ETIA's Visualization (same methods), drawn as SVG."""

    def __init__(self, matrix_pd, net_name="Graph", collection_name="Collection", names=None):
        if isinstance(matrix_pd, pd.DataFrame):
            names = matrix_pd.columns.to_list()
            M = matrix_pd.to_numpy()
        else:
            M = np.asarray(matrix_pd)
            names = names if names is not None else [f"X{i}" for i in range(len(M))]
        self.M, self.names = M, [str(n) for n in names]
        self.net_name, self.collection_name = net_name, collection_name
        self.node_size, self.node_shape, self.node_color = 35, "ellipse", "#FDD49E"
        self.colors, self.hidden_nodes, self.hidden_edges = {}, set(), set()
        self.confidence = None
        self.pos = force_layout(M)

    # ETIA's methods
    def plot_cytoscape(self):
        """No Cytoscape: returns the SVG text."""
        return self.to_svg()

    def create_visual_style(self, node_size=35, node_shape="ellipse", node_color="#FDD49E"):
        self.node_size, self.node_shape, self.node_color = node_size, node_shape, node_color

    def set_node_color(self, node_names, color):
        for n in node_names:
            self.colors[str(n)] = color

    def hide_nodes(self, nodes):
        self.hidden_nodes |= {str(n) for n in nodes}

    def plot_edge_confidence(self, edge_confidence, column="edge_consistency"):
        """``edge_confidence``: a DataFrame with source, target and the
        column (compute_edge_weights)."""
        self.confidence = {(str(r.source), str(r.target)): float(getattr(r, column))
                           for r in edge_confidence.itertuples()}

    def hide_edges(self, threshold):
        if self.confidence is None:
            raise ValueError("no edge confidence: call plot_edge_confidence first")
        self.hidden_edges = {k for k, v in self.confidence.items() if v < threshold}

    def set_layout(self, layout_name="force-directed", seed=0):
        if layout_name == "circle":
            self.pos = circle_layout(len(self.M))
        else:
            self.pos = force_layout(self.M, seed)

    def export_to_png(self, file_path="network.svg"):
        """Writes the SVG (the name keeps ETIA's method; give a .svg path)."""
        with open(file_path, "w") as fh:
            fh.write(self.to_svg())
        return file_path

    # drawing
    def to_svg(self, width=640, height=480, pad=40):
        P = self.pos * [width - 2 * pad, height - 2 * pad] + pad
        r = self.node_size / 2
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
               f'font-family="sans-serif" font-size="12">',
               '<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>'
               '<marker id="cir" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><circle cx="5" cy="5" r="4" fill="white" stroke="#333" '
               'stroke-width="1.5"/></marker></defs>']
        mk = {"Arrow": "url(#arr)", "Circle": "url(#cir)", "Tail": None}
        for e in matrix_to_edges(self.M, self.names).itertuples():
            if e.source in self.hidden_nodes or e.target in self.hidden_nodes or \
                    (e.source, e.target) in self.hidden_edges:
                continue
            i, j = self.names.index(e.source), self.names.index(e.target)
            a, b = P[i], P[j]
            d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            a2, b2 = a + d * r, b - d * r
            ms, mt = e.interaction_type.split("-")
            wdt, op = 1.5, 1.0
            if self.confidence is not None:
                c = self.confidence.get((e.source, e.target), 1.0)
                wdt = _interp(c, [0, 0.5, 0.7, 1.0], [0.4, 1, 2, 3])
                op = _interp(c, [0, 1], [150 / 255, 250 / 255])
            attrs = "".join(f' marker-{k}="{v}"' for k, v in (("start", mk[ms]), ("end", mk[mt])) if v)
            out.append(f'<line x1="{a2[0]:.1f}" y1="{a2[1]:.1f}" x2="{b2[0]:.1f}" y2="{b2[1]:.1f}" '
                       f'stroke="#333" stroke-width="{wdt:.2f}" stroke-opacity="{op:.2f}"{attrs}>'
                       f'<title>{html.escape(e.source)} {e.interaction_type} {html.escape(e.target)}</title></line>')
        for k, n in enumerate(self.names):
            if n in self.hidden_nodes:
                continue
            x, y = P[k]
            col = self.colors.get(n, self.node_color)
            out.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{r}" ry="{r * 0.7:.1f}" fill="{col}" stroke="#555"/>'
                       f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{html.escape(n)}</text>')
        out.append("</svg>")
        return "\n".join(out)
