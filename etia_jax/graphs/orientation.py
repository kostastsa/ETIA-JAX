"""Orientation rules on mark matrices (see ``encoding``).

* Meek rules R1-R4 for CPDAGs (Meek 1995).
* FCI rules R0-R4 and R8-R10 for PAGs without selection bias (Zhang 2008).

All functions change ``G`` in place and return True if a mark changed.
"""
import numpy as np
from .encoding import NONE, CIRCLE, ARROW, TAIL


def _adj(G, a, b):
    return G[a, b] != NONE


# ---------------------------------------------------------------------------
# Meek rules. Directed a->b: G[a,b]=ARROW, G[b,a]=TAIL. Undirected: TAIL/TAIL.
# ---------------------------------------------------------------------------

def _undirected(G, a, b):
    return G[a, b] == TAIL and G[b, a] == TAIL


def _directed(G, a, b):
    return G[a, b] == ARROW and G[b, a] == TAIL


def _orient(G, a, b):
    G[a, b] = ARROW
    G[b, a] = TAIL


def meek_rules(G, use_r4=False):
    """Apply Meek R1-R3 (and R4 if asked) until no rule fires."""
    n = len(G)
    any_change = False
    changed = True
    while changed:
        changed = False
        for a, b in zip(*np.nonzero((G == TAIL) & (G.T == TAIL))):
            if not _undirected(G, a, b):
                continue
            # R1: c -> a --- b, c and b nonadjacent  =>  a -> b
            pa = [c for c in range(n) if _directed(G, c, a) and c != b and not _adj(G, c, b)]
            if pa:
                _orient(G, a, b); changed = True; continue
            # R2: a -> c -> b and a --- b  =>  a -> b
            if any(_directed(G, a, c) and _directed(G, c, b) for c in range(n)):
                _orient(G, a, b); changed = True; continue
            # R3: a --- c1 -> b, a --- c2 -> b, c1, c2 nonadjacent  =>  a -> b
            cs = [c for c in range(n) if _undirected(G, a, c) and _directed(G, c, b)]
            if any(not _adj(G, c1, c2) for i, c1 in enumerate(cs) for c2 in cs[i + 1:]):
                _orient(G, a, b); changed = True; continue
            # R4: a --- c -> d -> b, a adj d (a --- c, c adj b) =>  a -> b
            if use_r4:
                hit = False
                for c in range(n):
                    if not (_undirected(G, a, c) and _adj(G, c, b)):
                        continue
                    for d in range(n):
                        if _directed(G, c, d) and _directed(G, d, b) and _adj(G, a, d):
                            hit = True; break
                    if hit:
                        break
                if hit:
                    _orient(G, a, b); changed = True; continue
        any_change |= changed
    return any_change


# ---------------------------------------------------------------------------
# FCI rules (Zhang 2008). Mark at j on edge i-j is G[i, j].
# ---------------------------------------------------------------------------

def unshielded_triples(G):
    """Yield (a, b, c) with a *-* b *-* c, a and c nonadjacent, a < c."""
    n = len(G)
    for b in range(n):
        nb = np.flatnonzero(G[:, b] != NONE)
        for i, a in enumerate(nb):
            for c in nb[i + 1:]:
                if G[a, c] == NONE:
                    yield int(a), int(b), int(c)


def fci_r1(G):
    """a *-> b o-* c, a, c nonadjacent  =>  b -> c."""
    n, ch = len(G), False
    for c, b in zip(*np.nonzero(G == CIRCLE)):  # G[c, b] == CIRCLE: b o-* c
        if G[c, b] != CIRCLE:
            continue
        if any(G[a, b] == ARROW and a != c and G[a, c] == NONE for a in range(n)):
            G[b, c], G[c, b] = ARROW, TAIL
            ch = True
    return ch


def fci_r2(G):
    """a -> b *-> c or a *-> b -> c, and a *-o c  =>  a *-> c."""
    n, ch = len(G), False
    for a, c in zip(*np.nonzero(G == CIRCLE)):
        if G[a, c] != CIRCLE:
            continue
        for b in range(n):
            if G[a, b] == ARROW and G[b, c] == ARROW and (G[b, a] == TAIL or G[c, b] == TAIL):
                G[a, c] = ARROW
                ch = True
                break
    return ch


def fci_r3(G):
    """a *-> b <-* c, a *-o d o-* c, a, c nonadjacent, d *-o b  =>  d *-> b."""
    n, ch = len(G), False
    for d, b in zip(*np.nonzero(G == CIRCLE)):
        if G[d, b] != CIRCLE:
            continue
        cand = [a for a in range(n) if a not in (b, d) and G[a, b] == ARROW and G[a, d] == CIRCLE]
        if any(G[a, c] == NONE for i, a in enumerate(cand) for c in cand[i + 1:]):
            G[d, b] = ARROW
            ch = True
    return ch


def find_discriminating_path(G, a, b, c, max_len=-1):
    """Return the end node ``d`` of a discriminating path <d, ..., a, b, c> for b.

    Conditions checked by the caller: a adjacent to b, b *-> a, and a -> c.
    Each node between d and b is a collider on the path and a parent of c.
    Breadth-first search, as in Tetrad.
    """
    n = len(G)
    prev = {a: b}
    queue = [a]
    dist = {a: 1}
    while queue:
        v = queue.pop(0)
        for e in range(n):
            if e in (b, c) or e in prev or G[e, v] != ARROW:
                continue
            # v must be a collider on the path: e *-> v <-* prev[v]
            if G[prev[v], v] != ARROW:
                continue
            if max_len > 0 and dist[v] + 1 > max_len:
                continue
            if G[e, c] == NONE:
                return e
            if G[e, c] == ARROW and G[c, e] == TAIL:  # e -> c
                prev[e] = v
                dist[e] = dist[v] + 1
                queue.append(e)
    return None


def fci_r4(G, is_noncollider, max_len=-1):
    """Discriminating path rule.

    ``is_noncollider(d, a, b, c)`` returns True if ``b`` is in the separating
    set of (d, c) (or (a, b, c) is a non-collider in the true MAG).
    If so: b o-* c becomes b -> c. Otherwise a <-> b <-> c.
    """
    n, ch = len(G), False
    for c, b in zip(*np.nonzero(G == CIRCLE)):  # b o-* c
        if G[c, b] != CIRCLE:
            continue
        for a in range(n):
            if a in (b, c) or G[a, b] == NONE or not (G[a, c] == ARROW and G[c, a] == TAIL):
                continue
            if G[b, a] != ARROW:  # a must be a collider on the path: b *-> a
                continue
            d = find_discriminating_path(G, a, b, c, max_len)
            if d is None:
                continue
            if is_noncollider(d, a, b, c):
                G[b, c], G[c, b] = ARROW, TAIL
            else:
                G[a, b] = G[b, a] = ARROW
                G[c, b] = G[b, c] = ARROW
            ch = True
            break
    return ch


def _pd_edge(G, u, v):
    """Edge u *-* v is potentially directed from u to v."""
    return G[u, v] != NONE and G[u, v] != TAIL and G[v, u] != ARROW


def _uncovered_pd_path_exists(G, start, first, target, forbidden=()):
    """Exists an uncovered potentially directed path start, first, ..., target.

    The edge start-first is assumed checked. ``forbidden`` nodes are excluded.
    """
    if first == target:
        return True
    n = len(G)
    stack = [(start, first, frozenset((start, first)))]
    seen = set()
    while stack:
        u, v, path = stack.pop()
        if (u, v) in seen:
            continue
        seen.add((u, v))
        for w in range(n):
            if w in path or w in forbidden:
                continue
            if not _pd_edge(G, v, w) or G[u, w] != NONE:  # uncovered: u, w nonadjacent
                continue
            if w == target:
                return True
            stack.append((v, w, path | {w}))
    return False


def fci_r8(G):
    """a -> b -> c or a -o b -> c, and a o-> c  =>  a -> c."""
    n, ch = len(G), False
    for a, c in zip(*np.nonzero((G == ARROW) & (G.T == CIRCLE))):
        if not (G[a, c] == ARROW and G[c, a] == CIRCLE):
            continue
        for b in range(n):
            if G[b, c] == ARROW and G[c, b] == TAIL and G[b, a] == TAIL and G[a, b] in (ARROW, CIRCLE):
                G[c, a] = TAIL
                ch = True
                break
    return ch


def fci_r9(G):
    """a o-> c and an uncovered p.d. path <a, b, ..., c>, b and c nonadjacent  =>  a -> c."""
    n, ch = len(G), False
    for a, c in zip(*np.nonzero((G == ARROW) & (G.T == CIRCLE))):
        if not (G[a, c] == ARROW and G[c, a] == CIRCLE):
            continue
        for b in range(n):
            if b in (a, c) or G[b, c] != NONE or not _pd_edge(G, a, b):
                continue
            if _uncovered_pd_path_exists(G, a, b, c):
                G[c, a] = TAIL
                ch = True
                break
    return ch


def fci_r10(G):
    """a o-> c, b -> c <- d, uncovered p.d. paths p1: a..b and p2: a..d whose
    second nodes mu, omega are distinct and nonadjacent  =>  a -> c."""
    n, ch = len(G), False
    for a, c in zip(*np.nonzero((G == ARROW) & (G.T == CIRCLE))):
        if not (G[a, c] == ARROW and G[c, a] == CIRCLE):
            continue
        parents = [b for b in range(n) if b != a and G[b, c] == ARROW and G[c, b] == TAIL]
        if len(parents) < 2:
            continue
        firsts = {}
        for b in parents:
            firsts[b] = [m for m in range(n) if m not in (a, c) and _pd_edge(G, a, m)
                         and _uncovered_pd_path_exists(G, a, m, b, forbidden=(c,))]
        done = False
        for i, b in enumerate(parents):
            for d in parents[i + 1:]:
                for mu in firsts[b]:
                    for om in firsts[d]:
                        if mu != om and G[mu, om] == NONE:
                            done = True; break
                    if done: break
                if done: break
            if done: break
        if done:
            G[c, a] = TAIL
            ch = True
    return ch


def fci_orient(G, is_noncollider, max_disc_len=-1, complete=True):
    """Apply R1-R4 to a fixed point, then (if ``complete``) R8-R10 with R1-R4."""
    while True:
        ch = fci_r1(G)
        ch |= fci_r2(G)
        ch |= fci_r3(G)
        ch |= fci_r4(G, is_noncollider, max_disc_len)
        if complete:
            ch |= fci_r8(G)
            ch |= fci_r9(G)
            ch |= fci_r10(G)
        if not ch:
            return G
