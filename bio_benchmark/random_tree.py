import random 
from collections import defaultdict

class RandomTree:
    """
    Build a random unrooted binary tree with n_leaves numbered 1..n.
    Internal nodes numbered n+1, n+2, ...
    """
    def __init__(self, n_leaves):
        self.n_leaves = n_leaves
        self.adj = defaultdict(list)
        self.lengths = {}   # store edge lengths: key=(u,v) in sorted order

        # Build random unrooted binary tree via random split growth.
        # (This is not Yule or coalescent, just a simple binary random shape.)
        #
        # Start with a cherry: 1 -- (n+1) -- 2
        internal_id = n_leaves + 1

        # adjacency for starting structure
        self._add_edge(1, internal_id)
        self._add_edge(2, internal_id)

        # Internal nodes used so far
        current_internal = [internal_id]

        # Add leaf 3..n
        for leaf in range(3, n_leaves + 1):
            # Pick a random existing edge to subdivide
            # Choose a random internal or leaf, then one of its neighbors
            u = random.choice(list(self.adj.keys()))
            v = random.choice(self.adj[u])

            # Remove edge u--v
            self._remove_edge(u, v)

            # Create new internal node
            internal_id += 1
            w = internal_id

            # New edges u--w, v--w, w--leaf
            self._add_edge(u, w)
            self._add_edge(v, w)
            self._add_edge(w, leaf)

        # Randomize lengths
        for u in self.adj:
            for v in self.adj[u]:
                if (u,v) not in self.lengths and (v,u) not in self.lengths:
                    L = random.uniform(0.1, 1.0)
                    self.lengths[(u,v)] = self.lengths[(v,u)] = L

    def _add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def _remove_edge(self, u, v):
        self.adj[u].remove(v)
        self.adj[v].remove(u)

    def length(self, u, v):
        # symmetric
        return self.lengths.get((u, v), self.lengths.get((v, u)))

    def __str__(self):
        #Build newick representation
        def build_newick(node, parent):
            children = [n for n in self.adj[node] if n != parent]
            if not children:
                return str(node)
            else:
                subtrees = [build_newick(c, node) + f":{self.length(node,c):.4f}" for c in children]
                return "(" + ",".join(subtrees) + ")"
        return build_newick(1, None) + ";"