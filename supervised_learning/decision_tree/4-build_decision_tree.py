#!/usr/bin/env python3
"""
Build decision tree classes.
"""

import numpy as np


class Node:
    """
    Node class for decision tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """
        Initialize node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None

    def update_bounds_below(self):
        """
        Compute bounds for children recursively.
        """

        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child, side in [
                (self.left_child, "left"),
                (self.right_child, "right")
        ]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if side == "left":
                    child.upper[self.feature] = self.threshold

                else:
                    child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()


class Leaf(Node):
    """
    Leaf class.
    """

    def __init__(self, value, depth=None):
        """
        Initialize leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """
        Update bounds below leaf.
        """
        pass


class Decision_Tree:
    """
    Decision Tree class.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random",
                 root=None):
        """
        Initialize decision tree.
        """

        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def update_bounds(self):
        """
        Update bounds of all nodes.
        """
        self.root.update_bounds_below()
