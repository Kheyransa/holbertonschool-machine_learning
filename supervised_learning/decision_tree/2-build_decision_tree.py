#!/usr/bin/env python3

import numpy as np


class Node:

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        if self.is_root:
            result = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )
        else:
            result = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )

        if self.left_child is not None:
            result += self.left_child_add_prefix(str(self.left_child))

        if self.right_child is not None:
            result += self.right_child_add_prefix(str(self.right_child))

        return result

    def left_child_add_prefix(self, text):
        lines = text.rstrip("\n").split("\n")
        result = "    +--" + lines[0] + "\n"

        for line in lines[1:]:
            result += "    |  " + line + "\n"

        return result

    def right_child_add_prefix(self, text):
        lines = text.rstrip("\n").split("\n")
        result = "    +--" + lines[0] + "\n"

        for line in lines[1:]:
            result += "       " + line + "\n"

        return result


class Leaf(Node):

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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

    def __str__(self):
        return self.root.__str__()
