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

    def __str__(self):
        """
        Return string representation of node.
        """
        if self.is_root:
            text = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold)
        else:
            text = "node [feature={}, threshold={}]".format(
                self.feature, self.threshold)

        if self.left_child is not None:
            text += "\n" + self.left_child_add_prefix(
                str(self.left_child))

        if self.right_child is not None:
            text += "\n" + self.right_child_add_prefix(
                str(self.right_child))

        return text

    def left_child_add_prefix(self, text):
        """
        Add prefix for left child.
        """
        lines = text.split("\n")
        new_text = "+---> " + lines[0]

        for line in lines[1:]:
            new_text += "\n| " + line

        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix for right child.
        """
        lines = text.split("\n")
        new_text = "+---> " + lines[0]

        for line in lines[1:]:
            new_text += "\n" + line

        return new_text

    def get_leaves_below(self):
        """
        Return all leaves below this node.
        """
        leaves = []

        if self.left_child is not None:
            leaves += self.left_child.get_leaves_below()

        if self.right_child is not None:
            leaves += self.right_child.get_leaves_below()

        return leaves


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

    def __str__(self):
        """
        Return string representation of leaf.
        """
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """
        Return all leaves below this leaf.
        """
        return [self]


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

    def __str__(self):
        """
        Return string representation of tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Return all leaves of the tree.
        """
        return self.root.get_leaves_below()
