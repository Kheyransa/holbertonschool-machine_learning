#!/usr/bin/env python3
"""
Defines classes for building a decision tree and printing its structure
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes a Node instance
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth of nodes below this node recursively
        """
        if self.left_child is None and self.right_child is None:
            return self.depth

        left_depth = 0
        right_depth = 0

        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes or leaves below this node recursively
        """
        left_nodes = 0
        right_nodes = 0

        if self.left_child:
            left_nodes = self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            right_nodes = self.right_child.count_nodes_below(only_leaves)

        if only_leaves:
            return left_nodes + right_nodes
        return 1 + left_nodes + right_nodes

    def left_child_add_prefix(self, text):
        """
        Adds prefixes for left child's string representation
        """
        lines = text.split("\n")
        new_text = " +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += (" |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds prefixes for right child's string representation
        """
        lines = text.split("\n")
        new_text = " +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Returns string representation of the node and its children
        """
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            out += self.right_child_add_prefix(self.right_child.__str__())

        return out


class Leaf(Node):
    """
    Represents a leaf node in a decision tree
    """
    def __init__(self, value, depth=None):
        """
        Initializes a Leaf instance
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since a leaf is always counted as 1 node/leaf
        """
        return 1

    def __str__(self):
        """
        Returns string representation of a leaf node
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """
    Represents a decision tree classifier/regressor
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree instance
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

    def depth(self):
        """
        Returns the maximum depth of the entire tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the total number of nodes or leaves in the entire tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns string representation of the entire tree
        """
        return self.root.__str__().rstrip('\n')
