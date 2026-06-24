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

    def update_bounds_below(self):
        """
        Recursively computes the bounds for the node and all nodes below
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold
                if not child.is_leaf:
                    child.update_bounds_below()

    def update_indicator(self):
        """
        Computes the indicator function for the node
        """
        def is_large_enough(x):
            return np.all(
                np.array([x[:, key] > self.lower[key]
                          for key in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(
                np.array([x[:, key] <= self.upper[key]
                          for key in self.upper]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Returns the prediction for input x at this node
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def update_bounds_below(self):
        """
        Leaf has no children; nothing to update below
        """
        pass

    def pred(self, x):
        """
        Returns the leaf value as prediction
        """
        return self.value


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

    def update_bounds(self):
        """
        Computes bounds for all nodes in the tree
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Updates the predict function using leaf indicators
        """
        self.update_bounds()
        all_leaves = self.get_leaves()
        for leaf in all_leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array(
            [self.pred(x) for x in A])

    def pred(self, x):
        """
        Returns prediction for a single individual x
        """
        return self.root.pred(x)

    def get_leaves(self):
        """
        Returns all leaves of the tree
        """
        return self._get_leaves_below(self.root)

    def _get_leaves_below(self, node):
        """
        Recursively collects all leaves below a node
        """
        if node.is_leaf:
            return [node]
        leaves = []
        if node.left_child:
            leaves += self._get_leaves_below(node.left_child)
        if node.right_child:
            leaves += self._get_leaves_below(node.right_child)
        return leaves

    def np_extrema(self, arr):
        """
        Returns the min and max of an array
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and threshold to split a node
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the decision tree on the given data
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """
        Recursively fits a node by splitting or making it a leaf
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] <= node.threshold)
        )

        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node
        """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates and returns an internal child node
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Returns the accuracy of the model on the given data
        """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
