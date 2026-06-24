#!/usr/bin/env python3
"""
Defines classes for building, printing, and training a decision tree
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
        self.lower = None
        self.upper = None
        self.indicator = None

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
            out = (f"node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            out += self.right_child_add_prefix(self.right_child.__str__())

        return out

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this node recursively
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively computes the upper and lower bounds for each feature
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Computes the indicator function from lower and upper bounds
        """
        def is_large_enough(x):
            """Checks if values are greater than lower bounds"""
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower]), axis=0)

        def is_small_enough(x):
            """Checks if values are less than or equal to upper bounds"""
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """
        Recursively determines prediction for a single individual
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

    def get_leaves_below(self):
        """
        Returns a list containing this leaf node itself
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaf nodes stop the recursion, boundaries are already set
        """
        pass

    def pred(self, x):
        """
        Returns the value of the leaf as the prediction
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

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the boundaries of all nodes in the tree starting from root
        """
        self.root.update_bounds_below()

    def update_indicators(self):
        """
        Updates the indicator function for all nodes in the tree
        """
        self.root.update_indicator()

    def update_predict(self):
        """
        Computes the vectorized prediction function for the tree
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.value * leaf.indicator(A) for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        """
        Predicts the class/value for a single individual
        """
        return self.root.pred(x)

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum of an array
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random split point based on node subpopulation
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Trains the decision tree on the provided dataset
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
            print("Training finished.")
            print(f"Depth : {self.depth()}")
            print(f"Number of nodes : {self.count_nodes()}")
            print(f"Number of leaves : {self.count_nodes(only_leaves=True)}")
            print(f"Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}\n")

    def fit_node(self, node):
        """
        Recursively fits nodes and builds the tree hierarchy
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & ~(
            self.explanatory[:, node.feature] > node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            len(np.unique(self.target[left_population])) == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            len(np.unique(self.target[right_population])) == 1
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
        value = np.argmax(np.bincount(self.target[sub_population]))
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
        Calculates the accuracy score of the tree predictions
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
