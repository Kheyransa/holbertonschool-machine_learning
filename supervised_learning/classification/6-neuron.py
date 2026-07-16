def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Args:
            X (numpy.ndarray): shape (nx, m), contains the input data
            Y (numpy.ndarray): shape (1, m), correct labels
            iterations (int): number of iterations to train over
            alpha (float): the learning rate

        Raises:
            TypeError: if iterations is not an integer
            ValueError: if iterations is not positive
            TypeError: if alpha is not a float
            ValueError: if alpha is not positive

        Returns:
            The evaluation of the training data after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
