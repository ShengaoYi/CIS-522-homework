import numpy as np


class LinearRegression:
    """
    The basic linear regression model using closed-form formula.
    """

    w: np.ndarray
    b: float

    def __init__(self) -> None:
        # Initializing the parameters
        self.w = np.array([])
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """

        Using closed-form formula to fit a linear regression model

        Args:
            X: np.ndarray, input data
            y: Target values

        Returns: None

        """

        # Appending a column of ones in X to add the bias term.
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Compute the transpose of X
        X_T = X.T

        # Compute X^T X
        XT_X = X.T @ X

        # Compute the inverse of X^T X
        inv_XT_X = np.linalg.inv(XT_X)

        # Compute the dot product of (X^T X)^-1 and X^T
        self.w = inv_XT_X @ (X_T @ y)
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """

        Predict the output for the given input.

        Args:
            X: Input data

        Returns: np.ndarray, output

        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """

        Args:
            X: np.ndarray, The input data
            y: Target values
            lr: learning rate for gradient descent.
            epochs: number of epochs

        Returns: None

        """

        m, n = X.shape

        self.w = np.zeros(n)

        self.b = 0.0

        # Implementing Gradient Descent
        for i in range(epochs):
            y_hat = X @ self.w + self.b

            """
                    We use Mean Squared Error Equation as our loss function,
                    then calculate the partial derivative of the loss function.
                    J(w) = 1/2m * (y_hat - y) ** 2, 1/2 is for simplification

                    dJ/dy_hat = 1/m * (y_hat - y) and dy_hat/dw = X,
                    so dJ/dw = dJ/dy_hat * dy_hat/dw = 1/m * (y_hat - y) @ X
            """
            # derived_w = (1 / m) * (X.T @ (y_hat - y))
            derived_b = (1 / m) * np.sum(y_hat - y)
            derived_w = 1 / m * (y_hat - y) @ X

            # Updating parameters
            self.w -= lr * derived_w
            self.b -= lr * derived_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
