import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from A2helpers import generateData

#Question 1a
def minBinDev(X, y, lamb):
    """
    Minimizes the regularized binomial deviance loss.

    Parameters:
    - X: An n x d input matrix.
    - y: An n x 1 target/label vector.
    - lamb: Regularization hyper-parameter (lambda > 0).

    Returns:
    - w: A d x 1 vector of weights.
    - w0: A scalar intercept.
    """
    n, d = X.shape

    # Initial guess for w and w0
    w0_init = 0.0
    w_init = np.zeros(d)
    params_init = np.concatenate(([w0_init], w_init))

    # Define the loss function
    def loss(params, X, y, lamb):
        w0 = params[0]
        w = params[1:]
        y = y.flatten()  # Ensure y is a 1D array
        z = - y * (X @ w + w0)

        # Compute log(1 + exp(z)) in a numerically stable way
        loss_terms = np.logaddexp(0, z)
        loss_value = np.sum(loss_terms) + (lamb / 2) * np.sum(w ** 2)
        return loss_value

    # Optimize using BFGS algorithm
    res = minimize(loss, params_init, args=(X, y, lamb), method='BFGS')

    # Check if the optimization was successful
    if not res.success:
        raise ValueError("Optimization did not converge: " + res.message)

    # Extract w0 and w from the result
    w0_opt = res.x[0]
    w_opt = res.x[1:].reshape(-1, 1)

    return w_opt, w0_opt

#Question 1b
def minHinge(X, y, lamb, stabilizer=1e-5):
    """
    Solves the regularized hinge loss problem using quadratic programming.

    Parameters:
    - X: An n x d input matrix.
    - y: An n x 1 target/label vector.
    - lamb: Regularization hyper-parameter (lambda > 0).
    - stabilizer: A small positive scalar for numerical stability.

    Returns:
    - w: A d x 1 vector of weights.
    - w0: A scalar intercept.
    """
    n, d = X.shape

    # Create the required matrices for quadratic programming
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = np.eye(d) * lamb  # Regularization term
    P += stabilizer * np.eye(d + 1 + n)  # Stabilizer for numerical stability

    q = np.zeros(d + 1 + n)
    q[d + 1:] = 1  # Coefficients for slack variables

    G = np.zeros((2 * n, d + 1 + n))
    G[:n, :d] = -X * y  # -y_i * X_i
    G[:n, d] = -y.flatten()  # -y_i * w0
    G[:n, d + 1:] = -np.eye(n)  # -ξ_i
    G[n:, d + 1:] = -np.eye(n)  # ξ_i >= 0

    h = np.zeros(2 * n)
    h[:n] = -1  # Corresponding to max(0, 1 - y_i * (X_i w + w0))

    # Convert numpy arrays to cvxopt matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    # Solve the quadratic program
    sol = solvers.qp(P, q, G, h)

    # Extract solution
    params = np.array(sol['x']).flatten()
    w = params[:d].reshape(-1, 1)  # d x 1 weight vector
    w0 = params[d]  # Scalar intercept

    return w, w0

#Question 1c
def classify(Xtest, w, w0):
    # Compute the decision values: Xtest @ w + w0
    print(f"Weights: {w}")
    print(f"Intercept (w0): {w0}")

    decision_values = Xtest @ w + w0
    print(f"Decision Values: {decision_values}")

    # Apply np.sign to get predictions (-1 or 1)
    prediction = np.sign(decision_values)
    
    # Ensure the output is of shape (m, 1) instead of (m,)
    print(prediction)
    
    print(f"Unique Predictions: {np.unique(prediction)}")  # Debugging output
    return prediction

def accuracy(y_true, y_pred):
    """
    Compute the accuracy as the percentage of correctly predicted labels.
    """
    return np.mean(y_true.flatten() == y_pred.flatten())

#question 1d
def synExperimentsRegularize():
    n_runs = 100  # Number of runs
    n_train = 100  # Training samples
    n_test = 1000  # Test samples

    lamb_list = [0.001, 0.01, 0.1, 1.0]  # Regularization parameters
    gen_model_list = [1, 2, 3]  # Different data generation models

    # Initialize matrices to store accuracies
    train_acc_bindev = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_bindev = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    train_acc_hinge = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_hinge = np.zeros((len(lamb_list), len(gen_model_list), n_runs))

    # Run experiments
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                # Generate training and test data
                Xtrain, ytrain = generateData(n_train, gen_model)
                Xtest, ytest = generateData(n_test, gen_model)

                # Train and evaluate using binomial deviance loss
                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)
                ytest_pred = classify(Xtest, w, w0)
                train_acc_bindev[i, j, r] = accuracy(ytrain, ytrain_pred)
                test_acc_bindev[i, j, r] = accuracy(ytest, ytest_pred)

                # Train and evaluate using hinge loss
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)
                ytest_pred = classify(Xtest, w, w0)
                train_acc_hinge[i, j, r] = accuracy(ytrain, ytrain_pred)
                test_acc_hinge[i, j, r] = accuracy(ytest, ytest_pred)

    # Compute average accuracies over all runs
    avg_train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    avg_test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    # Combine results into 4x6 matrices (bindev + hinge)
    train_acc = np.hstack([avg_train_acc_bindev, avg_train_acc_hinge])
    test_acc = np.hstack([avg_test_acc_bindev, avg_test_acc_hinge])

    return train_acc, test_acc

if __name__ == "__main__":
    train_acc, test_acc = synExperimentsRegularize()
    print("Train Accuracy Matrix:\n", train_acc)
    print("Test Accuracy Matrix:\n", test_acc)