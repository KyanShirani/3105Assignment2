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

#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """
    Solves the dual form of the SVM (regularized hinge loss) using quadratic programming.

    Parameters:
    - X: n x d input matrix
    - y: n x 1 target/label vector (-1 or 1)
    - lamb: regularization hyper-parameter (λ > 0)
    - kernel_func: callable kernel function
    - stabilizer: small positive scalar to ensure numerical stability (default: 1e-5)

    Returns:
    - a: n x 1 vector of weights/parameters α
    - b: scalar intercept b
    """
    n = X.shape[0]
    y = y.flatten()  # Ensure y is a 1-D array of length n

    # Compute the kernel matrix K
    K = kernel_func(X, X)  # K is an n x n matrix

    # Compute the matrix Q = (1/(λ)) * Δ(y) K Δ(y)
    # Δ(y) is a diagonal matrix with y_i on the diagonal
    Y = np.diag(y)
    Q = (1 / lamb) * (Y @ K @ Y)

    # Add stabilizer to the diagonal for numerical stability
    Q = Q + stabilizer * np.eye(n)

    # Convert Q to cvxopt matrix (must be positive semidefinite)
    P = cvxopt.matrix(Q)

    # The linear term in the objective function: -1^T α
    q = cvxopt.matrix(-np.ones(n))

    # Inequality constraints: 0 ≤ α ≤ 1
    G_std = np.vstack((-np.eye(n), np.eye(n)))  # Stacked: -I_n and I_n
    h_std = np.hstack((np.zeros(n), np.ones(n)))  # Stacked: zeros and ones

    G = cvxopt.matrix(G_std)
    h = cvxopt.matrix(h_std)

    # Equality constraint: α^T y = 0
    A = cvxopt.matrix(y.reshape(1, -1))  # Make y a row vector
    b_eq = cvxopt.matrix(0.0)

    # Suppress solver output
    solvers.options['show_progress'] = False

    # Solve the quadratic program
    solution = solvers.qp(P, q, G, h, A, b_eq)

    # Extract the solution
    alpha = np.array(solution['x']).flatten()

    # Compute the intercept b
    # Find indices where 0 < α_i < 1
    idx = np.where((alpha > 1e-5) & (alpha < 1 - 1e-5))[0]

    if len(idx) > 0:
        # Use the first valid index
        i = idx[0]
        k_i = K[i, :]
        b = y[i] - (1 / lamb) * (k_i @ Y @ alpha)
    else:
        # If no α_i satisfies 0 < α_i < 1, use the average over all support vectors
        support_vectors = np.where(alpha > 1e-5)[0]
        if len(support_vectors) == 0:
            # If no support vectors, set b to zero (fallback)
            b = 0.0
        else:
            b_values = []
            for i in support_vectors:
                k_i = K[i, :]
                b_i = y[i] - (1 / lamb) * (k_i @ Y @ alpha)
                b_values.append(b_i)
            b = np.mean(b_values)

    # Reshape alpha to be n x 1 vector
    alpha = alpha.reshape(-1, 1)

    return alpha, b


#3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    """
    Classify test data using the dual SVM model parameters.

    Parameters:
    - Xtest: m x d numpy array of test data points to predict on.
    - a: n x 1 numpy array of dual weights/parameters alpha obtained from training.
    - b: scalar intercept obtained from training.
    - X: n x d numpy array of training data used to obtain 'a' and 'b'.
    - y: n x 1 numpy array of training labels (-1 or 1) used during training.
    - lamb: scalar regularization hyper-parameter used during training.
    - kernel_func: callable kernel function used during training.

    Returns:
    - yhat: m x 1 numpy array of predicted labels (-1 or 1) for the test data.
    """
    # Ensure y is a 1-D array
    y = y.flatten()
    
    # Compute the kernel matrix between test data and training data
    K_test = kernel_func(Xtest, X)  # K_test is an m x n matrix

    # Compute the decision function values for the test data
    # Multiply a by y to get Δ(y) α
    alpha_y = a.flatten() * y  # α_i * y_i for each training sample

    # Compute s = (1/λ) * K_test * (Δ(y) α) + b
    s = (1 / lamb) * (K_test @ alpha_y) + b  # s is an m x 1 vector

    # Compute the predictions by taking the sign of the decision function
    yhat = np.sign(s)

    # Ensure yhat is an m x 1 column vector
    yhat = yhat.reshape(-1, 1)

    return yhat

if __name__ == "__main__":
    train_acc, test_acc = synExperimentsRegularize()
    print("Train Accuracy Matrix:\n", train_acc)
    print("Test Accuracy Matrix:\n", test_acc)