import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData

def binomial_devLoss(parameters ,X,y, lamb):
    n, d = X.shape
    w = parameters[:d]  # Extract the first d elements as w
    w0 = parameters[d]  # The last element is w0
    linear_combination = (X @ w + w0) * -y
    loss_terms = np.logaddexp(0, linear_combination).sum()
    reg_term = (lamb / 2) * np.linalg.norm(w) ** 2
    regized = loss_terms + reg_term
    return regized

def minBinDev(X, y, lamb):
    n, d = X.shape
    initial_parameters = np.zeros(d + 1)  # Initialize d weights + 1 intercept to 0
    
    #use scipy minimize
    res = minimize(binomial_devLoss,initial_parameters,args=(X,y,lamb),method='BFGS')
    # Extract the optimal values
    optimized_w = res.x  
    w = optimized_w[:d]     # First d elements are w
    w0 = optimized_w[d]     # Last element is w0
    return w, w0

def minHinge(X, y, lamb, stablizer=1e-5):
    n, d = X.shape

    # Ensure y is flattened to shape (n,)
    y = y.flatten()

    # Create the matrix P with the stabilizer for numerical stability
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = np.eye(d)  # L2 regularization term for w
    P = P + stablizer * np.eye(d + 1 + n)  # Add stabilizer to diagonal

    # Construct the q vector (linear term)
    q = np.hstack([np.zeros(d + 1), lamb * np.ones(n)])

    # Construct the G matrix (inequality constraints)
    G = np.zeros((2 * n, d + 1 + n))
    G[:n, :d] = -y[:, np.newaxis] * X  # -y * X
    G[:n, d] = -y  # -y * w0 (intercept term)
    G[:n, d + 1:] = -np.eye(n)  # Slack variables (negative identity matrix)
    G[n:, d + 1:] = -np.eye(n)  # Non-negative slack variables constraint

    # Construct the h vector (inequality bounds)
    h = np.hstack([-np.ones(n), np.zeros(n)])

    # Use cvxopt to solve the quadratic programming problem
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    # Extract the solution
    solution = np.array(sol['x']).flatten()
    w = solution[:d]  # First d elements are weights
    w0 = solution[d]  # Next element is intercept

    return w, w0

def classify(Xtest, w, w0):
    prediction = np.sign(Xtest @ w + w0)  
    return prediction

def accuracy(y_true, y_pred):
    """
    Compute the accuracy as the percentage of correctly predicted labels.
    """
    return np.mean(y_true.flatten() == y_pred.flatten())

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