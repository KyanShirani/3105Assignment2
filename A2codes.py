import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData, polyKernel, linearKernel, gaussKernel


#Question 1a
def minBinDev(X, y, lamb):

    #First we get our shape snd w,w0 are intialized 
    n, d = X.shape
    w0_init = 0.0
    w_init = np.zeros(d)
    params_init = np.concatenate(([w0_init], w_init))

    # Then we have to define the loss function which in our case is the binomial deviance loss we do it in the function so it solves our orginal problem of passing the param
    def loss(params, X, y, lamb):
        w0 = params[0]
        w = params[1:]
        y = y.flatten() 
        z = - y * (X @ w + w0)
        loss_terms = np.logaddexp(0, z)
        loss_value = np.sum(loss_terms) + (lamb / 2) * np.sum(w ** 2)
        return loss_value

    # we use scipys optimize to minimze
    res = minimize(loss, params_init, args=(X, y, lamb), method='BFGS')

    # Check if the optimization was successful to check for the optimization issues we were having 
    if not res.success:
        raise ValueError("Optimization did not converge: " + res.message)

    # Extract w0 and w from the result
    w0 = res.x[0]
    w = res.x[1:].reshape(-1, 1)
    return w, w0

#Question 1b
def minHinge(X, y, lamb, stabilizer=1e-5):

    #Same process as last time get our shape 
    n, d = X.shape

    # Create the required matrices for quadratic programming first we create P matrix
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = np.eye(d) * lamb  
    P += stabilizer * np.eye(d + 1 + n)  

    #Then we make our linear term or q
    q = np.zeros(d + 1 + n)
    q[d + 1:] = 1  

    #G matrix 
    G = np.zeros((2 * n, d + 1 + n))
    G[:n, :d] = -X * y  
    G[:n, d] = -y.flatten()  
    G[:n, d + 1:] = -np.eye(n) 
    G[n:, d + 1:] = -np.eye(n)  

    #h matrix 
    h = np.zeros(2 * n)
    h[:n] = -1 

    # Convert numpy arrays to cvxopt matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    # Solve the quadratic program
    sol = solvers.qp(P, q, G, h)

    # Then we just get out solutions 
    params = np.array(sol['x']).flatten()
    w = params[:d].reshape(-1, 1)  
    w0 = params[d]  

    return w, w0

#Question 1c
def classify(Xtest, w, w0):
    # Compute the resulting val Xtest @ w + w0 and apply sign activation function
    result = Xtest @ w + w0
    prediction = np.sign(result)
    return prediction

#We need accuracy function for 1d it wasnt in helpers 
def accuracy(y_true, y_pred):
    return np.mean(y_true.flatten() == y_pred.flatten())

#question 1d
def synExperimentsRegularize():
    n_runs = 100 
    n_train = 100  
    n_test = 1000  
    lamb_list = [0.001, 0.01, 0.1, 1.0]  
    gen_model_list = [1, 2, 3]  
    train_acc_bindev = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_bindev = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    train_acc_hinge = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    test_acc_hinge = np.zeros((len(lamb_list), len(gen_model_list), n_runs))
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                # Generate training and test data
                Xtrain, ytrain = generateData(n_train, gen_model)
                Xtest, ytest = generateData(n_test, gen_model)

                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)
                ytest_pred = classify(Xtest, w, w0)
                train_acc_bindev[i, j, r] = accuracy(ytrain, ytrain_pred)
                test_acc_bindev[i, j, r] = accuracy(ytest, ytest_pred)

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


    # combine accuracies (bindev and hinge)
    train_acc = np.hstack([avg_train_acc_bindev, avg_train_acc_hinge])
    test_acc = np.hstack([avg_test_acc_bindev, avg_test_acc_hinge])

    # return 4-by-6 train accuracy and 4-by-6 test accuracy
    return train_acc, test_acc


#2a
def adjBinDev(X, y, lamb, kernel_func):
    n = X.shape[0]
    K = kernel_func(X, X)  # Compute the kernel matrix
    y = y.flatten()  # Ensure y is a 1-D array of length n

    def objective(params):
        alpha = params[0:n]
        alpha0 = params[n]
        s = K @ alpha + alpha0  # Compute s = Kα + α0
        y_s = y * s  # Element-wise multiplication
        term1 = np.sum(np.log(1 + np.exp(-y_s)))
        term2 = (lamb / 2) * alpha.T @ K @ alpha
        L = term1 + term2
        return L

    def gradient(params):
        alpha = params[0:n]
        alpha0 = params[n]
        s = K @ alpha + alpha0
        y_s = y * s
        p = 1 / (1 + np.exp(y_s))  # Compute p_i = 1 / (1 + exp(y_i * s_i))
        gradient_s = -y * p  # Gradient with respect to s
        gradient_alpha = K @ gradient_s + lamb * K @ alpha  # Gradient w.r.t α
        gradient_alpha0 = np.sum(gradient_s)  # Gradient w.r.t α0
        grad = np.concatenate([gradient_alpha, [gradient_alpha0]])
        return grad

    # Initial guess for α and α0
    initial_params = np.zeros(n + 1)

    # Perform the optimization
    res = minimize(objective, initial_params, method='L-BFGS-B', jac=gradient)

    params_opt = res.x
    alpha_opt = params_opt[0:n]
    alpha0_opt = params_opt[n]

    return alpha_opt[:, np.newaxis], alpha0_opt


#2b
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    y = y.flatten()

    # Compute the kernel matrix K
    K = kernel_func(X, X)
    
    # Set up the quadratic programming problem
    # Variables: x = [alpha; alpha0; xi] ∈ R^{2n + 1}
    # Size of x: (n) + (1) + (n) = 2n + 1
    P = np.zeros((2*n + 1, 2*n + 1))
    # Top-left block: lambda * K + stabilizer * I_n
    P[:n, :n] = lamb * K + stabilizer * np.eye(n)
    # Add stabilizer to the diagonal elements for numerical stability
    P = P + stabilizer * np.eye(2*n + 1)
    
    # q vector: [zeros(n + 1); ones(n)]
    q = np.zeros((2*n + 1))
    q[n+1:] = 1.0  # Coefficient for xi in the objective function
    
    # Inequality constraints G x ≤ h
    # First constraint: -xi ≤ 0  =>  -I_n * xi ≤ 0
    G1 = np.zeros((n, 2*n + 1))
    G1[:, n+1:] = -np.eye(n)
    h1 = np.zeros(n)
    
    # Second constraint: -diag(y) * (K * alpha + alpha0 * 1_n) - xi ≤ -1
    G2 = np.zeros((n, 2*n + 1))
    # -diag(y) * K
    G2[:, :n] = -np.diag(y) @ K
    # -diag(y) * alpha0 * 1_n
    G2[:, n] = -y
    # -xi
    G2[:, n+1:] = -np.eye(n)
    h2 = -np.ones(n)
    
    # Combine G and h
    G = np.vstack((G1, G2))
    h = np.hstack((h1, h2))
    
    # Convert numpy arrays to cvxopt matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    
    # Solve the quadratic program
    solvers.options['show_progress'] = False  # Suppress output
    solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    
    x_opt = np.array(solution['x']).flatten()
    
    # Extract alpha, alpha0, and xi from x_opt
    alpha_opt = x_opt[:n]
    alpha0_opt = x_opt[n]
    # xi_opt = x_opt[n+1:]  # Not needed for this function
    
    return alpha_opt[:, np.newaxis], alpha0_opt

#2c
def adjClassify(Xtest, a, a0, X, kernel_func):

    # Compute the kernel between test data and training data
    K_test = kernel_func(Xtest, X)  # K_test is an m x n matrix

    # Compute the decision function values
    s = K_test @ a + a0  # s is an m x 1 vector

    # Compute the predictions by taking the sign
    yhat = np.sign(s)

    return yhat


"""
#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    y = y.flatten()  # Ensure y is a 1-D array of length n

    # Compute the kernel matrix K
    K = kernel_func(X, X)

    # Compute Q = (1/λ) * Δ(y) K Δ(y)
    Y = np.diag(y)
    Q = (1 / lamb) * (Y @ K @ Y)

    # Add stabilizer to the diagonal for numerical stability
    Q += stabilizer * np.eye(n)

    # Convert Q to cvxopt matrix
    P = matrix(Q)
    q = matrix(-np.ones(n))

    # Inequality constraints: 0 ≤ α ≤ 1
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n)]))

    # Equality constraint: α^T y = 0
    A = matrix(y, (1, n), 'd')  # Reshape y properly
    b_eq = matrix(0.0)

    # Suppress solver output
    solvers.options['show_progress'] = False

    # Solve the quadratic program
    solution = solvers.qp(P, q, G, h, A, b_eq)

    # Extract solution
    alpha = np.array(solution['x']).flatten()

    # Compute intercept b
    idx = np.where((alpha > 1e-5) & (alpha < 1 - 1e-5))[0]

    if len(idx) > 0:
        i = idx[0]
        k_i = K[i, :]
        b = y[i] - (1 / lamb) * (k_i @ Y @ alpha)
    else:
        support_vectors = np.where(alpha > 1e-5)[0]
        if len(support_vectors) == 0:
            b = 0.0
        else:
            b_values = [y[i] - (1 / lamb) * (K[i, :] @ Y @ alpha) for i in support_vectors]
            b = np.mean(b_values)

    alpha = alpha.reshape(-1, 1)
    return alpha, b



#3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):

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

def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    # Load the dataset
    train_data = pd.read_csv(os.path.join(dataset_folder, 'A2train.csv'), header=None).to_numpy()
    X = train_data[:, 1:] / 255.  # Normalize pixel values
    y = train_data[:, 0][:, None]
    y[y == 4] = -1  # Set digit 4 as class -1
    y[y == 9] = 1   # Set digit 9 as class +1

    # Prepare an accuracy matrix
    cv_acc = np.zeros([len(lamb_list), len(kernel_list)])

    # Manually split data into k folds
    fold_size = X.shape[0] // k  # Size of each fold

    for i, lamb in enumerate(lamb_list):
        for j, (kernel_name, kernel_func) in enumerate(kernel_list):
            fold_acc = []  # Store accuracy for each fold

            for fold in range(k):
                # Create validation and training splits manually
                start, end = fold * fold_size, (fold + 1) * fold_size
                Xval, yval = X[start:end], y[start:end]  # Validation set
                Xtrain = np.concatenate((X[:start], X[end:]), axis=0)  # Training set
                ytrain = np.concatenate((y[:start], y[end:]), axis=0)

                # Train using dual hinge loss
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)

                # Predict on validation set
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)

                # Calculate accuracy
                accuracy = np.mean(yhat == yval)
                fold_acc.append(accuracy)

            # Store the average accuracy for the current (lambda, kernel) pair
            cv_acc[i, j] = np.mean(fold_acc)

    # Identify the best lambda and kernel combination
    best_index = np.unravel_index(np.argmax(cv_acc, axis=None), cv_acc.shape)
    best_lamb = lamb_list[best_index[0]]
    best_kernel_name = kernel_list[best_index[1]][0]  # Extract the name of the best kernel

    return cv_acc, best_lamb, best_kernel_name

# Define hyperparameters
lamb_list = [0.1, 1]

kernel_list = [("linearKernel", linearKernel),("polyKernel (degree=3)", lambda X1, X2: polyKernel(X1, X2, degree=3)),("gaussKernel (width=5)", lambda X1, X2: gaussKernel(X1, X2, width=5))]

# Call the cvMnist function
dataset_folder = 'C:\\Users\\kyans\\3105Assignment2'
cv_acc, best_lamb, best_kernel = cvMnist(dataset_folder, lamb_list, kernel_list)

# Print the results
print("Cross-Validation Accuracy Matrix:\n", cv_acc)
print(f"Best Lambda: {best_lamb}")
print(f"Best Kernel: {best_kernel}")
"""