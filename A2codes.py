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
    #these lines make the kernel matrix 
    n = X.shape[0]
    K = kernel_func(X, X)
    #this makes sure that the target label are flat meaning a 1-D array
    y = y.flatten()

    #this function calculates the loss to be minimized
    def objective(params):
        #first element of params
        alpha = params[0:n]
        alpha0 = params[n]
        #the value that we get from combining the kernel matrix and alpha the first element
        s = K @ alpha + alpha0
        y_s = y * s
        term1 = np.sum(np.log(1 + np.exp(-y_s)))
        term2 = (lamb / 2) * alpha.T @ K @ alpha
        #total loss
        L = term1 + term2
        return L

    #this function gets the gradient that is need for optimizing
    def gradient(params):
        #again this is the first element of the list
        alpha = params[0:n]
        alpha0 = params[n]
        #the value that we get from combining the kernel matrix and alpha the first element
        s = K @ alpha + alpha0
        y_s = y * s
        #this is the predicted value y hat
        p = 1 / (1 + np.exp(y_s)) 
        #gradient with respect to s
        gradient_s = -y * p 
        #gradient with respect to alpha
        gradient_alpha = K @ gradient_s + lamb * K @ alpha 
        #gradient with respect to alpha0
        gradient_alpha0 = np.sum(gradient_s) 
        #putting all the array into a single one 
        grad = np.concatenate([gradient_alpha, [gradient_alpha0]])
        #return the full array
        return grad

    # Initial guess for α and α0
    initial_params = np.zeros(n + 1)

    # Perform the optimization
    res = minimize(objective, initial_params, method='L-BFGS-B', jac=gradient)

    params_opt = res.x
    #gets the optimal alpha value
    alpha_opt = params_opt[0:n]
    alpha0_opt = params_opt[n]
    #return a 2-D array and the optimal alpha value
    return alpha_opt[:, np.newaxis], alpha0_opt


#2b
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    #this makes sure that the target label are flat meaning a 1-D array
    n = X.shape[0]
    y = y.flatten()

    #this lines make the kernel matrix 
    K = kernel_func(X, X)
    
    #this matrix is The matrix representing the quadratic term, it's initalized to zero and has a specific size
    P = np.zeros((2*n + 1, 2*n + 1))
    #put lemda * K + stabilizer * identiy matrix in the top left corner
    P[:n, :n] = lamb * K + stabilizer * np.eye(n)
    # Add stabilizer to the diagonal elements for numerical stability
    P = P + stabilizer * np.eye(2*n + 1)
    
    #A vector representing the linear term in this function
    q = np.zeros((2*n + 1))
    #slack variable that is set to 1
    q[n+1:] = 1.0 
    
    #G1 and h1 represent the first set of inequality constraints that enforce that the slack variables that are non-negative
    G1 = np.zeros((n, 2*n + 1))
    G1[:, n+1:] = -np.eye(n)
    h1 = np.zeros(n)
    
    #G2 and h2 represent the second set of inequality constraints based on the hinge loss formula.
    G2 = np.zeros((n, 2*n + 1))
    G2[:, :n] = -np.diag(y) @ K
    G2[:, n] = -y
    G2[:, n+1:] = -np.eye(n)
    h2 = -np.ones(n)
    
    #Combine G and h
    G = np.vstack((G1, G2))
    h = np.hstack((h1, h2))
    
    #Convert numpy arrays to cvxopt matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    
    #Solve the quadratic program
    solvers.options['show_progress'] = False  # Suppress output
    solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    #The optimal solution is retrieved and flattened into a 1D array
    x_opt = np.array(solution['x']).flatten()
    
    #get the first alpha element which is the first element for the support vector
    alpha_opt = x_opt[:n]
    #this is the next element
    alpha0_opt = x_opt[n]
    
    #returns alpha_opt as a 2D array and alpha0_opt
    
    return alpha_opt[:, np.newaxis], alpha0_opt

#2c
def adjClassify(Xtest, a, a0, X, kernel_func):

    #This line computes the kernel matrix between the test data and the training data
    K_test = kernel_func(Xtest, X)

    #The s values are calculated by taking the dot product of the kernel matrix and a, then adding a0.
    s = K_test @ a + a0

    #y hat label is obtained by applying the np.sign() function to the prev s value
    yhat = np.sign(s)

    return yhat


def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [
        linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)
    ]
    kernel_names = ['Linear', 'Poly(d=2)', 'Poly(d=3)', 'Gauss(sigma=1)', 'Gauss(sigma=0.5)']
    gen_model_list = [1, 2, 3]
    gen_model_names = ['Model 1', 'Model 2', 'Model 3']

    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                #Train using adjBinDev
                a_bindev, a0_bindev = adjBinDev(Xtrain, ytrain, lamb, kernel)
                #Predict on training data
                ytrain_pred = adjClassify(Xtrain, a_bindev, a0_bindev, Xtrain, kernel)
                #Compute training accuracy
                train_acc_bindev[i, j, r] = np.mean(ytrain_pred.flatten() == ytrain.flatten())
                #Predict on test data
                ytest_pred = adjClassify(Xtest, a_bindev, a0_bindev, Xtrain, kernel)
                #Compute test accuracy
                test_acc_bindev[i, j, r] = np.mean(ytest_pred.flatten() == ytest.flatten())

                #Train using adjHinge
                a_hinge, a0_hinge = adjHinge(Xtrain, ytrain, lamb, kernel)
                #Predict on training data
                ytrain_pred = adjClassify(Xtrain, a_hinge, a0_hinge, Xtrain, kernel)
                #Compute training accuracy
                train_acc_hinge[i, j, r] = np.mean(ytrain_pred.flatten() == ytrain.flatten())
                #Predict on test data
                ytest_pred = adjClassify(Xtest, a_hinge, a0_hinge, Xtrain, kernel)
                #Compute test accuracy
                test_acc_hinge[i, j, r] = np.mean(ytest_pred.flatten() == ytest.flatten())

    #average accuracies over runs
    avg_train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    avg_test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    #Combine accuracies (BinDev and Hinge) into matrices
    train_acc = np.hstack((avg_train_acc_bindev, avg_train_acc_hinge))
    test_acc = np.hstack((avg_test_acc_bindev, avg_test_acc_hinge))
    #return 5-by-6 train accuracy and 5-by-6 test accuracy
    return train_acc, test_acc

#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    #this makes sure that the target label are flat meaning a 1-D array
    n = X.shape[0]
    y = y.flatten()

    #this lines make the kernel matrix 
    K = kernel_func(X, X)

    #Y is a diagonal matrix made from the target labels y. This matrix will be used to adjust the kernel matrix according to the labels.
    Y = np.diag(y)
    #Q is a quadratic matrix which will be used in the optimization problem.
    Q = (1 / lamb) * (Y @ K @ Y)

    #the stabilizer is added to the diagonal of Q to ensure numerical stability during optimization
    Q += stabilizer * np.eye(n)

    #Convert Q to cvxopt matrix
    P = matrix(Q)
    q = matrix(-np.ones(n))

    #The first half of G contains −In−In​ to ensure α≥0α≥0
    #The second half contains InIn​ to ensure α≤1α≤1.
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    #h specifies the upper and lower bounds for these constraints
    h = matrix(np.hstack([np.zeros(n), np.ones(n)]))

    #This constraint ensures that the sum of the dual variables weighted by their corresponding labels equals zero
    A = matrix(y, (1, n), 'd') 
    b_eq = matrix(0.0)

    #This line prevents the solver from displaying output during optimization
    solvers.options['show_progress'] = False

    #Solve the quadratic program
    solution = solvers.qp(P, q, G, h, A, b_eq)

    #The optimal dual variables are extracted from the solution and flattened into a 1D array
    alpha = np.array(solution['x']).flatten()

    #this is used to identify the indices of non-bound support vectors 
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

    #The dual variables alpha are reshaped into a column vector, and both alpha and b are returned
    alpha = alpha.reshape(-1, 1)
    return alpha, b



#3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    #this makes sure that the target label are flat meaning a 1-D array
    y = y.flatten()
    
    #This line computes the kernel matrix between the test data and the training data
    K_test = kernel_func(Xtest, X)

    #This calculate the product of the dual coefficients a and the training labels y
    alpha_y = a.flatten() * y

    #The s values for the test samples are computed using the kernel matrix and the adjusted dual coefficients
    s = (1 / lamb) * (K_test @ alpha_y) + b

    #The predicted class labels are obtained by applying the np.sign() to the s value
    yhat = np.sign(s)

    #This line reshapes yhat to ensure it is a column vector with M row
    yhat = yhat.reshape(-1, 1)

    return yhat


#3c
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(os.path.join(dataset_folder, 'A2train.csv'), header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1

    cv_acc = np.zeros([len(lamb_list), len(kernel_list)])

    # Manually split data into k folds
    fold_size = X.shape[0] // k  # Size of each fold

    for i, lamb in enumerate(lamb_list):
        for j, (kernel_name, kernel_func) in enumerate(kernel_list):
            fold_acc = []

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

            # calculate validation accuracy
            cv_acc[i, j] = np.mean(fold_acc)

    # identify the best lamb and kernel functio
    best_index = np.unravel_index(np.argmax(cv_acc, axis=None), cv_acc.shape)
    best_lamb = lamb_list[best_index[0]]
    best_kernel_name = kernel_list[best_index[1]][0]


    # return a "len(lamb_list)-by-len(kernel_list)" accuracy variable, the best lamb and the best kernel
    return cv_acc, best_lamb, best_kernel_name


''' use this code to test 2d meaning to get the tables for q2
train_acc, test_acc = synExperimentsKernel()

# Print the results
print("Train Accuracies:")
print(train_acc)
print("\nTest Accuracies:")
print(test_acc)
'''

'''
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
'''