import numpy as np

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, N, D, C, iid):
    if (iid==0):
        samples_per_user = np.random.lognormal(4, 2, N).astype(int) + 50
    else:
        samples_per_user = [2000 for _ in range(N)]
    
    X_split = [[] for _ in range(N)]
    Y_split = [[] for _ in range(N)]
    
    mean_W = np.random.normal(0, alpha, N)
    mean_b = mean_W
    B = np.random.normal(0, beta, N)
    mean_x = np.zeros((N, D))
    
    diagonal = np.zeros(D)
    for i in range(D):
        diagonal[i] = np.power((i+1), -1.2)
    Sigma = np.diag(diagonal)
    
    for i in range(N):
        if iid == 1:
            mean_x[i] = np.ones(D)*B[i]
        else:
            mean_x[i] = np.random.normal(B[i], 1, D)
            
    if iid == 1:
        W_global = np.random.normal(0, 1, (D, C))
        b_global = np.random.normal(0, 1,  C)
        
    
    for i in range(N):
        W = np.random.normal(mean_W[i], 1, (D, C))
        b = np.random.normal(mean_b[i], 1, C)
        
        if iid == 1:
            W = W_global
            b = b_global
        
        X = np.random.multivariate_normal(mean_x[i], Sigma, samples_per_user[i])
        Y = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            Y[j] = np.argmax(softmax(np.dot(X[j], W) + b))
    
        X_split[i] = X
        Y_split[i] = Y
    
    return X_split, Y_split