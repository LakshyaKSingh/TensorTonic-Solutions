import numpy as np
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X=np.array(X)
    y=np.array(y)
    n_features=X.shape[1]
    XT=X.T
    XTX=np.dot(XT,X)
    lambda_I=lam*np.eye(n_features)
    weights=np.linalg.inv(XTX+lambda_I).dot(XT).dot(y)
    return weights.tolist()