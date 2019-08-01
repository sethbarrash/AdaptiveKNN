import numpy as np

def generate_data(eta, noise, n):
    if eta == "para":
        x = np.random.normal(0, 1, (n,))
        y = x ** 2 + np.random.normal(0, noise, (n,))
        xtest = np.random.normal(0, 1)
        ytest = xtest ** 2 + np.random.normal(0, noise)
    elif eta == "sine":
        x = np.random.uniform(0, 2 * np.pi, (n,))
        y = np.sin(x) + np.random.normal(0, noise, (n,))
        xtest = np.random.uniform(0, 1)
        ytest = np.sin(xtest) + np.random.normal(0, noise)

    return x, y, xtest, ytest