import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.svm import SVR

from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


RAND=13

def franke_function(x, y):
    """
    Returns the Franke's function that has two Gaussian peaks of different heights and a smaller dip. 

    Args:
        x (np.Array[float]):        Inputs within [0, 1]
        y (np.Array[float]):        Inputs within [0, 1]

    Returns:
        z (np.Array[float]):        Outputs of the Franke's function
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


def prepare_data(N, sigma=0.03):
    """
    Returns the Franke's function that has two Gaussian peaks of different heights and a smaller dip. 
    The noise may be added stochastic noise with the normal distribution N[0, 1].
    """

    x = np.linspace(0, 1, N) 
    y = np.linspace(0, 1, N) 
    x_2d, y_2d = np.meshgrid(x, y)
    x = np.ravel(x_2d)
    y = np.ravel(y_2d)

    data = franke_function(x, y)

    noise = np.random.normal(0,sigma,(x.shape))
    data = data + noise

    return data, x, y


def design_matrix(x, y, degree):
    """"
    Implements the Nd polynomial design matrix of a given degree based on a dataset. Includes intercept.

    Args:
        x (np.Array[float]):            x-data
        y (np.Array[float]):            y-data
        degree (int):                   N degree of the polynomial

    Returns:
        X (np.Array['z, n', float]):    Design matrix X
    """

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    P = int((degree+1)*(degree+2)/2)
    X = np.ones((N,P))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = x**(i-j) * y**j

    return X


def split_and_scale_data(X, z, test_size=0.2):

    # Train test split
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, shuffle=True, random_state=RAND)

    for i in range(len(X_train.T)):
        X_train[:,i] = X_train[:,i] - np.mean(X_train[:,i])
        X_test[:,i]  = X_test[:,i] - np.mean(X_test[:,i])

    z_train = z_train - np.mean(z_train)
    z_test = z_test - np.mean(z_test)

    return X_train, X_test, z_train, z_test


def make_plot(title, model, max_degree, z, x, y):

    # Initialize arrays
    mse = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    var = np.zeros(max_degree)
    degrees = range(1, max_degree+1)

    for d in degrees:

        # Split and train
        X = design_matrix(x, y, d)
        X_train, X_test, y_train, y_test = split_and_scale_data(X, z, test_size=0.2)

        # Get scores
        mse[d-1], bias[d-1], var[d-1] = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=100, random_seed=RAND)

    # Plot the graph:
    plt.figure()
    plt.plot(degrees, mse, label='MSE')
    plt.plot(degrees, bias, "--", label='bias')
    plt.plot(degrees, var, "--", label='variance')

    # Customize
    plt.xlabel('Polynomal model complexity [degrees]')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(str(title))
    plt.legend()
    plt.grid()

    print("{}: MSE: {}".format(str(model), mse))
    print("{}: Smallest MSE: {}".format(str(model), min(mse)))

    # Save output
    plt.savefig(str(title))

    plt.close()


def make_plot_svm(title, max_degree, z, x, y):

    # Initialize arrays
    mse = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    var = np.zeros(max_degree)
    degrees = range(1, max_degree+1)

    for d in degrees:

        model = SVR(kernel='poly', degree=d)

        # Split and train
        X = design_matrix(x, y, d)
        X_train, X_test, y_train, y_test = split_and_scale_data(X, z, test_size=0.2)

        # Get scores
        mse[d-1], bias[d-1], var[d-1] = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=100, random_seed=RAND)

    # Plot the graph:
    plt.figure()
    plt.plot(degrees, mse, label='MSE')
    plt.plot(degrees, bias, "--", label='bias')
    plt.plot(degrees, var, "--", label='variance')

    # Customize
    plt.xlabel('Polynomal model complexity [degrees]')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(str(title))
    plt.legend()
    plt.grid()

    print("{}: MSE: {}".format(str(model), mse))
    print("{}: Smallest MSE: {}".format(str(model), min(mse)))

    # Save output
    plt.savefig(str(title))

    plt.close()


def make_plot_tree(title, model, max_depth, z, x, y):

    # Initialize arrays
    mse = np.zeros(max_depth)
    bias = np.zeros(max_depth)
    var = np.zeros(max_depth)
    depths = range(1, max_depth+1)

    for d in depths:

        # Split and train
        X = design_matrix(x, y, d)
        X_train, X_test, y_train, y_test = split_and_scale_data(X, z, test_size=0.2)

        # Get scores
        mse[d-1], bias[d-1], var[d-1] = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=100, random_seed=RAND)

    # Plot the graph:
    plt.figure()
    plt.plot(depths, mse, label='MSE')
    plt.plot(depths, bias, "--", label='bias')
    plt.plot(depths, var, "--", label='variance')

    # Customize
    plt.xlabel('Polynomal model complexity [tree depth]')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(str(title))
    plt.legend()
    plt.grid()

    print("{}: MSE: {}".format(str(model), mse))
    print("{}: Smallest MSE: {}".format(str(model), min(mse)))

    # Save output
    plt.savefig(str(title))

    plt.close()


def main():

    N = 20
    sigma = 0.03

    # Read data
    data, x, y = prepare_data(N, sigma)

    make_plot("linear_regression-ols", LinearRegression(), 10, data, x, y)
    make_plot("linear_regression-ridge", Ridge(alpha=0.001), 35, data, x, y)
    make_plot("linear_regression-lasso", Lasso(alpha=0.0000001), 40, data, x, y)
    
    make_plot("trees-decision_tree", DecisionTreeRegressor(), 30, data, x, y)
    make_plot("trees-bagging", BaggingRegressor(base_estimator=DecisionTreeRegressor(), random_state=RAND), 30, data, x, y)
    make_plot("trees-gradient_boosting", GradientBoostingRegressor(), 10, data, x, y)
    make_plot("trees-random_forest", RandomForestRegressor(), 10, data, x, y)

    make_plot_svm("SVM-SVR_kernel_poly", 5, data, x, y)

if __name__ == "__main__":
    main()