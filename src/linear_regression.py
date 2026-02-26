
import numpy as np
import scipy.stats as stats

class LinearRegression:     # Define the Linear Regression class
  

    def __init__(self, confidence_level=0.95): # Initialize the Linear Regression model - later deciides on what the model will be trained on
        self.confidence_interval = confidence_level # Confidence level for confidence intervals (default is 0.95 for 95% confidence intervals)
        self.beta = None        # Coefficients of the linear model (later will be trained to find the best coefficients)
        self.n = None           # Number of samples (for the model to remember the number of samples it was trained on)
        self.d = None           # Number of features (needed for f-test, t-test, and variance calculations)
        self.residuals = None     # Residuals of the model (difference between observed and predicted values)
        self.y_hat = None          # Predicted values from the model (the output of the model when given input features)
        self.SSE = None           # Sum of Squared Errors (used for variance estimation)
        self.sigma2_hat = None    # Estimated variance of the errors (used for t-tests and confidence intervals)

    def fit(self, X, y):
        """
        Fit the linear regression model to the data X and y using Ordinary Least Squares (OLS).
        This method will compute the coefficients (beta) that minimize the sum of squared errors.
        It will also compute the predicted values (y_hat) and residuals (y - y_hat).
        """
        self.n = X.shape[0] # Get the number of samples (n) from the shape of X
        self.d = X.shape[1]-1 # Get the number of features (d) from the shape of X, subtracting 1 for the intercept term
        
        XtX = X.T @ X # Compute X^T * X
        XtY = X.T @ y # Compute X^T * y
        self.beta = np.linalg.inv(XtX) @ XtY # Compute the coefficients (beta) using the OLS formula: beta = (X^T * X)^(-1) * X^T * y

        self.y_hat = X @ self.beta # Compute the predicted values (y_hat) using the coefficients
        self.residuals = y - self.y_hat # Compute the residuals (y - y_hat)
    
    def estimate_variance(self):
        """ Estimate the variance of the errors (sigma^2) using the residuals from the fitted model. 
        This is done by computing the Sum of Squared Errors (SSE) and then dividing by (n - d - 1), 
        where n is the number of samples and d is the number of features. 
        The resulting sigma^2_hat is an estimate of the variance of the errors in the linear regression model.
        """
        self.SSE = np.sum(self.residuals ** 2) # Compute the Sum of Squared Errors (SSE)
        self.sigma2_hat = self.SSE / (self.n - self.d - 1) # Estimate the variance of the errors using the formula
        return self.sigma2_hat # Return the estimated variance
    
    def standard_deviation(self):
        """ Compute the standard deviation of the errors (sigma) from the estimated variance (sigma^2_hat). 
        The standard deviation is the square root of the variance and provides a measure of the average magnitude of the errors in the linear regression model.
        """
        return np.sqrt(self.estimate_variance()) # Return the standard deviation by taking the square root of the estimated variance
    
    def rmse(self):
         """Compute the Root Mean Squared Error (RMSE) from the residuals. 
         RMSE is a measure of the average magnitude of the errors between predicted and observed values."""
         return np.sqrt(np.mean(self.residuals ** 2)) # Compute and return the Root Mean Squared Error (RMSE) from the residuals



