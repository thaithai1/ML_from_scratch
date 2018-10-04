import numpy as np
from utils import *
import matplotlib.pyplot as plt

class LinearReg():
    """
    Linear Regression Model
    """


    def __init__(self, X_train, Y_train, X_test, Y_test):
        """
        n : Number of features excluding the one corresponding to the bias 
        """
        self.n, self.m = X_train.shape
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.init_param()


    def init_param(self):
        """
        Initialize parameters
        """
        self.W=np.random.randn(1,self.n+1)


    def predict(self, X, W = None):
        """
        Predict X using parameters W
        input:
            X : shape (n ,m)
            W : shape (1, n+1 )
        """
        if W is None :
            W = self.W
        X_ones = vstack_one(X)
        return np.dot(W, X_ones)


    def compute_cost(self, X, Y,W=None):
        if W is None :
            W = self.W
        errors = np.linalg.norm(self.predict(X, W) - Y, axis=0, keepdims = True)**2
        return float(np.mean(errors, axis=1))
    

    def compute_grad(self, X, Y, W = None):
        if W is None :
            W = self.W
        grad = np.zeros((1, self.n+1))
        diff = self.predict(X, W) - Y
        X_ones = vstack_one(X)
        grad = 2 * np.dot(diff, X_ones.T) / X.shape[1]
        return grad


    def Gradient_Descent(self, lr = 0.01):
        """
        One step of Gradient descent
        """
        grad = self.compute_grad(self.X_train, self.Y_train)
        self.W += - lr* grad


    def train(self, iter = 100, lr = 0.01, verbose = False):
        """
        Verbose = True : plot de the learning curves
        """
        if verbose:
            cost_train=[]
            cost_test=[]
            idx=[]
        for i in range(iter):
            self.Gradient_Descent()
            if verbose and i%10==0:
                idx.append(i)
                cost_train.append(self.compute_cost(self.X_train, self.Y_train))
                cost_test.append(self.compute_cost(self.X_test, self.Y_test))
        
        if verbose:
            plt.figure(figsize = (15,5))
            plt.plot(idx, cost_train, 'c+', label = 'Train Loss')
            plt.plot(idx, cost_test, 'r+', label = 'Test Loss')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
            plt.title('Loss ~ iter')
            plt.show()
        print(f'Train Error : {self.compute_cost(self.X_train, self.Y_train)}')
        print(f'Test Error : {self.compute_cost(self.X_test, self.Y_test)}')        


    def gradient_checking(self, eps = 1e-5):
        """
        Check if gradient implementation is right
        """
        n, m = 10, 100
        W = np.random.randn(1,n+1)
        X = np.random.randn(n,m)
        Y = np.random.randn(1,m)

        #Gradient
        grad = self.compute_grad(X, Y, W)

        #Gradient approximation
        grad_approx=np.zeros((1, n+1))
        for i in range(n+1):
            W_temp = W.copy()
            W_temp[0,i] = W[0,i] - eps
            J_minus = self.compute_cost(X, Y, W_temp)

            W_temp[0,i]= W[0,i] + eps
            J_plus = self.compute_cost(X, Y, W_temp)

            grad_approx[0,i] = (J_plus - J_minus)/(2*eps)
        
        error = np.linalg.norm(grad_approx - grad)
        rel_error = error / (np.linalg.norm(grad_approx) + np.linalg.norm(grad))
        print("Relative error : {}".format(rel_error))

    




    
    