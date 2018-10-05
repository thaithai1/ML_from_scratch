import numpy as np
from utils import *
import matplotlib.pyplot as plt


class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    
    def init_param(self):
        pass
    
    def predict(self, X):
        pass
    
    def compute_cost(self, X, Y):
        pass
    
    def compute_grad(self, X, Y, W = None):
        pass

    def train(self, iter = 100, lr = 0.01, verbose = False):
        pass
    

class LinearReg(Model):
    """
    Linear Regression Model
    """
    def __init__(self, X_train, Y_train, X_test, Y_test):
        """
        n : Number of features excluding the one corresponding to the bias 
        """
        super().__init__(X_train, Y_train, X_test, Y_test)
        self.n, _ = X_train.shape
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

    def Gradient_Descent(self, lr = 0.01, X_train = None, Y_train = None):
        """
        One step of Gradient descent
        """
        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train
        
        grad = self.compute_grad(X_train, Y_train)
        self.W += - lr* grad

    def train(self, iter = 100, lr = 0.01,  X_train = None, Y_train = None, verbose = False):
        """
        Verbose = True : plot de the learning curves
        """
        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train

        if verbose:
            cost_train=[]
            cost_test=[]
            idx=[]

        for i in range(iter):
            self.Gradient_Descent(lr, X_train, Y_train)
            if verbose and i%5==0:
                idx.append(i)
                cost_train.append(self.compute_cost(X_train, Y_train))
                cost_test.append(self.compute_cost(self.X_test, self.Y_test))
        
        if verbose:
            plt.figure(figsize = (15,5))
            plt.plot(idx, cost_train, 'c', label = 'Train Loss')
            plt.plot(idx, cost_test, 'r', label = 'Test Loss')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
            plt.title('Loss ~ iter')
            plt.show()
            print(f'Train Error : {self.compute_cost(self.X_train, self.Y_train)}')
            print(f'Test Error : {self.compute_cost(self.X_test, self.Y_test)}')   

    def train_ann_lr(self, iter = 100,  X_train = None, Y_train = None, verbose = False):
        """
        Verbose = True : plot de the learning curves
        """
        lr=1e-5
        cost=1e10
        shrinked = False

        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train
        if verbose:
            cost_train=[]
            cost_test=[]
            idx=[]

        for i in range(iter):
            #Change learning rate
            temp = self.compute_cost(X_train, Y_train)
            if temp< cost and shrinked == False:
                lr *= 2
            elif temp > cost or np.isnan(temp):
                shrinked = True
                lr/= 2
                cost = temp
            #Gradient Descent
            self.Gradient_Descent(lr, X_train, Y_train)
            if verbose and i%5==0:
                idx.append(i)
                cost_train.append(self.compute_cost(X_train, Y_train))
                cost_test.append(self.compute_cost(self.X_test, self.Y_test))
        
        if verbose:
            plt.figure(figsize = (15,5))
            plt.plot(idx, cost_train, 'c', label = 'Train Loss')
            plt.plot(idx, cost_test, 'r', label = 'Test Loss')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
            plt.title('Loss ~ iter')
            plt.show()
            print(f'Train Error : {self.compute_cost(self.X_train, self.Y_train)}')
            print(f'Test Error : {self.compute_cost(self.X_test, self.Y_test)}')

    def Gradient_checking(self, eps = 1e-5):
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

    def learning_curve_ex(self, iter = 300, lr = 0.01, m_step = 20, max_ex = None):
        if max_ex is None:
            max_ex = self.X_train.shape[1]
        else:
            max_ex = min(max_ex, self.X_train.shape[1])

        m_list = list(range(1, max_ex,m_step)) 
        train_loss = list()
        test_loss = list()
        for m in m_list:
            train_loss_temp = list()
            test_loss_temp = list()
            for i in range(20):
                idx=np.random.permutation(self.X_train.shape[1])
                X_train_temp=self.X_train[:,idx[:m]]
                Y_train_temp=self.Y_train[:,idx[:m]]
                self.init_param()
                self.train(iter, lr, X_train_temp, Y_train_temp)
                train_loss_temp.append(self.compute_cost(X_train_temp, Y_train_temp))
                test_loss_temp.append(self.compute_cost(self.X_test, self.Y_test))
            train_loss.append(np.mean(train_loss_temp))
            test_loss.append(np.mean(test_loss_temp))
        
        plt.figure(figsize = (15, 8))
        plt.title('Loss against number of examples')
        plt.plot(m_list, train_loss, c = 'b', label = 'Train loss')
        plt.plot(m_list, test_loss, c = 'r', label = 'Test loss')
        plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Number of training examples')
        plt.ylabel('Loss')
        plt.show()



class LogisticReg(Model):
    """
    Logistic Regression
    """
    def __init__(self, X_train, Y_train, X_test, Y_test):
        """
        n : Number of features excluding the one corresponding to the bias 
        """
        super().__init__(X_train, Y_train, X_test, Y_test)
        self.n, _ = X_train.shape
        self.init_param()
    
    def init_param(self):
        """
        Initialize parameters
        """
        self.W=np.random.randn(1,self.n+1)
    
    def predict_prob(self, X, W = None):
        """
        Predict X using parameters W
        input:
            X : shape (n ,m)
            W : shape (1, n+1 )
        """
        if W is None :
            W = self.W
        X_ones = vstack_one(X)
        return sigmoid(np.dot(W, X_ones))

    def predict(self, X, W = None):
        Y_pred=self.predict_prob(X,W)
        return (Y_pred>0.5).astype(int)

    def accuracy(self, X, Y, W = None):
        return np.mean(self.predict(X)==Y)

    def compute_cost(self, X, Y, W=None):
        if W is None :
            W = self.W
        pred = self.predict_prob(X, W)
        cost = - Y * np.log(pred) - (1-Y) * np.log(1-pred) 
        return float(np.mean(cost, axis=1))
    
    def compute_grad(self, X, Y, W = None):
        if W is None :
            W = self.W
        grad = np.zeros((1, self.n+1))
        diff = self.predict(X, W) - Y
        X_ones = vstack_one(X)
        grad =  np.dot(diff, X_ones.T) / X.shape[1]
        return grad

    def Gradient_Descent(self, lr = 0.01, X_train = None, Y_train = None):
        """
        One step of Gradient descent
        """
        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train

        grad = self.compute_grad(X_train, Y_train)
        self.W += - lr* grad

    def train(self, iter = 100, lr = 0.01, X_train = None, Y_train = None, verbose = False):
        """
        Verbose = True : plot de the learning curves
        """
        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train
        if verbose:
            cost_train=[]
            cost_test=[]
            idx=[]
        for i in range(iter):
            self.Gradient_Descent(lr, X_train, Y_train)
            if verbose and i%5==0:
                idx.append(i)
                cost_train.append(self.compute_cost(X_train, Y_train))
                cost_test.append(self.compute_cost(self.X_test, self.Y_test))
        
        if verbose:
            plt.figure(figsize = (15,5))
            plt.plot(idx, cost_train, 'c', label = 'Train Loss')
            plt.plot(idx, cost_test, 'r', label = 'Test Loss')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
            plt.title('Loss ~ iter')
            plt.show()
            print(f'Train Error : {self.compute_cost(self.X_train, self.Y_train)}')
            print(f'Test Error : {self.compute_cost(self.X_test, self.Y_test)}')
            print(f'Accuracy Test : {self.accuracy(self.X_test, self.Y_test)}')        

    def train_ann_lr(self, iter = 100,  X_train = None, Y_train = None, verbose = False):
        """
        Verbose = True : plot de the learning curves
        """
        lr=1e-5
        cost=1e10
        shrinked = False

        if X_train is None and Y_train is None:
            X_train = self.X_train
            Y_train = self.Y_train
        if verbose:
            cost_train=[]
            cost_test=[]
            idx=[]

        for i in range(iter):
            #Change learning rate
            temp = self.compute_cost(X_train, Y_train)
            if temp < cost and shrinked == False:
                lr *= 2
            elif temp > cost or np.isnan(temp):
                shrinked = True
                lr/= 2
                cost = temp
            #Gradient Descent
            self.Gradient_Descent(lr, X_train, Y_train)
            if verbose and i%5==0:
                idx.append(i)
                cost_train.append(self.compute_cost(X_train, Y_train))
                cost_test.append(self.compute_cost(self.X_test, self.Y_test))
        
        if verbose:
            plt.figure(figsize = (15,5))
            plt.plot(idx, cost_train, 'c', label = 'Train Loss')
            plt.plot(idx, cost_test, 'r', label = 'Test Loss')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
            plt.title('Loss ~ iter')
            plt.show()
            print(f'Train Error : {self.compute_cost(self.X_train, self.Y_train)}')
            print(f'Test Error : {self.compute_cost(self.X_test, self.Y_test)}')
            print(f'Accuracy Test : {self.accuracy(self.X_test, self.Y_test)}')

    def Gradient_checking(self, eps = 1e-5):
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

    def learning_curve_ex(self, iter = 300, lr = 0.01, m_step = 20, max_ex = None):
        if max_ex is None:
            max_ex = self.X_train.shape[1]
        else:
            max_ex = min(max_ex, self.X_train.shape[1])

        m_list = list(range(1, max_ex,m_step)) 
        train_loss = list()
        test_loss = list()
        for m in m_list:
            train_loss_temp = list()
            test_loss_temp = list()
            for i in range(20):
                idx=np.random.permutation(self.X_train.shape[1])
                X_train_temp=self.X_train[:,idx[:m]]
                Y_train_temp=self.Y_train[:,idx[:m]]
                self.init_param()
                self.train(iter, lr, X_train_temp, Y_train_temp)
                train_loss_temp.append(self.compute_cost(X_train_temp, Y_train_temp))
                test_loss_temp.append(self.compute_cost(self.X_test, self.Y_test))
            train_loss.append(np.mean(train_loss_temp))
            test_loss.append(np.mean(test_loss_temp))
        
        plt.figure(figsize = (15, 8))
        plt.title('Loss against number of examples')
        plt.plot(m_list, train_loss, c = 'b', label = 'Train loss')
        plt.plot(m_list, test_loss, c = 'r', label = 'Test loss')
        plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Number of training examples')
        plt.ylabel('Loss')
        plt.show()
    