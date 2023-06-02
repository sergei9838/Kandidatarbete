import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

#%% Based on /Users/sergei/Python/PycharmProjects/pytorchbnn/Gaussian_regression.py
# now input can be a vector rather than a number:
# response(x_1,...,x_p) = sum_j^D eta_j(m_j, v_j) h_j(x_1,...,x_p),
# where eta_j(m_j, v_j) are independent Gaussian r.v.'s with mean m_j and variance v_j
# In particular, h_0=1 and h_j(x_1,...,x_p)=x_j gives the simplest model:
# response(x_1,...,x_p) = eta_0(m_0, v_0) + sum_{j=1}^p eta_j(m_j, v_j) x_j

class GaussianMultRegression:
    def __init__(self, funcs: list, means=None, st_devs=None):
        '''
        :param funcs: list of torch functions of p arguments. Each function operates on
            a tensor of size (*,p). The output should be a column tensor (*,1)
        :param means: means of the coefficients
        :param st_devs: standard veciations of the coefficients
        '''
        self.funcs = funcs
        self.dim_regression = len(funcs)
        if means is None:
            self.means = torch.zeros(self.dim_regression, requires_grad=True)
        else:
            self.means = torch.tensor([float(m) for m in means], requires_grad=True)
        if st_devs is None:
            self.st_devs = torch.ones(self.dim_regression, requires_grad=True)
        else:
            self.st_devs = torch.tensor([float(v) for v in st_devs], requires_grad=True)

    def mean_response(self, X, means=None):
        '''
        :param X: input variables: a tensor with (*, self.dim_regression) shape
        :param means: the means of the coefs, if none, self.means are used
        :return: expected value of the regression at each row from X
        '''
        if means is None:
            means = self.means
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if X.dim() == 1: # X is a single vector
            X = X.reshape(1,-1)
        ndat = X.shape[0]
        ave = torch.zeros(ndat).reshape(-1,1)
        for j in range(self.dim_regression):
            ave += means[j] * self.funcs[j](X)
        return ave.reshape(-1,1)

    def std_response(self, X, st_devs=None):
        '''
        :param X: input variables: a matrix (*, self.dim_regression) shape
        :param st_devs: the variances of the coefs, if none, self.st_devs are used
        :return: variance of the regression at each row from X
        '''
        if st_devs is None:
            st_devs = self.st_devs
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if X.dim() == 1:  # X is a single vector
            X = X.reshape(1, -1)
        ndat = X.shape[0]
        var = torch.zeros(ndat).reshape(-1,1)
        for j in range(self.dim_regression):
            var += (st_devs[j] * self.funcs[j](X))**2
        return torch.sqrt(var).reshape(-1,1)

    def simulate(self, X, means=None, st_devs=None):
        '''
        simulate regression
        :param X: input variables: a matrix (*, self.dim_regression) shape
        :param means: the means of the coefs, if none, self.means are used
        :param st_devs: the standard deviations of the coefs, if none, self.st_devs are used
        :return: values of the regression at each row of X
        '''
        if means is None:
            means = self.means
        if st_devs is None:
            st_devs = self.st_devs
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if X.dim() == 1:  # X is a single vector
            X = X.reshape(1,-1)
        ndat = X.shape[0]
        sim = torch.zeros(ndat).reshape(-1,1)
        for j in range(self.dim_regression):
            sim += (torch.randn(ndat).reshape(-1,1) * st_devs[j] + means[j]) * self.funcs[j](X)
        return sim.reshape(-1,1)

    def neg_log_pr(self, y, mean: torch.Tensor=torch.tensor([0.]), std: torch.Tensor=torch.tensor([1.])):
        return torch.log(std) + 0.5 * ((y - mean) / std)**2 # ignoring constant log(2*pi)

    def neg_loglik(self, X: torch.Tensor, y: torch.Tensor):
        if X.dim() == 1:  # X is a single vector
            X = X.reshape(1, -1)
        return self.neg_log_pr(y, self.mean_response(X), self.std_response(X)).sum()
        # ndat = X.shape[0]
        # log_probs = torch.zeros(1)
        # for i in range(ndat):
        #     log_probs += self.neg_log_pr(Y[i], self.mean_response(X[i, :]), self.variance_response(X[i, :]))
        # return(log_probs)

    def coefs_classic(self, x, y):
        ''' Fit the coefficients beta of the standard linear regression:
        y_i = sum_j beta_j h_j(x_i)+ epsilon_i, or Y = X beta + epsilon.
        The estimate of beta is
        beta_hat = (X' X)^(-1) X' y
        where X=||h_j(x_i)|| with h_0=1 if the intercept is fitted
        and X' is the transpose of X
        :param x: column tensor of the input
        :param y: column tensor of the response
        :return: row tensor of the estimated beta_hat
        '''
        X = self.funcs[0](x)
        for f in self.funcs[1:]:
            X = torch.cat((X, f(x)), dim=1)
        XX_inv = torch.inverse(torch.matmul(X.transpose(0,1), X))
        coefs = torch.matmul(torch.matmul(XX_inv, X.transpose(0,1)), y.reshape(-1, 1))
        return torch.tensor([c for c in coefs], requires_grad=True)

# %%
    def fit(self, data_x, data_y, params=None, start_means=None, start_stds=None,
            algo=torch.optim.SGD, no_steps=1000, lr=0.01, verbose=False):
        '''
        Estimate the parameters means and/or st-devs by maximizing the log-likelihood
        :param data_x:  locations data
        :param data_y: response data
        :param params: if None, fit both means and st_devs,
            if params=[self.means] - fit only means,
            if params=[self.st_devs] - fit only the standard deviations
        :param start_means: starting values for the means. If none, estimate from the
            ordinary (non-stochastic) regression is used.
        :param start_stds: starting values for the st. deviations. If None, self.st_devs is used
        :param algo: optimizer algorithm to be used, e.g. torch.optim.Adam
        :param no_steps: the number of steps for minimization
        :param lr: learning rate
        :param : if True, print the current values of minus log-likelihood and lr
        :return: updated self.means and/or self.st_devs and the value of minus log-likelihood
        '''
        if type(data_x) is not torch.Tensor:
            data_x = torch.tensor(data_x)
        if type(data_y) is not torch.Tensor:
            data_y = torch.tensor(data_y)

        if params is None:
            params = [self.means, self.st_devs] # estimate both means and st_devs

        # As the starting point for the means, if not given,
        # take the coefficients of the standard linear regression: Y = X beta + epsilon
        if start_means is None:
            self.means = self.coefs_classic(data_x, data_y)
        else:
            if type(start_means) is not torch.Tensor:
                self.means = torch.tensor(start_means)
            else:
                self.means = start_means

        if start_stds is not None:
            if type(start_stds) is not torch.Tensor:
                self.st_devs = torch.tensor(start_stds)
            else:
                self.st_devs = start_stds

        ## Turning on the grads
        self.means.requires_grad_(True)
        self.st_devs.requires_grad_(True)

        optimizer = algo(params, lr=lr)

        for steps in range(no_steps):
            NLL = self.neg_loglik(data_x, data_y)
            optimizer.zero_grad()
            NLL.backward(retain_graph=True)
            optimizer.step()
            if verbose:
                print(f'LL={NLL}, lr={lr}')
            else:
                if steps % 50 == 49:
                    print('.') # print dot and get to a new line
                else:
                    print('.', end="")  # print 50 dots in a row
        return NLL

##################### Analysis of Boston house prices ###############################
if __name__ == "__main__":
    def one(x: torch.Tensor):
        return x.pow(0)

    def pow1(x: torch.Tensor):
        return x

    def pow2(x: torch.Tensor):
        return x.pow(2)

 #%%
    df = pd.read_csv('boston_housing.csv')
    variables = df.columns

    data = df.to_numpy()
    # full model
    # x = data[:, :-1] # all but last
    y = np.log(data[:, -1]) # log-link for the last column: 'medv'

    # The full ordinary regression model has Rsq = 0.786
    # Preliminary analysis with Matlab shows that the following models are almost as good
    # lm1 = fitlm([crim nox rm dis rad tax ptratio b lstat], lmedv)
    # with RMSE = 0.191366, Rsq_adj = 0.780774, Rsq = 0.7851
    # and another even simpler:
    # lm2 = fitlm([crim nox rm dis rad ptratio b lstat], lmedv) % 0,4,5,7,8,9,11,12
    # with RMSE = 0.1949,  Rsq_adj = 0.773, Rq = 0.776

    # Fitting the simplest model: lm2
    x = df[['crim', 'nox', 'rm', 'dis', 'rad', 'ptratio', 'b', 'lstat']].to_numpy()
    # model_vars = [0,4,5,7,8,9,11,12]
    # x = data[:, model_vars]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    X = torch.cat((torch.ones(x.shape[0]).reshape(-1, 1), x), dim=1) # adding column of 1's for intercept
    XX_inv = torch.inverse(torch.matmul(X.transpose(0, 1), X))
    coefs = torch.matmul(torch.matmul(XX_inv, X.transpose(0, 1)), y.reshape(-1, 1))
    # these are the coefficients for the classic regression starting with intercept
    print([np.round(c.item(),5) for c in coefs])
    # for log of the price:
    # [4.02695, -0.01025, -0.84122, 0.10379, -0.0457, 0.00572, -0.04418, 0.00045, -0.0291]

    #%% functions operating on tensors (*, self.dim_regression)
    def h0(x): return torch.ones(x.shape[0]).reshape(-1,1)
    def h1(x): return x[:,0].reshape(-1,1)
    def h2(x): return x[:,1].reshape(-1,1)
    def h3(x): return x[:,2].reshape(-1,1)
    def h4(x): return x[:,3].reshape(-1,1)
    def h5(x): return x[:,4].reshape(-1,1)
    def h6(x): return x[:,5].reshape(-1,1)
    def h7(x): return x[:,6].reshape(-1,1)
    def h8(x): return x[:,7].reshape(-1,1)

    # multiple regression model: const + sum_j N(m_j, v_j)* x_j
    reg = GaussianMultRegression([h0, h1, h2, h3, h4, h5, h6, h7, h8])
    print(reg.coefs_classic(x, y))
    # should be the same as coefs above
    reg.fit(x,y)

