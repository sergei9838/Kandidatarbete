import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

#torch.manual_seed(30) # <-- if seed is wanted
N = 1000
x = torch.linspace(-3, 5, N)
y = (torch.randn(N)+-0.5) * x + (2+torch.randn(N))

# Plot the data points
plt.scatter(x, y)
plt.show()

# Number of locations of measure
M = 17

# Measure for slope (a) and intercept (b) of linear model
a = pm.Measure(torch.linspace(-4, 4, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-2, 6, M), torch.ones(M) / M)

# Linear regression model
def regression_model(x,list):
     return list[0]*x+list[1]

measures = [a,b]

# Instance of optimizer
opt = pm.Optimizer(measures, "KDEnll", lr = 0.1)
# Call to minimizer
new_mes=opt.minimize([x,y],regression_model,max_epochs=2500,verbose = True, print_freq=100, smallest_lr=1e-10)
# Visualize measures and gradient
new_mes[0].visualize()
plt.show()
new_mes[1].visualize()
plt.show()

check=pm.Check(opt,regression_model,x,y,normal=True)
check.check()
