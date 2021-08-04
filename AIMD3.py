import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

beta = [ 0.1, 0.5, 0.9 ]
alpha = [ 0.2, 0.3 , 0.4]
capacity = 2
x0 = [ 2, 0, 0 ]

convergence_point = [alpha[i]/(1- beta[i]) for i in range(0,len(alpha))]
convergence_point = np.array(convergence_point)
convergence_point = convergence_point / np.linalg.norm(convergence_point,1)

B = np.zeros((len(beta), len(beta)))
for i in range(0,len(beta)):
    B[i,i] = beta[i]

alpha_vect = [[a] for a in alpha]
alpha_vect = np.array(alpha_vect)
D = [1 - b for b in beta]
D = np.array(D)
E = np.multiply(alpha_vect,D)
E *= 1 / np.sum(alpha)

# Matrix Iterations
A = B + E
print(np.linalg.eigvals(A))

xdata = []
ydata = []
zdata = []
color = []

x = x0
x = np.array(x)
xdata.append(x[0])
ydata.append(x[1])
zdata.append(x[2])
color.append(0)

for i in range(0,10):
    x = A @ x
    xdata.append(x[0])
    ydata.append(x[1])
    zdata.append(x[2])
    color.append(color[-1]+1)



# plot the data
xdata = np.array(xdata)
ydata = np.array(ydata)
zdata = np.array(zdata)
distance = abs(xdata[1:] - xdata[:-1]) + abs(ydata[1:] - ydata[:-1]) + abs(zdata[1:] - zdata[:-1])
distance = np.concatenate(([1],distance ), axis=0)
q = distance[2:] / distance[1:-1]
q = np.concatenate( ([0,0], q) , axis=0)
data = np.stack((xdata, ydata, zdata, distance,q), axis=1)
df = pd.DataFrame(data, columns=['x', 'y', 'z', 'd', 'q'])
df.to_latex("./data3.tex")