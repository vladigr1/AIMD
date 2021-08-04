import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

beta = [ 0.7, 0.8 ]
alpha = [ 0.8, 0.7 ]
capacity = 1
x0 = [ 1, 0]

convergence_point = [alpha[i]/(1- beta[i]) for i in range(0,len(alpha))]
convergence_point = np.array(convergence_point)
convergence_point = convergence_point / np.linalg.norm(convergence_point,1)

B = np.array([
    [beta[0], 0],
    [0, beta[1]],
])

alpha_vect = [[a] for a in alpha]
alpha_vect = np.array(alpha_vect)
D = [1 - b for b in beta]
D = np.array(D)
E = np.multiply(alpha_vect,D)
E *= 1 / np.sum(alpha)

# Matrix Iterations
A = B + E
print(f"eigen values: {[round(x,3) for x in np.linalg.eigvals(A)]}") 


xdata = []
ydata = []
color = []

x = x0
x = np.array(x)
xdata.append(x[0])
ydata.append(x[1])
color.append(0)

for i in range(0,10):
    x = A @ x
    xdata.append(x[0])
    ydata.append(x[1])
    color.append(color[-1]+1)

# plot the data
xdata = np.array(xdata)
ydata = np.array(ydata)
distance = abs(xdata[1:] - xdata[:-1]) + abs(ydata[1:] - ydata[:-1])
distance = np.concatenate(([0],distance ), axis=0)
q = distance[2:] / distance[1:-1]
q = np.concatenate( ([0,0], q) , axis=0)
data = np.stack((xdata, ydata, distance,q), axis=1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel='x', ylabel='y',
       title='AIMD Iterations')
plt.plot([0,1],[1,0],'--k')
points = plt.scatter(xdata, ydata,c= color, label="iter")
plt.scatter(convergence_point[0], convergence_point[1], c= "r",marker= "+",  label="conv. p")
plt.colorbar(points)
plt.legend()
plt.show()

# AIMD steps Iterations
x = x0
xdata = []
ydata = []
color = []

x = np.array(x)
x = x.reshape(-1,1)
xdata.append(x[0][0])
ydata.append(x[1][0])
color.append(0)

for i in range(0,10):
    # md
    x = B @ x
    xdata.append(x[0][0])
    ydata.append(x[1][0])
    color.append(color[-1]+1)

    # ai
    adding_factor = ( capacity - np.sum(x) ) / np.sum(alpha_vect)
    x += adding_factor * alpha_vect
    xdata.append(x[0][0])
    ydata.append(x[1][0])
    color.append(color[-1]+1)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel='x', ylabel='y',
       title='AIMD steps by steps Iterations')
plt.plot([0,1],[1,0],'--k')
points = plt.scatter(xdata, ydata,c= color, label="iter")
plt.scatter(convergence_point[0], convergence_point[1], c= "r",marker= "+",  label="conv. p")
plt.colorbar(points)
plt.legend()
plt.show()

# AIMD steps Iterations with change of capacity
x = x0
xdata = []
ydata = []
color = []

x = np.array(x)
x = x.reshape(-1,1)
xdata.append(x[0][0])
ydata.append(x[1][0])
color.append(0)

for i in range(0,5):
    # md
    x = B @ x
    xdata.append(x[0][0])
    ydata.append(x[1][0])
    color.append(color[-1]+1)

    # ai
    if i < 4:
        adding_factor = ( capacity - np.sum(x) ) / np.sum(alpha_vect)
        x += adding_factor * alpha_vect
        xdata.append(x[0][0])
        ydata.append(x[1][0])
        color.append(color[-1]+1)

capacity = 2 
adding_factor = ( capacity - np.sum(x) ) / np.sum(alpha_vect)
x += adding_factor * alpha_vect
xdata.append(x[0][0])
ydata.append(x[1][0])
color.append(color[-1]+1)

for i in range(0,5):
    # md
    x = B @ x
    xdata.append(x[0][0])
    ydata.append(x[1][0])
    color.append(color[-1]+1)

    # ai
    adding_factor = ( capacity - np.sum(x) ) / np.sum(alpha_vect)
    x += adding_factor * alpha_vect
    xdata.append(x[0][0])
    ydata.append(x[1][0])
    color.append(color[-1]+1)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set(xlabel='x', ylabel='y',
       title='AIMD steps Iter. with change of cap.')
plt.plot([0,1],[1,0],'--k')
plt.plot([0,2],[2,0],'--k')
points = plt.scatter(xdata, ydata,c= color, label="iter")
plt.scatter([convergence_point[0],convergence_point[0]*capacity] , [convergence_point[1], convergence_point[1]* capacity], c= "r",marker= "+",  label="conv. p")
plt.colorbar(points)
plt.legend()
plt.show()