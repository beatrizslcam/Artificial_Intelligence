import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import pandas as pd


# reading the archieves
wine = pd.read_csv('winequality-red.csv', delimiter=';')

# getting the x and y values
x = wine[wine.columns[:-1]].to_numpy()
y = wine[wine.columns[-1:]].to_numpy()

# normalizing x
X = np.divide(x-np.mean(x, axis=0), np.std(x, axis=0))

m = len(wine.columns[:-1])


# adding a column with 1
X = np.insert(X, 0, 1, axis=1)


theta = np.zeros(m+1).reshape(m+1, 1)


alpha = 0.0001
it = 500
J = np.zeros(it)
e = np.zeros(it)
print(np.shape(theta))
for i in range(it):
    H = np.dot(X, theta)
    E = H - y
    temp = (np.dot(E.T, E))
    J = np.divide(temp, (2*m))
    e[i] = J
    theta = theta - (alpha * (np.dot(X.T, E)))

plt.figure()
plt.title("Loss X Updates")
plt.xlabel('iteracoes')
plt.ylabel('J')
plt.plot(e)
plt.show()
