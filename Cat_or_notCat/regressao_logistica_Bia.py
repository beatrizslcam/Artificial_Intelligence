import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import pandas as pd
import cv2
import glob
from utils import salva_imagem_com_predicao

arquivos_de_gatos = "./data/train/cat/*.png"
arquivos_nao_gatos = "./data/train/noncat/*.png"

X = []
Y = []


for arquivo_de_gato in glob.glob(arquivos_de_gatos):
    imagem = cv2.imread(arquivo_de_gato)
    imagem = np.reshape(imagem, (64*64*3))
    print(imagem.shape)
    X.append(imagem)
    Y.append(1)

for arquivo_nao_gato in glob.glob(arquivos_nao_gatos):
    print(arquivo_nao_gato)
    imagem = cv2.imread(arquivo_nao_gato)
    imagem = np.reshape(imagem, (64*64*3))
    X.append(imagem)
    Y.append(0)

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape)
print(Y.shape)


# normalizing x
X = X / 255

X = np.insert(X, obj=0, values=1, axis=1)
Y = np.expand_dims(Y, axis=1)

print(X.shape)
print(Y.shape)

print("Fotos com gatos ={}, fotos sem gatos={}".format(
    np.sum(Y == 1), np.sum(Y != 1)))


theta = np.zeros((X.shape[1], 1))+0.1
alpha = 0.01
it = 500
J = np.zeros((it, 1))
acuracia = np.zeros((it, 1))
m = len(Y)
H_theta = np.ones((X.shape[0], 1))


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


for i in range(it):
    Z = np.dot(X, theta)
    H_theta = sigmoid(Z)
    J[i] = 1/m * np.sum(- Y * np.log(H_theta) - (1-Y) * np.log(1 - H_theta))
    e = H_theta - Y
    theta = theta - (alpha * (np.dot(X.T, e)))
    predicao = H_theta >= 0.5
    predicao = np.around(H_theta)
    acuracia = np.sum(Y == predicao)/len(Y)


for i in range(0, len(X)):
    tmp = X[i, 1:]*255
    tmp = np.reshape(tmp, (64, 64, 3))
    tmp = np.uint8(tmp)
    salva_imagem_com_predicao(tmp,
                              "resultados/imagem_{}.png".format(i),
                              H_theta[i, 0])

plt.figure()
plt.title("Loss X Updates")
plt.xlabel('iteracoes')
plt.ylabel('J')
plt.plot(acuracia)
plt.show()
