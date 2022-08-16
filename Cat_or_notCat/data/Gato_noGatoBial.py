import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import pandas as pd
import cv2
import glob
from utils import salva_imagem_com_predicao

#arquivos_de_gatos = "./data/train/cat/*.png"
#arquivos_nao_gatos = "./data/train/noncat/*.png"

arquivos_de_gatos = "./data/train/testando/cat1/*.png"
arquivos_nao_gatos = "./data/train/testando/noncat1/*.png"


X = []
Y = []


def inicial_for_validation(theta, alpha, it):
    X1 = []
    Y1 = []

    arquivos_de_gatos1 = "./data/train/testando/cat2/*.png"
    arquivos_nao_gatos1 = "./data/train/testando/noncat2/*.png"
    for arquivo_de_gato1 in glob.glob(arquivos_de_gatos1):
        imagem = cv2.imread(arquivo_de_gato1)
        imagem = np.reshape(imagem, (64*64*3))
        X1.append(imagem)
        Y1.append(1)

    for arquivo_nao_gato1 in glob.glob(arquivos_nao_gatos1):
        imagem = cv2.imread(arquivo_nao_gato1)
        imagem = np.reshape(imagem, (64*64*3))
        X1.append(imagem)
        Y1.append(0)

    X1 = np.asarray(X1)
    Y1 = np.asarray(Y1)
    X1 = X1 / 255

    X1 = np.insert(X1, obj=0, values=1, axis=1)
    Y1 = np.expand_dims(Y1, axis=1)

    print("Fotos com gatos ={}, fotos sem gatos={}".format(
        np.sum(Y == 1), np.sum(Y != 1)))

    J = np.zeros((it, 1))

    acuracia = np.zeros((it, 1))

    m = len(Y)

    H_theta = np.ones((X.shape[0], 1))
    for i in range(it):
        # equação da reta
        Z = np.dot(X, theta)

        # função de ativação
        H_theta = sigmoid(Z)

        # Loss
        J[i] = 1/m * np.sum(- Y * np.log(H_theta) -
                            (1-Y) * np.log(1 - H_theta))

        e = H_theta - Y

    predicao = H_theta >= 0.5
    predicao = np.around(H_theta)
    acuracia = np.sum(Y == predicao)/len(Y)
    for i in range(0, len(X)):
        tmp = X1[i, 1:]*255
        tmp = np.reshape(tmp, (64, 64, 3))
        tmp = np.uint8(tmp)
        salva_imagem_com_predicao(tmp,
                                  "resultados2/i{}.png".format(
                                      i),
                                  H_theta[i, 0])


for arquivo_de_gato in glob.glob(arquivos_de_gatos):
    imagem = cv2.imread(arquivo_de_gato)
    imagem = np.reshape(imagem, (64*64*3))
    X.append(imagem)
    Y.append(1)

for arquivo_nao_gato in glob.glob(arquivos_nao_gatos):
    imagem = cv2.imread(arquivo_nao_gato)
    imagem = np.reshape(imagem, (64*64*3))
    X.append(imagem)
    Y.append(0)


X = np.asarray(X)
Y = np.asarray(Y)


# normalizing x
X = X / 255

X = np.insert(X, obj=0, values=1, axis=1)
Y = np.expand_dims(Y, axis=1)


print("Fotos com gatos ={}, fotos sem gatos={}".format(
    np.sum(Y == 1), np.sum(Y != 1)))

# criando theta com sendo uma coluna com o numero de linha de X
theta = np.zeros((X.shape[1], 1))+0.00001

alpha = 0.0009991
it = 100000

J = np.zeros((it, 1))

acuracia = np.zeros((it, 1))

m = len(Y)

H_theta = np.ones((X.shape[0], 1))


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


# treino
for i in range(it):
    # equação da reta
    Z = np.dot(X, theta)

    # função de ativação
    H_theta = sigmoid(Z)

    # Loss
    J[i] = 1/m * np.sum(- Y * np.log(H_theta) - (1-Y) * np.log(1 - H_theta))

    e = H_theta - Y

    # atualização do theta
    theta = theta - (alpha * (1/m)*(np.dot(X.T, e)))

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

plt.title("Loss X Updates")
plt.xlabel('iteracoes')
plt.ylabel('J')
plt.plot(J)

plt.show()

inicial_for_validation(theta, alpha, it)
print("fim")
