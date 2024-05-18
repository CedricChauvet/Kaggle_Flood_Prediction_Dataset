#!/usr/bin/env python3
"""
Main file
"""
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def flood_prediction():
    # Selectionne dans le dataset les 2000 premiers échantillons
    datasets = np.loadtxt('./dataset/train.csv', delimiter=",", dtype=str)
    
    # La valeur a l'index 0 estr le titre de la donnée, on l'enleve
    X_train = np.array(datasets[1:,0:20],dtype=float)
    Y_train = np.array(datasets[1:,21],dtype=float)

    # validation des shape de X_train et Y_train
    print("Xtrain shape", X_train.shape)
    print("Y_train shape",Y_train.shape)
  
    # utiliser pour debug, verification de l'intégrité des données
    print("Y_train",Y_train[0:10])
    print("X_train",X_train[0])
        
    
    # Création du modèle
    network = Sequential([
    Dense(256, activation='tanh', input_shape=(20,)),
    Dense(256, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(1, activation='sigmoid')  # Utilisation de sigmoid pour la sortie dans [0, 1]
    ])

    # affiche la structur du reseau
    network.summary()



    # parametre d'optimisation adam
    alpha = 0.001 #learning rate
    beta1 = 0.9   # gradient momentum (il me semble) 
    beta2 = 0.999 # Root Mean Square Prop


    # Création des mini batches et definition du nombre d'epochs
    batch_size = 1024
    epochs = 10
    
    # Compilation du modèle, le mean square error est utilisé pour le cas ou le label est un flottant
    
    optim = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optim,
              loss='mean_squared_error',  # Utilisation de l'erreur quadratique moyenne
              metrics=['mean_absolute_error']) # metrics sert a donner la précision du modele

    network.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,verbose=True)

    # any idea, feature scaling? other??


flood_prediction()