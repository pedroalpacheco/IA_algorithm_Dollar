#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:39:05 2019

@author: PedroAlPAcheco - pedro.pacheco.a@gmail.com
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
from datetime import datetime

oque = "CoverteDolarPReal"

#Pega data e hora
agora = datetime.now()
hoje = str(agora)

def espaco():
    print("==============================================")
    print("                                              ")


"""Dados para treino treinando os dados"""
dolar_q    = np.array([2,4,6,9,12,24,35,66,69,76],  dtype=float)
real_a = np.array([7.84,15.68,23.52,35.28,47.04,94.08,137.2,258.72,270.48,297.92],  dtype=float)


for i,c in enumerate(dolar_q):
  print("{} degrees Dolares = {} degrees Reais".format(c, real_a[i]))
  
espaco()
  
"""Criando o modelo"""
l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 

espaco()

"""Montando uma camada de modelo"""
model = tf.keras.Sequential([l0])

espaco()

"""Compilando o modelo"""
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
espaco()


"""Treinando o modelo"""
history = model.fit(dolar_q, real_a, epochs=500, verbose=False)
print("Finalizado o treino do modelo")
espaco()

"""Estatistica do treino"""
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.savefig(oque + '-' + hoje +'.png')    

espaco()

"""Previsão de valores
Formula converte Dolar para real 
Cotação : 1$ = R$ 3,92 (18/04/2019)
Algoritmo IA previu:

dolar * valrCotação = RealR$    
    
"""

print(model.predict([100]))

"""Pegando a equação gerada"""
print("As variaveis da camada: {}".format(l0.get_weights()))