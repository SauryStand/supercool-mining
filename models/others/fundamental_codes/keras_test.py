#-*- coding: utf-8 -*-
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
# Stacking layers
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorial_crossentropy', optimizer='sgd', metrics=['accuracy'])

