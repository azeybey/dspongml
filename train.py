#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:07:18 2022

@author: zeybey
"""
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression


data = pd.read_csv('pong_data_train.csv')

X = data[['ball_x', 'ball_y']]

Y = data['paddle_y']

model = LinearRegression()

model.fit(X, Y)

dump(model, 'mymodel.joblib')