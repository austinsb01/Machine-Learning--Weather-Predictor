import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.drop(columns=['temp'])  
    y = data['temp']  
    return X, y

def split_data(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=1):
    temp_size = 1 - train_size
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=temp_size, random_state=random_state)
    
    val_proportion = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_proportion, random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test
