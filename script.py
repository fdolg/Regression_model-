import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import r2_score


#Setting dataset 
data_set = pd.read_csv('admissions_data.csv')
#print(data_set.head())

#Spliting labels and features 

labels = data_set.iloc[:,-1]
features = data_set.iloc[:,0:-1]

# Spliting train and test data 
features_train, features_test, labels_train, labels_test = train_test_split(features,labels, test_size=0.3, random_state=64)

# Standarization data 
ct = ColumnTransformer([("numeric", StandardScaler(),features.columns)], remainder="passthrough")

features_train_scaler = pd.DataFrame(ct.fit_transform(features_train))
features_test_scales = pd.DataFrame(ct.fit_transform(features_test))



#Designing neural network 
def create_NN() :
    model = Sequential()
    
    # Adding and setting input layer 
    input_layer = layers.InputLayer(input_shape = (features.shape[1],))
    model.add(input_layer)
    
    # Adding hidden Layer 
    model.add(layers.Dense(16, activation = 'relu'))
    
    #Adding output Layer 
    model.add(layers.Dense(1))
    
    #Summary 
    print(model.summary())
    
    # Adam optimizer 
    opt = Adam(learning_rate = 0.01)
    model.compile(loss = 'mse', metrics =['mae'], optimizer = opt)
    #Training
    es = EarlyStopping(monitor = 'val_loss', mode= 'min', verbose = 1, patience = 20)
    history = model.fit(features_train_scaler,labels_train, epochs = 50,
                        batch_size =15, verbose=0,
                        validation_split = 0.2, callbacks=[es])
    
    #Evaluating 
    res_mse, res_mae = model.evaluate(features_test_scales, labels_test, verbose = 0)
    
    return res_mse, res_mae, history, model

res_mse, res_mae, history, model = create_NN()

print(res_mse,res_mae)


# Plotting performance 
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

 # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()


#Calculating coefficient of determination 
predicted_values = model.predict(features_test_scales)
print(r2_score(labels_test,predicted_values))
