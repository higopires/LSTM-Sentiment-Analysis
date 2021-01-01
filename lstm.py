from numpy.random import seed
seed(42)
import tensorflow as tf
tf.random.set_seed(42)

with open("dataset.txt") as f1:
    lines = f1.readlines()

x = []

for value in lines:
    temp = value.split('\t')
    x.append(temp[0])

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=2500,split=' ')
tokenizer.fit_on_texts(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X)

import pandas as pd

result = pd.read_csv('dataset_transformers.csv')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(result['label'])

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(2500,128,input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(300, recurrent_dropout=0.2,dropout=0.2))
model.add(Dense(3,activation='softmax'))

model.summary()

custom_adam = tf.keras.optimizers.Adam(learning_rate=1e-05, epsilon=1e-08)

model.compile(loss="sparse_categorical_crossentropy",optimizer=custom_adam,metrics=['accuracy'])

from sklearn.model_selection import StratifiedKFold
import numpy as np

accuracies = []

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    model.fit(X_train,y_train,epochs=10,verbose=1,batch_size=32)
    
    accuracies.append(model.evaluate(X_test,y_test)[1])


accuracies = np.array(accuracies)

print('Mean accuracy: ', np.mean(accuracies, axis=0))
print('Std for accuracy: ', np.std(accuracies, axis=0))