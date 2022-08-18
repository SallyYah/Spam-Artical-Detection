# Spam-Artical-Detection

**Import statements**
```
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
```

**Data cleaning**
```
train_path='train.csv'
test_path='test.csv'

# Read from .csv file
df_train = pd.read_csv(train_path).astype(str)
df_test = pd.read_csv(test_path).astype(str)
temp=pd.read_csv('labels.csv').astype(int)

# Add extra column to test set for labels
df_test['label'] = (temp['label'].values).astype(int)


#df_train = df_train.sample(frac=0.7)

x_train = df_train['text'].values
y_train = (df_train['label'].values).astype(int)

x_test = df_test['text'].values
y_test = df_test['label'].values
```

**Data processing**

Tokenize input text

Generate a word dictionnary with key-value pair (word:occurence) word is represented by integer numbers starting from 1 ordered descending by word frequency of apperence (most to least).

ex: {1:50, 2:10, 3:9}

```
tokenizer = Tokenizer(lower=None)
tokenizer.fit_on_texts(x_train)
word_dict = tokenizer.index_word
```

Using word_dict as reference, change text to a sequence

Example:

x_train = "Hello World hi hi" -> x_train_seq = [1,2,3,3]

```
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)


MAX_LEN = -1
# Looking for max line length in x_train_seq
"""
for line in x_train_seq:
    MAX_LEN = max(MAX_LEN, len(line))
"""
# MAX_LEN = 24512
```

For quicker training purpose, MAX_LEN used for each sequence is 200.

Will add padding (0) to end of each line with length < 200

```
X_train_pad = pad_sequences(x_train_seq, maxlen=200, padding='post')
X_test_pad = pad_sequences(x_test_seq, maxlen=200, padding='post')
```

**Model training**
```
# Total number of distinct word in word_dict is 279687
# len(dict1) -> 279687

# Fit LSTM model
model = Sequential()
model.add(Embedding(input_dim=279687+1, input_length=20, output_dim=20))
model.add(LSTM(400))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


summary = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))
```

Outputs:

```
Model: "sequential_11"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_11 (Embedding)     (None, 20, 20)            5593760   
_________________________________________________________________
lstm_11 (LSTM)               (None, 400)               673600    
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 401       
=================================================================
Total params: 6,267,761
Trainable params: 6,267,761
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/5
WARNING:tensorflow:Model was constructed with shape (None, 20) for input Tensor("embedding_11_input:0", shape=(None, 20), dtype=float32), but it was called on an input with incompatible shape (64, 200).
WARNING:tensorflow:Model was constructed with shape (None, 20) for input Tensor("embedding_11_input:0", shape=(None, 20), dtype=float32), but it was called on an input with incompatible shape (64, 200).
325/325 [==============================] - ETA: 0s - loss: 0.3767 - accuracy: 0.8427  WARNING:tensorflow:Model was constructed with shape (None, 20) for input Tensor("embedding_11_input:0", shape=(None, 20), dtype=float32), but it was called on an input with incompatible shape (None, 200).
325/325 [==============================] - 482s 1s/step - loss: 0.3767 - accuracy: 0.8427 - val_loss: 1.4548 - val_accuracy: 0.6281
Epoch 2/5
325/325 [==============================] - 452s 1s/step - loss: 0.1659 - accuracy: 0.9463 - val_loss: 1.7573 - val_accuracy: 0.6102
Epoch 3/5
325/325 [==============================] - 450s 1s/step - loss: 0.1197 - accuracy: 0.9612 - val_loss: 2.0175 - val_accuracy: 0.6127
Epoch 4/5
325/325 [==============================] - 443s 1s/step - loss: 0.1821 - accuracy: 0.9343 - val_loss: 1.8795 - val_accuracy: 0.6073
Epoch 5/5
325/325 [==============================] - 425s 1s/step - loss: 0.2722 - accuracy: 0.8864 - val_loss: 1.7562 - val_accuracy: 0.5492
```
