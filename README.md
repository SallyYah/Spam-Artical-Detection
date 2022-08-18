# Spam-Artical-Detection


Outputs:

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

WARNING:tensorflow:Model was constructed with shape (None, 20) for input Tensor("embedding_11_input:0", shape=(None, 20), dtype=float32), but it was called on an input with incompatible shape (64, 200)
.
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
