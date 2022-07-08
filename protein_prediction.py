#!/usr/bin/env python
# coding: utf-8
import numpy as np
import gzip
import pickle

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, BatchNormalization, Flatten, Input, concatenate
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


from sklearn.model_selection import train_test_split

from time import time
from timeit import default_timer as timer

#our deep CNN model
def get_model():

    LR = 0.00009
    drop_out = 0.7
    w_reg = regularizers.l2(0)
    windowSize = 19
    classSize = 8

    loss = 'categorical_crossentropy'
    conv1_input = Input(shape=(windowSize, 21), name='InputWindow')

    conv_1 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter1')(conv1_input)
    conv_1 = BatchNormalization(name='BN1')(conv_1)
    conv_1 = Dropout(drop_out)(conv_1)
    conv_2 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter2')(conv1_input)
    conv_2 = BatchNormalization(name='BN2')(conv_2)
    conv_2 = Dropout(drop_out)(conv_2)
    conv_3 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter3')(conv1_input)
    conv_3 = BatchNormalization(name='BN3')(conv_3)
    conv_3 = Dropout(drop_out)(conv_3)

    merge_1 = concatenate([conv_1, conv_2, conv_3], name='Network1')
    input_for_second = concatenate([conv1_input, merge_1], name='Network1-and-input')



    conv_4 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter1')(input_for_second)
    conv_4 = BatchNormalization(name='BN4')(conv_4)
    conv_4 = Dropout(drop_out)(conv_4)
    conv_5 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter2')(input_for_second)
    conv_5 = BatchNormalization(name='BN5')(conv_5)
    conv_5 = Dropout(drop_out)(conv_5)
    conv_6 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter3')(input_for_second)
    conv_6 = BatchNormalization(name='BN6')(conv_6)
    conv_6 = Dropout(drop_out)(conv_6)

    merge_2 = concatenate([conv_4, conv_5, conv_6], name='Network2')
    input_for_third = concatenate([conv1_input, merge_1, merge_2],name='Network1-Network2-and-input')



    conv_7 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter1')(input_for_third)
    conv_7 = BatchNormalization(name='BN7')(conv_7)
    conv_7 = Dropout(drop_out)(conv_7)
    conv_8 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter2')(input_for_third)
    conv_8 = BatchNormalization(name='BN8')(conv_8)
    conv_8 = Dropout(drop_out)(conv_8)
    conv_9 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter3')(input_for_third)
    conv_9 = BatchNormalization(name='BN9')(conv_9)
    conv_9 = Dropout(drop_out)(conv_9)

    merge_3 = concatenate([conv_7, conv_8, conv_9],name='Network3')
    input_for_4 = concatenate([conv1_input, merge_1, merge_2, merge_3],name='Network123-and-input')



    conv_10 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network4-filter1')(input_for_4)
    conv_10 = BatchNormalization(name='BN10')(conv_10)
    conv_10 = Dropout(drop_out)(conv_10)
    conv_11 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network4-filter2')(input_for_4)
    conv_11 = BatchNormalization(name='BN11')(conv_11)
    conv_11 = Dropout(drop_out)(conv_11)
    conv_12 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network4-filter3')(input_for_4)
    conv_12 = BatchNormalization(name='BN12')(conv_12)
    conv_12 = Dropout(drop_out)(conv_12)

    merge_4 = concatenate([conv_10, conv_11, conv_12],name='Network4')
    input_for_5 = concatenate([conv1_input, merge_1, merge_2, merge_3, merge_4],name='Network1234-and-input')



    conv_13 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network5-filter1')(input_for_5)
    conv_13 = BatchNormalization(name='BN13')(conv_13)
    conv_13 = Dropout(drop_out)(conv_13)
    conv_14 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network5-filter2')(input_for_5)
    conv_14 = BatchNormalization(name='BN14')(conv_14)
    conv_14 = Dropout(drop_out)(conv_14)
    conv_15 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network5-filter3')(input_for_5)
    conv_15 = BatchNormalization(name='BN15')(conv_15)
    conv_15 = Dropout(drop_out)(conv_15)

    merge_5 = concatenate([conv_13, conv_14, conv_15],name='Network5')

    merge_final = concatenate([merge_1, merge_2, merge_3,merge_4, merge_5], name='Final')

    flatten  = Flatten()(merge_final)
    first_dense = Dense(128, activation='relu', use_bias=True,  kernel_regularizer=w_reg, name='last')(flatten)
    first_dense = BatchNormalization(name='BN16')(first_dense)

    final_model_output = Dense(classSize, activation = 'softmax', name='softmax')(first_dense)

    m = Model(inputs=conv1_input, outputs=final_model_output)

    opt = Adam(lr=LR)
    m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

    print("\nHyper Parameters\n")
    print("Learning Rate: " + str(LR))
    print("Drop out: " + str(drop_out))
    print("Regularizers: " + str(w_reg.l2))
    print("\nLoss: " + loss + "\n")
    #m.summary()

    return m

def get_dataset():

    f = gzip.GzipFile('X_train_window19Middle-repeating-left-right0.npy.gz', "r")
    X_train_window = np.load(f)

    f = gzip.GzipFile('X_valid_window19Middle-repeating-left-right0.npy.gz', "r")
    X_valid_window  = np.load(f)

    X_train_window = np.concatenate((X_train_window, X_valid_window), axis = 0)
    print(X_train_window.shape)

    del X_valid_window
    f = gzip.GzipFile('cb513-window19Middle-repeating-left-right0.npy.gz', "r")
    X_valid_window  = np.load(f)

    x_train_final = X_train_window[:,:,35:56]
    y_train_final = X_train_window[:,:,22:30]

    x_valid_final = X_valid_window[:,:,35:56]
    y_valid_final = X_valid_window[:,:,22:30]

    print(x_train_final.shape, "training data")
    print(y_train_final.shape, "labels for training data")
    print(x_valid_final.shape, "validation data")
    print(y_valid_final.shape, "labels for training validation")

    y_train_final = y_train_final[:,0,:]
    print(y_train_final.shape)

    y_valid_final = y_valid_final[:,0,:]
    print(y_valid_final.shape)

    batch_size = 256 * 4
    print("Batch dim: " + str(batch_size))
    
    return (
        tf.data.Dataset.from_tensor_slices((x_train_final, y_train_final)).batch(batch_size).shuffle(x_train_final.shape[0], reshuffle_each_iteration=True),
        tf.data.Dataset.from_tensor_slices((x_valid_final, y_valid_final)).batch(batch_size)
    )

def main():

    nn_epochs = 150
    train_set, val_set = get_dataset()

    #callbacks
    filepath= "model_iridis" + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]

    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    print("Number of epochs: " + str(nn_epochs))

    
    
    # Open a strategy scope.
    with strategy.scope():
    	 # Everything that creates variables should be under the strategy scope.
   	 # In general this is only model construction & `compile()`.
       	 m = get_model()
    
    #training

    start_time = timer()

    history = m.fit(train_set, epochs=nn_epochs,
                validation_data=val_set, shuffle=True,  callbacks=callbacks_list)

    end_time = timer()
    print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")

    #saving results
    with open('model_iridis', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    main()
