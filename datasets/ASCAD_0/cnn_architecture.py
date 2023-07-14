import os.path
import sys
import time
import pickle
import h5py
import random
import numpy as np
import importlib as importer
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.losses import  categorical_crossentropy
from sklearn import preprocessing

import tensorflow as tf

import sys

### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD
from tensorflow.python.keras import Sequential


def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return


def hw(input):
    return bin(int(input)).count("1")

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


#### ASCAD helper to load profiling and attack data (traces and labels) (source : https://github.com/ANSSI-FR/ASCAD)
# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['plaintext'])


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3, early_stop = None):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    #Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[ASCAD_0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[ASCAD_0], X_test.shape[1], 1))

    # One Cycle Policy
   # lr_manager = OneCycleLR(max_lr=max_lr, end_percentage=ASCAD_0.2, scale_percentage=ASCAD_0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)

    callbacks=[save_model]
    if early_stop is not None:
        callbacks.append(early_stop)
           
    history = model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=9), validation_data=(X_test, to_categorical(Y_test, num_classes=9)), batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks)
    return history

def timeFunction(y_true, y_pred, function):
    timesList = []
    for i in range(1000):
        startTime = time.time()
        for j in range(100):
            temp = function(y_true, y_pred)
        timesList.append((time.time() - startTime)/100)

    return np.mean(timesList), np.std(timesList)

#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

def run(model_name, metrics_names, loss_func, nb_epochs, batch_size, learning_rate, reshape_x_attack, early_stopping,
        early_stopping_patience, *args, **kwargs):

    # Our folders
    root = "./datasets/ASCAD_0/"
    ASCAD_data_folder = root + "training_data/"
    ASCAD_trained_models_folder = "results/ASCAD_0/trained_models/"
    history_folder = "results/ASCAD_0/training_history/"
    predictions_folder = "results/ASCAD_0/model_predictions/"

    # Choose the name of the model
    input_size = 700
    nb_traces_attacks = 300
    nb_attacks = 100
    real_key = np.load(ASCAD_data_folder + "key.npy")

    start = time.time()

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(
        ASCAD_data_folder + "ASCAD.h5", load_metadata=True)

    # Shuffle data
    (X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    Y_profiling = np.vectorize(hw)(Y_profiling)
    Y_attack = np.vectorize(hw)(Y_attack)
    X_profiling = X_profiling.astype('float32')
    X_attack = X_attack.astype('float32')


    # Standardization and Normalization (between ASCAD_0 and 1)
    scaler = preprocessing.StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)
    if reshape_x_attack:
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    #################################################
    #################################################

    ####                Training               ######

    #################################################
    #################################################

    # Choose your model
    loss_function = getattr(importer.import_module("metrics"), loss_func)
    metrics = list(map(lambda x: getattr(importer.import_module("metrics"), x), metrics_names))
    model = getattr(importer.import_module("models"), model_name)(loss_function=loss_function, metrics=metrics,
                                                                  learning_rate=learning_rate, input_size=input_size,
                                                                  classes=9)
    model_name = "{:s}_{:s}_e{:d}_b{:d}.h5".format(model_name, loss_func,nb_epochs, batch_size, )

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")

    if early_stopping:
        early_stop = EarlyStopping(patience=early_stopping_patience, monitor='val_loss', restore_best_weights=True)
    else:
        early_stop = None

    # Record the metrics
    history = train_model(X_profiling[:45000], Y_profiling[:45000], X_profiling[45000:], Y_profiling[45000:], model,
                          ASCAD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size, early_stop=early_stop,
                          max_lr=learning_rate)

    end = time.time()
    print('Temps execution = %d' % (end - start))

    print("\n############### Training Done #################\n")

    # Save the metrics
    with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    #################################################
    #################################################

    ####               Prediction              ######

    #################################################
    #################################################

    print("\n############### Starting Predictions #################\n")

    predictions = model.predict(X_attack)

    print("\n############### Predictions Done #################\n")

    np.save(predictions_folder + 'predictions_' + model_name + '.npy', predictions)

    #################################################
    #################################################

    ####            Perform attacks            ######

    #################################################
    #################################################

    print("\n############### Starting Attack on Test Set #################\n")

    exploit_pred = importer.import_module("datasets.ASCAD_0.exploit_pred")

    avg_rank = exploit_pred.perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2,
                               filename=model_name)

    print(np.where(avg_rank <= 0))

    se = nb_epochs
    if early_stopping:
        se = early_stop.stopped_epoch

    np.save("results/ASCAD_0/plot_data/b{:d}_se{:d}.npy".format(batch_size, se), avg_rank)

    print("\n############### Attack on Test Set Done #################\n")
