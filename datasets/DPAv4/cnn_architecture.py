import os.path
import sys
import time
import importlib as importer
import numpy as np
import pickle
import random
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential

import metrics


### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def shuffle_data(profiling_x, label_y):
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100,
                early_stop=None):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    # early_stop = EarlyStopping(patience=6, restore_best_weights=True)
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check

    # Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[ASCAD_0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[ASCAD_0], X_test.shape[1], 1))

    callbacks = [save_model]
    if early_stop is not None:
        callbacks.append(early_stop)
    # callbacks.append(early_stop)

    history = model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=9),
                        validation_data=(X_test, to_categorical(Y_test, num_classes=9)), batch_size=batch_size,
                        verbose=1, epochs=epochs, callbacks=callbacks)
    return history


def hw(input):
    return bin(int(input)).count("1")


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

def run(model_name, metrics_names, loss_func, nb_epochs, batch_size, learning_rate, reshape_x_attack, early_stopping,
        early_stopping_patience, *args, **kwargs):
    # Our folders
    root = "./datasets/DPAv4/"
    DPAv4_data_folder = root + "training_data/"
    DPAv4_trained_models_folder = "results/DPAv4/trained_models/"
    history_folder = "results/DPAv4/training_history/"
    predictions_folder = "results/DPAv4/model_predictions/"
    input_size = 4000
    nb_traces_attacks = 30
    nb_attacks = 100
    real_key = np.load(DPAv4_data_folder + "key.npy")
    mask = np.load(DPAv4_data_folder + "mask.npy")
    att_offset = np.load(DPAv4_data_folder + "attack_offset_dpav4.npy")

    start = time.time()

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (np.load(
        DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy')), (
                                                                                        np.load(
                                                                                            DPAv4_data_folder + 'attack_traces_dpav4.npy'),
                                                                                        np.load(
                                                                                            DPAv4_data_folder + 'attack_labels_dpav4.npy')), (
                                                                                        np.load(
                                                                                            DPAv4_data_folder + 'profiling_plaintext_dpav4.npy'),
                                                                                        np.load(
                                                                                            DPAv4_data_folder + 'attack_plaintext_dpav4.npy'))

    # Shuffle data
    (X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    Y_profiling = np.vectorize(hw)(Y_profiling)
    Y_attack = np.vectorize(hw)(Y_attack)
    X_profiling = X_profiling.astype('float32')
    X_attack = X_attack.astype('float32')

    # Standardization + Normalization (between ASCAD_0 and 1)
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
                                                                  learning_rate=learning_rate, input_size=input_size)
    model_name = "{:s}_{:s}_e{:d}b{:d}DPA-contest_v4sadas.h5".format(model_name, loss_func, nb_epochs, batch_size)

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")

    if early_stopping:
        early_stop = EarlyStopping(patience=early_stopping_patience, monitor='val_loss', restore_best_weights=True)
    else:
        early_stop = None

    history = train_model(X_profiling[:4000], Y_profiling[:4000], X_profiling[4000:], Y_profiling[4000:], model,
                          DPAv4_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size,
                          early_stop=early_stop)
    end = time.time()

    print('Execution Time = %d' % (end - start))

    print("\n############### Training Done #################\n")

    # Save the DL metrics (loss and accuracy)
    with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    #################################################
    #################################################

    ####               Prediction              ######

    #################################################
    #################################################

    print("\n############### Starting Predictions #################\n")

    predictions = model.predict(X_attack)
    # print(metrics.ratio_guess(Y_attack, predictions))
    print("\n############### Predictions Done #################\n")

    np.save(predictions_folder + 'predictions_' + model_name + '.npy', predictions)

    #################################################
    #################################################

    ####            Perform attacks            ######

    #################################################
    #################################################

    print("\n############### Starting Attack on Test Set #################\n")

    exploit_pred = importer.import_module("datasets.DPAv4.exploit_pred")

    avg_rank = exploit_pred.perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key,
                                            mask=mask,
                                            offset=att_offset, byte=0, filename=model_name)

    se = nb_epochs

    if early_stopping:
        se = early_stop.stopped_epoch

    np.save("results/DPAv4/plot_data/" + "b{:d}_se{:d}.npy".format(batch_size, se), avg_rank)

    print("\n############### Attack on Test Set Done #################\n")
