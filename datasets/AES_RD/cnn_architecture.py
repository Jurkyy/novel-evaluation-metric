import os.path
import sys
import time
import pickle
import numpy as np
import importlib as importer
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    np.random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3, early_stop=None):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    # if input_layer_shape[1] != len(X_profiling[ASCAD_0]):
    #     print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[ASCAD_0])))
    #     sys.exit(-1)
    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # One Cycle Policy
    #lr_manager = OneCycleLR(max_lr=max_lr, end_percentage=ASCAD_0.2, scale_percentage=ASCAD_0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)

    callbacks=[save_model]
    #, lr_manager
    if early_stop is not None:
        callbacks.append(early_stop)
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=9), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=9)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
    return history

def hw(input):
    return float(bin(int(input)).count("1"))

#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
def run(model_name, metrics_names, loss_func, nb_epochs, batch_size, learning_rate, early_stopping,
        early_stopping_patience, *args, **kwargs):
    root = "./datasets/AES_RD/"
    AESRD_data_folder = root + "training_data/"
    AESRD_trained_models_folder = "results/AES_RD/trained_models/"
    history_folder = "results/AES_RD/training_history/"
    predictions_folder = "results/AES_RD/model_predictions/"

    # Choose the name of the model

    input_size = 3500
    nb_traces_attacks = 15
    nb_attacks = 100
    real_key = np.load(AESRD_data_folder + "key.npy")

    start = time.time()

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (np.load(
        AESRD_data_folder + 'profiling_traces_AES_RD.npy'), np.load(
        AESRD_data_folder + 'profiling_labels_AES_RD.npy')), (np.load(AESRD_data_folder + 'attack_traces_AES_RD.npy'),
                                                              np.load(
                                                                  AESRD_data_folder + 'attack_labels_AES_RD.npy')), (
                                                                                    np.load(
                                                                                        AESRD_data_folder + 'profiling_plaintext_AES_RD.npy'),
                                                                                    np.load(
                                                                                        AESRD_data_folder + 'attack_plaintext_AES_RD.npy'))

    # Shuffle data
    (X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    Y_profiling = np.vectorize(hw)(Y_profiling)
    Y_attack = np.vectorize(hw)(Y_attack)
    X_profiling = X_profiling.astype('float32')
    X_attack = X_attack.astype('float32')

    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    #################################################
    #################################################

    ####                Training               ######

    #################################################
    #################################################

    # Choose your model
    loss_function = getattr(importer.import_module("metrics"), loss_func)
    metrics = list(map(lambda x: getattr(importer.import_module("metrics"), x), metrics_names))
    model = getattr(importer.import_module("models"), model_name)(loss_function=loss_function, metrics=metrics, learning_rate=learning_rate, input_size=input_size, classes=9)
    model_name = "AES_RD_{:s}_e{:d}_b{:d}.h5".format(loss_func, nb_epochs, batch_size)

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")

    if early_stopping:
        early_stop = EarlyStopping(patience=early_stopping_patience, monitor='val_loss', restore_best_weights=True)
    else:
        early_stop = None
    # Record the metrics
    history = train_model(X_profiling[:20000], Y_profiling[:20000], X_profiling[20000:], Y_profiling[20000:], model,
                          AESRD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size,
                          max_lr=learning_rate, early_stop=early_stop)

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
    exploit_pred = importer.import_module("datasets.AES_RD.exploit_pred")

    avg_rank = exploit_pred.perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=0,
                               filename=model_name)

    np.save("results/AES_RD/plot_data/" + model_name.replace('.h5', '.npy'), avg_rank)

    print("\n############### Attack on Test Set Done #################\n")
