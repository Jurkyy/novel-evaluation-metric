import os.path
import time
import pickle
import importlib as importer
import sys
import random
import numpy as np
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, AveragePooling1D, Flatten, Input, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
import metrics


def hw(input):
    return bin(int(input)).count('1')

### Scripts based on ASCAD github : https://github.com/ANSSI-FR/ASCAD

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


### CNN Best model
def architecture_model(node=200, layer_nb=6, input_size=1250, learning_rate=0.00001, classes=9, loss="llr"):
    model = Sequential()
    model.add(Dense(node, input_dim=input_size, activation='relu'))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation='relu'))


    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(learning_rate)
    if loss=='llr':
        model.compile(loss=metrics.better_tf_version_with_logs, optimizer=optimizer, metrics=['accuracy'])
    # elif loss == 'cer':
    #     model.compile(loss=lf.cross_entropy_ratio, optimizer=optimizer, metrics=['accuracy'])
    elif loss == 'rl':
        model.compile(loss=metrics.loss_sca(), optimizer=optimizer, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def cnn_architecture(input_size=1250, learning_rate=0.00001, classes=9, loss = "llr"):
    # Designing input layer
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    score_layer = Dense(classes, activation=None)(x)
    pred_layer = Activation('softmax')(score_layer)


    # Create model
    inputs = img_input
    model = Model(inputs, pred_layer, name='aes_hd_model')
    optimizer = Adam(lr=learning_rate)
    if loss=='llr':
        model.compile(loss=metrics.better_tf_version_with_logs, optimizer=optimizer, metrics=['accuracy'])
    # elif loss == 'cer':
    #     model.compile(loss=lf.cross_entropy_ratio, optimizer=optimizer, metrics=['accuracy'])
    elif loss == 'rl':
        model.compile(loss= metrics.loss_sca(score_layer, alpha_value=10, nb_class=classes), optimizer=optimizer, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


#### Training high
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, early_stop = None):
    # check_file_exists(save_file_name)
    print(save_file_name)
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    #early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    # if input_layer_shape[ASCAD_0][1] != len(X_profiling[ASCAD_0]):
    #     print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[ASCAD_0])))
    #     sys.exit(-1)

    # Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[ASCAD_0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[ASCAD_0], X_test.shape[1], 1))

    callbacks=[save_model]
    #, lr_manager
    if early_stop is not None:
        callbacks.append(early_stop)
    history = model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=9), validation_data=(X_test, to_categorical(Y_test, num_classes=9)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)

    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

def run(model_name, metrics_names, loss_func, nb_epochs, batch_size, learning_rate, early_stopping, reshape_x_attack,
        early_stopping_patience, *args, **kwargs):
    # Our folders
    root = "./datasets/AES_HD/"
    AESHD_data_folder = root + "training_data/"
    AESHD_trained_models_folder = "results/AES_HD/trained_models/"
    history_folder = "results/AES_HD/training_history/"
    predictions_folder = "results/AES_HD/model_predictions/"


    input_size = 1250

    nb_traces_attacks = 1500
    nb_attacks = 100
    real_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

    start = time.time()

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (ciphertext_profiling, ciphertext_attack) = (np.load(
        AESHD_data_folder + 'profiling_traces_AES_HD.npy'), np.load(
        AESHD_data_folder + 'profiling_labels_AES_HD.npy')), (np.load(AESHD_data_folder + 'attack_traces_AES_HD.npy'),
                                                              np.load(
                                                                  AESHD_data_folder + 'attack_labels_AES_HD.npy')), (
                                                                                                  np.load(
                                                                                                      AESHD_data_folder + 'profiling_ciphertext_AES_HD.npy'),
                                                                                                  np.load(
                                                                                                      AESHD_data_folder + 'attack_ciphertext_AES_HD.npy'))

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
                          AESHD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size,
                          early_stop=early_stop)
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
    exploit_pred = importer.import_module("datasets.AES_HD.exploit_pred")
    avg_rank = exploit_pred.perform_attacks(nb_traces_attacks, predictions, nb_attacks, ciph=ciphertext_attack, key=real_key, byte=0,
                               filename=model_name)

    if early_stop is not None:
        np.save("plot_data/" + f"se{early_stop.stopped_epoch}" + model_name.replace('.h5', '.npy'), avg_rank)
    else:
        np.save("plot_data/" + model_name.replace('.h5', '.npy'), avg_rank)
    print("\n t_GE = ")
    print(np.where(avg_rank <= 0))

    print("\n############### Attack on Test Set Done #################\n")
