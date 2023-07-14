from keras.models import Model
from keras.optimizers import RMSprop
from keras import Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation
from keras.optimizers import Adam


def AES_HD_MLP(loss_function, metrics, node=200, layer_nb=6, input_size=1250, learning_rate=0.00001, classes=9):
    model = Sequential()
    model.add(Dense(node, input_dim=input_size, activation='relu'))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation='relu'))


    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


def AES_HD_CNN(loss_function, metrics, input_size=1250, learning_rate=0.00001, classes=9):
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
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


### CNN network
def AES_RD_CNN(loss_function, metrics, input_size, learning_rate, classes):
    # Designing input layer
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(7, strides=7, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_rd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


def DPAv4_MLP(loss_function, metrics, input_size, learning_rate, classes=256, node=200, layer_nb=6):
    model = Sequential()
    model.add(Dense(node, input_dim=input_size, activation='relu'))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN network
def DPAv4_CNN(loss_function, metrics, input_size, learning_rate, classes=9):
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
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    # model.compile(loss=metrics.better_tf_version_with_logs, optimizer=optimizer, metrics=['accuracy'])
    return model


def ASCAD_0_MLP(loss_function, metrics, node=200, layer_nb=6, input_size=4000,learning_rate=0.00001,classes=256):
    model = Sequential()
    model.add(Dense(node, input_dim=input_size, activation='relu'))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


### CNN network
def ASCAD_0_CNN(loss_function, metrics, input_size=700, learning_rate=0.00001, classes=256):
    # Designing input layer
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='ascad')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


### CNN network
def ASCAD_50_CNN(loss_function, metrics, input_size=700, learning_rate=0.00001, classes=9):
    # Personal design
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(
        img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model


### CNN Best model
def ASCAD_100_CNN(loss_function, metrics, input_size=700, learning_rate=0.00001, classes=9):
    # Personal design
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(
        img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    return model