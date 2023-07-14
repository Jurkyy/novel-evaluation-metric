import keras.losses
import tensorflow as tf


def cross_entropy_ratio(y_true, y_pred):
    correct_key_val = keras.losses.categorical_crossentropy(y_true, y_pred)
    temp = 0
    N = 30
    for i in range(N):
        # TODO 'proper regulation term'
        temp += keras.losses.categorical_crossentropy(tf.random.shuffle(y_true), y_pred)
    result = correct_key_val / (temp / N)
    # print(correct_key_val/(temp/N))

    return result
