import  numpy as np
import tensorflow as tf
import math
def ratio_guess(y_true, y_pred):



    def where_ones(labels):
        return_arr = np.zeros(len(labels))
        for j in range(len(labels)):
            for k in range(len(labels)):
                if labels[j][k] == 1:
                    return_arr[j] = k
                    break

        return return_arr

    def summer(labels: np.ndarray, preds, epsilon):
        score = 0


        for j in range(labels.size):
            if not math.isinf(preds[j][int(labels[j])]):
                score += preds[j][int(labels[j])]
            else:
                score += epsilon
        return score


    N = 2
    logs_nump = np.log(y_pred)
    pred_num = where_ones(y_true)
    epsilon = np.log(np.min(y_pred[np.nonzero(y_pred)]))

    top_score = summer(pred_num, logs_nump, epsilon)
    #print(top_score)
    temp = 0
    for i in range(N):
        np.random.shuffle(pred_num)
        temp += summer(pred_num, logs_nump, epsilon)
        #print(temp)

    return top_score/(temp/N)


def tf_version_tf_ratio(y_true, y_pred):
    def ratio_guess(y_true, y_pred):

        def where_ones(labels):
            return_arr = np.zeros(len(labels))
            for j in range(len(labels)):
                for k in range(len(labels)):
                    if labels[j][k] == 1:
                        return_arr[j] = k
                        break

            return return_arr

        def summer(labels, preds):
            score = 0
            for j in range(labels.shape[0]):
                if not math.isinf(preds[j][int(labels[j])]):
                    score += preds[j][int(labels[j])]
            return score

        N = 2
        logs_nump = np.log(y_pred)
        pred_num = where_ones(y_true)
        top_score = summer(pred_num, logs_nump)
        #print(top_score)
        temp = 0
        for i in range(N):
            pred_num = tf.random.shuffle(pred_num)
            temp += summer(pred_num, logs_nump)
            #print(temp)

        return top_score / (temp / N)

    return ratio_guess(y_true.numpy(), y_pred.numpy())
