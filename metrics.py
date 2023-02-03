import tensorflow as tf
from scripts.utils.common_utils import YCBM_CLASS_WEIGHTS_TF, YCBM_CLASS_WEIGHTS
import numpy as np

#tf.config.run_functions_eagerly(True)

def mAR(y_true, y_pred, smooth=0.01):
    # pred_class_per_pixel = tf.argmax(y_pred, axis=-1)
    # true_class_per_pixel = tf.argmax(y_true, axis=-1)
    #
    # # find where the predictions match the truth
    # correct_preds = tf.equal(true_class_per_pixel, pred_class_per_pixel)
    # #weights = tf.cast(tf.gather(YCBM_CLASS_WEIGHTS, true_class_per_pixel), dtype=tf.float32)
    #
    # # find where the true values are not background
    # true_non_background = tf.not_equal(true_class_per_pixel, tf.zeros(1, dtype=tf.int64))
    # # find where the predictions are correct AND not background
    # true_background_correct = tf.math.logical_and(correct_preds, true_non_background)
    # true_background_correct = tf.where(true_background_correct, x=1, y=0)
    # true_background_correct = tf.cast(true_background_correct, dtype=tf.float32)# * weights
    # # find the total non_background
    # true_non_background = tf.where(true_non_background, x=1, y=0)
    # true_non_background = tf.cast(true_non_background, dtype=tf.float32)# * weights
    #
    # num_correct = tf.reduce_sum(true_background_correct, axis=(1, 2))
    # # add small value to precent nan
    # num_total = tf.reduce_sum(true_non_background, axis=(1, 2)) #+ tf.constant(1, dtype=tf.float32)
    #
    # recall = tf.reduce_mean(num_correct/num_total)
    pred_class = tf.argmax(y_pred, axis=-1)
    true_class = tf.argmax(y_true, axis=-1)

    num_classes = len(YCBM_CLASS_WEIGHTS)
    mAR = tf.zeros((1,))
    classes_in_img = tf.zeros((1,))
    for class_num in range(num_classes):
        class_num += 1
        preds_for_class = pred_class == class_num
        trues_for_class = true_class == class_num

        has_class = tf.reduce_any(trues_for_class)
        has_class = tf.where(has_class, x=1.0, y=0.0)
        classes_in_img += has_class

        true_preds = tf.logical_and(trues_for_class, preds_for_class)
        true_preds = tf.where(true_preds, x=1, y=0)
        true_preds = tf.cast(tf.reduce_sum(true_preds, axis=[1, 2]), dtype=tf.float32)

        true_positives = tf.where(trues_for_class, x=1, y=0)
        true_positives = tf.cast(tf.reduce_sum(true_positives, axis=[1, 2]), dtype=tf.float32)

        false_negatives = tf.logical_and(tf.logical_not(preds_for_class), trues_for_class)
        false_negatives = tf.where(false_negatives, x=1, y=0)
        false_negatives = tf.cast(tf.reduce_sum(false_negatives, axis=[1, 2]), dtype=tf.float32)

        recall_per_batch = (2 * true_preds) / (true_positives + false_negatives + smooth)
        temp = tf.reduce_mean(recall_per_batch)
        # temp = tf.where(has_class, x=temp, y=0.0)
        mAR += temp
        # dice += dice_per_batch

    mAR /= classes_in_img
    return mAR


def mAP(y_true, y_pred, smooth=0.01):
    # pred_class_per_pixel = tf.argmax(y_pred, axis=-1) # batch, x, y, val is class
    # true_class_per_pixel = tf.argmax(y_true, axis=-1) # batch, x, y, val is class
    #
    # equals = tf.equal(pred_class_per_pixel, true_class_per_pixel)
    # #weights = tf.gather(YCBM_CLASS_WEIGHTS_TF, true_indices)
    #
    # pred_non_background = tf.not_equal(pred_class_per_pixel, tf.constant(0, dtype=tf.int64))
    #
    # pred_background_correct = tf.math.logical_and(equals, pred_non_background)
    # pred_background_correct = tf.where(pred_background_correct, x=1, y=0)
    # pred_background_correct = tf.cast(pred_background_correct, dtype=tf.float32)# * weights
    #
    # pred_non_background = tf.where(pred_non_background, x=1, y=0)
    # pred_non_background = tf.cast(pred_non_background, dtype=tf.float32)# * weights
    #
    # num_correct = tf.reduce_sum(pred_background_correct, axis=(1, 2))
    # num_total = tf.reduce_sum(pred_non_background, axis=(1, 2)) #+ tf.constant(1, dtype=tf.float32)
    #
    # precision = tf.reduce_mean(num_correct/(num_total+1))

    pred_class = tf.argmax(y_pred, axis=-1)
    true_class = tf.argmax(y_true, axis=-1)

    num_classes = len(YCBM_CLASS_WEIGHTS)
    mAP = tf.zeros((1,))
    classes_in_img = tf.zeros((1,))
    for class_num in range(num_classes):
        class_num += 1
        preds_for_class = pred_class == class_num
        trues_for_class = true_class == class_num

        has_class = tf.reduce_any(trues_for_class)
        has_class = tf.where(has_class, x=1.0, y=0.0)
        classes_in_img += has_class

        true_preds = tf.logical_and(trues_for_class, preds_for_class)
        true_preds = tf.where(true_preds, x=1, y=0)
        true_preds = tf.cast(tf.reduce_sum(true_preds, axis=[1, 2]), dtype=tf.float32)

        true_positives = tf.where(trues_for_class, x=1, y=0)
        true_positives = tf.cast(tf.reduce_sum(true_positives, axis=[1, 2]), dtype=tf.float32)

        false_positives = tf.logical_and(tf.logical_not(trues_for_class), preds_for_class)
        false_positives = tf.where(false_positives, x=1, y=0)
        false_positives = tf.cast(tf.reduce_sum(false_positives, axis=[1, 2]), dtype=tf.float32)

        precision_per_batch = (2 * true_preds) / (true_positives + false_positives + smooth)
        mAP += tf.reduce_mean(precision_per_batch)
        # dice += dice_per_batch

    mAP /= classes_in_img

    return mAP


def dice_coef(y_true, y_pred, smooth=0.01):
    # true_class_per_pixel = tf.argmax(y_true, axis=-1) # batch, x, y, val is class
    # weights = tf.cast(tf.gather(YCBM_CLASS_WEIGHTS_TF, true_class_per_pixel), dtype=tf.float32) # batch, x, y, val is weight
    # weights = tf.expand_dims(weights, axis=-1)
    # #weighted_true = tf.expand_dims(weights, axis=-1) * y_true
    # #weighted_preds = tf.expand_dims(weights, axis=-1) * y_pred
    #
    # intersection = tf.reduce_sum(weights * y_true[:, :, :, 1:] * y_pred[:, :, :, 1:], axis=[1, 2])
    #
    # union = tf.reduce_sum(weights * y_true[:, :, :, 1:], axis=[1, 2])\
    #         + tf.reduce_sum(weights * y_pred[:, :, :, 1:], axis=[1, 2])\
    #
    # dice = tf.reduce_mean((2. * intersection)/(union + smooth), axis=1)
    # dice = tf.reduce_mean(dice)

    pred_class = tf.argmax(y_pred, axis=-1)
    true_class = tf.argmax(y_true, axis=-1)

    num_classes = len(YCBM_CLASS_WEIGHTS) + 1

    all_weights = np.asarray([1, *YCBM_CLASS_WEIGHTS])
    all_weights = all_weights / all_weights.sum()
    classes_in_img = tf.zeros((1,))
    dice = tf.zeros(1)
    for class_num in range(num_classes):
        preds_for_class = pred_class == class_num
        trues_for_class = true_class == class_num

        has_class = tf.reduce_any(trues_for_class)
        has_class = tf.where(has_class, x=1.0, y=0.0)
        classes_in_img += has_class

        intersection = tf.logical_and(trues_for_class, preds_for_class)
        intersection = tf.cast(tf.where(intersection, x=1, y=0), dtype=tf.float32)
        #intersection = tf.reduce_sum(intersection, axis=[1, 2])

        where_trues = tf.cast(tf.where(trues_for_class, x=1, y=0), dtype=tf.float32)
        #where_trues = tf.reduce_sum(where_trues, axis=[1, 2])

        where_preds = tf.cast(tf.where(preds_for_class, x=1, y=0), dtype=tf.float32)
        #where_preds = tf.reduce_sum(where_preds, axis=[1, 2])

        dice_per_class = (2 * tf.reduce_sum(intersection)) / (tf.reduce_sum(where_trues) + tf.reduce_sum(where_preds) + smooth)
        dice += dice_per_class# * all_weights[class_num]
        #dice += dice_per_batch

    dice /= classes_in_img
    return dice