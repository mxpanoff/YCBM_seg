from scripts.utils.metrics import dice_coef
import tensorflow as tf
import sys
from scripts.utils.common_utils import YCBM_CLASS_WEIGHTS_TF

#tf.config.run_functions_eagerly(True)


def MSE_loss(y_true, y_pred):
    MSE_loss_func = tf.keras.losses.MeanSquaredError(reduction='none')
    MSE_loss = MSE_loss_func(y_true, y_pred)
    return MSE_loss

def cce_and_dice_loss(y_true, y_pred, smooth=1E-5):
    cce_loss = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    dice = dice_loss(y_true, y_pred, smooth)
    raw_loss = cce_loss(y_true, y_pred)
    comb_loss = raw_loss * dice
    return comb_loss


def dice_loss(y_true, y_pred, smooth=1.0):
    dice_loss = tf.constant(1, dtype=tf.float32) - dice_coef(y_true, y_pred, smooth=smooth)
    dice_loss = (dice_loss + tf.reduce_mean(dice_loss)) / 2
    return dice_loss

def fm_loss(y_true, y_pred):
    # mse_loss = tf.keras.losses.MeanSquaredError(reduction='none')
    # mae_loss = tf.keras.losses.MeanAbsoluteError(reduction='none')
    # mape_loss = tf.keras.losses.MeanAbsolutePercentageError(reduction='none')
    # msle_loss = tf.keras.losses.MeanSquaredLogarithmicError(reduction='none')
    #
    # l1_l2_loss = mse_loss(y_true, y_pred) + mae_loss(y_true, y_pred)
    # ratio_loss = (mape_loss(y_true, y_pred)/100) + msle_loss(y_true, y_pred)

    #log_true = tf.math.log(tf.abs(y_true)+1.0)
    #log_pred = tf.math.log(tf.abs(y_pred)+1.0)
    #
    # fm_diff = log_true - log_pred
    #
    # fm_diff = fm_diff ** 2
    # print('\npred', y_pred)
    # print('\ntrue', y_true)
    #fm_diff = tf.square((log_pred - log_true)/log_true)
    #fm_denom = log_true
    # print('\ntrue', y_true)
    # print('\npred', y_pred)
    # print('\ndiff', fm_diff)
    # print('\ndenom', fm_denom)
    # denom = tf.abs(y_true) + tf.reduce_mean(tf.abs(y_true))/100
    #
    # MAPE = fm_diff / denom
    #tf.print('\nfm', y_pred, output_stream=sys.stdout)

    # assert not tf.reduce_any(tf.math.is_nan(y_true)), "found nan in trues"
    # assert not tf.reduce_any(tf.math.is_nan(y_pred)), "found nan in preds {}".format(y_pred)

    # y_true = tf.where(tf.math.is_nan(y_true), x=0.0, y=y_true)
    # y_pred = tf.where(tf.math.is_nan(y_pred), x=0.0, y=y_pred)

    fm_diff = tf.abs(y_pred - y_true) #* (tf.abs(tf.reduce_mean(tf.abs(y_true)) - tf.reduce_mean(tf.abs(y_pred))))

    y_true_nonzero = tf.where(y_true == 0, x=1E-5, y=y_true)

    fm_pull_against_all_zereo = tf.reduce_mean(tf.abs(y_pred))
    fm_pull_against_all_zereo = tf.where(fm_pull_against_all_zereo == 0, x=1E-9, y=fm_pull_against_all_zereo)

    fm_diff = fm_diff / (tf.abs(y_true_nonzero) * fm_pull_against_all_zereo)



    return fm_diff