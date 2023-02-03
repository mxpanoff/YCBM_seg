import tensorflow as tf
import numpy as np
import io
import os
import random
from PIL import Image as PIL_Image

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.efficientnet import preprocess_input

from scripts.utils.common_utils import seg_out_to_img
from scripts.utils.Feature_Extraction import gen_FE_model, gen_feature_map

from matplotlib import pyplot as plt

class rbg_out_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, freq, train_gen, val_gen, **kwargs):
        super(rbg_out_callback, self).__init__(**kwargs)
        self.val_dir = '/Data/MultiCamPose/Datasets/lm/train_pbr/val/000040/'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.current_batch_num = 0
        self.freq = freq
        self.val_gen = val_gen
        self.train_gen = train_gen
        self.FE_model = gen_FE_model()

    def on_train_batch_begin(self, batch, logs=None):
        self.current_batch_num += 1
        if self.current_batch_num % self.freq == 0:
            sample_num = np.random.randint(0, 100)
            rgb_img, mask, weights = self.val_gen.__getitem__(sample_num)
            rgb_img = np.expand_dims(rgb_img[0], 0) / 255
            self.val_model_in = rgb_img

            with self.writer.as_default():
                tf.summary.image("Val RGB Image", rgb_img, step=batch)

            rgb_img, mask, weights = self.train_gen.__getitem__(sample_num)
            rgb_img = np.expand_dims(rgb_img[0], 0) / 255
            self.train_model_in = rgb_img
            with self.writer.as_default():
                tf.summary.image("Input RGB Image", rgb_img, step=batch)

class seg_out_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, freq, train_gen, val_gen, num_models, **kwargs):
        super(seg_out_callback, self).__init__(**kwargs)
        self.val_dir = '/Data/MultiCamPose/Datasets/lm/train_pbr/val/000040/'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.current_batch_num = 0
        self.freq = freq
        self.val_gen = val_gen
        self.train_gen = train_gen
        self.FE_model = gen_FE_model()
        self.num_models = num_models
        self.seg_out = seg_out_to_img(num_models)

    def on_train_batch_end(self, batch, logs=None):
        if self.current_batch_num % self.freq == 0:
            self.seg_out_as_image(batch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.current_batch_num += 1
        if self.current_batch_num % self.freq == 0:
            sample_num = np.random.randint(0, 100)
            rgb_img, mask, weights = self.val_gen.__getitem__(sample_num)
            if type(rgb_img) == list:
                rgb_img[0] = np.expand_dims(rgb_img[0][0], 0)
                rgb_img[1] = np.expand_dims(rgb_img[1][0], 0)
                rgb_img[2] = np.expand_dims(rgb_img[2][0], 0)
            else:
                rgb_img = np.expand_dims(rgb_img[0], 0)
            self.val_model_in = rgb_img
            mask = np.expand_dims(mask[0], 0)
            mask_as_image = self.seg_out.transform_seg(mask)
            weights = np.expand_dims(weights[0], 0)
            weights = ((weights - weights.min())/weights.max())
            with self.writer.as_default():
                try:
                    tf.summary.image("Val RGB Image", rgb_img, step=batch)
                except ValueError:
                    tf.summary.image("Val RGB Image", rgb_img[0], step=batch)
                tf.summary.image("Val Mask Image", mask_as_image, step=batch)
                tf.summary.image("Val weights", weights, step=batch)

            rgb_img, mask, weights = self.train_gen.__getitem__(sample_num)
            if type(rgb_img) == list:
                rgb_img[0] = np.expand_dims(rgb_img[0][0], 0)
                rgb_img[1] = np.expand_dims(rgb_img[1][0], 0)
                rgb_img[2] = np.expand_dims(rgb_img[2][0], 0)
            else:
                rgb_img = np.expand_dims(rgb_img[0], 0)
            self.train_model_in = rgb_img
            mask = np.expand_dims(mask[0], 0)
            mask_as_image = self.seg_out.transform_seg(mask)
            weights = np.expand_dims(weights[0], 0)
            weights = ((weights - weights.min()) / weights.max())
            with self.writer.as_default():
                try:
                    tf.summary.image("Input RGB Image", rgb_img, step=batch)
                except ValueError:
                    tf.summary.image("Input RGB Image", rgb_img[0], step=batch)
                tf.summary.image("Input Mask Image", mask_as_image, step=batch)
                tf.summary.image("Input weights", weights, step=batch)

    def seg_out_as_image(self, batch, logs):
      # Use the model to predict the values from the validation dataset.
      test_pred_raw = self.model.predict(self.val_model_in)

      # Calculate the confusion matrix.
      # Log the confusion matrix as an image summary.
      val_seg_image = self.seg_out.transform_seg(test_pred_raw)

      test_pred_raw = self.model.predict(self.train_model_in)
      train_seg_image = self.seg_out.transform_seg(test_pred_raw)

      # Log the confusion matrix as an image summary.
      with self.writer.as_default():
        tf.summary.image("Val Out", val_seg_image, step=batch)
        tf.summary.image("Train Out", train_seg_image, step=batch)

        self.writer.flush()


def train_seg(model, training_generator, validation_generator, model_path, batch_size, num_epochs, num_models):
    '''
    Trains a keras model
    :param model: keras model instance
    :param training_generator: generator (keras sequence) of training data
    :param validation_generator: generator (keras sequence) of validation data
    :param model_path: Path to save model checkpoints
    :param batch_size: batch size to use in training
    :param num_epochs: number of epochs to use in training
    :return: model training history and the trained model
    '''
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'.2D_Seg_Tran{epoch:02d}-{val_mAR:.4f}.h5',
                                                    save_best_only=True, monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/seg', histogram_freq=1)
    tensorboard_batch = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/seg', update_freq='batch', histogram_freq=200)
    seg_image = seg_out_callback(log_dir='../TBlogs/seg', freq=200, val_gen=validation_generator,
                                 train_gen=training_generator, num_models=num_models)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.01, verbose=True, factor=0.5)
    #dataset_wrapped = tf.data.Dataset.from_generator(training_generator)
    history = model.fit(x=training_generator, validation_data=validation_generator, batch_size=batch_size,
                        epochs=num_epochs, callbacks=[save_model, tensorboard, lr_reducer, seg_image, tensorboard, tensorboard_batch])
                        #use_multiprocessing=True, workers=4)
    return history, model


def train_dist(model, training_generator, validation_generator, model_path, batch_size, num_epochs):
    '''
    Trains a keras model
    :param model: keras model instance
    :param training_generator: generator (keras sequence) of training data
    :param validation_generator: generator (keras sequence) of validation data
    :param model_path: Path to save model checkpoints
    :param batch_size: batch size to use in training
    :param num_epochs: number of epochs to use in training
    :return: model training history and the trained model
    '''
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'.Dist{epoch:02d}-{val_loss:.4f}.h5',
                                                    save_best_only=True, monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/dist', histogram_freq=1)
    tensorboard_batch = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/dist', update_freq='batch', histogram_freq=200)
    rbg_image = rbg_out_callback(log_dir='../TBlogs/dist', freq=200, val_gen=validation_generator,
                                train_gen=training_generator)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.01, verbose=True, factor=0.5)
    #dataset_wrapped = tf.data.Dataset.from_generator(training_generator)
    history = model.fit(x=training_generator, validation_data=validation_generator, batch_size=batch_size,
                        epochs=num_epochs, callbacks=[save_model, tensorboard, lr_reducer, tensorboard, tensorboard_batch, rbg_image])
                        #use_multiprocessing=True, workers=4)
    return history, model

def train_seg_comb(model, training_generator, validation_generator, model_path, batch_size, num_epochs, num_models):
    '''
    Trains a keras model
    :param model: keras model instance
    :param training_generator: generator (keras sequence) of training data
    :param validation_generator: generator (keras sequence) of validation data
    :param model_path: Path to save model checkpoints
    :param batch_size: batch size to use in training
    :param num_epochs: number of epochs to use in training
    :return: model training history and the trained model
    '''
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'.2D_Seg_Tran{epoch:02d}-{val_classify_dice_coef:.4f}.h5',
                                                    save_best_only=True, monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/seg', histogram_freq=1)
    tensorboard_batch = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/seg', update_freq='batch', histogram_freq=200)
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()
    # seg_image = seg_out_callback(log_dir='../TBlogs/seg', freq=200, val_gen=validation_generator,
    #                              train_gen=training_generator, num_models=num_models)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.001, verbose=True, factor=0.5)
    #dataset_wrapped = tf.data.Dataset.from_generator(training_generator)
    history = model.fit(x=training_generator, validation_data=validation_generator, batch_size=batch_size,
                        epochs=num_epochs, callbacks=[save_model, tensorboard, lr_reducer, tensorboard,
                                                      tensorboard_batch, stop_on_nan])
                        #use_multiprocessing=True, workers=4)
    return history, model

def train_complete(model, training_generator, validation_generator, model_path, batch_size, num_epochs):
    '''
    Trains a keras model
    :param model: keras model instance
    :param training_generator: generator (keras sequence) of training data
    :param validation_generator: generator (keras sequence) of validation data
    :param model_path: Path to save model checkpoints
    :param batch_size: batch size to use in training
    :param num_epochs: number of epochs to use in training
    :return: model training history and the trained model
    '''
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'.complete{epoch:02d}-{val_sparse_categorical_accuracy:.4f}.h5',
                                                    save_best_only=True, monitor='sparse_categorical_accuracy')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/complete', write_images=True, histogram_freq=1)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, min_delta=0.001, verbose=True)
    history = model.fit(x=training_generator, validation_data=validation_generator, batch_size=batch_size,
                        epochs=num_epochs, callbacks=[save_model, tensorboard, lr_reducer])
    return history, model

def train_AE(model, training_generator, validation_generator, model_path, batch_size, num_epochs):
    '''
    Trains a keras model
    :param model: keras model instance
    :param training_generator: generator (keras sequence) of training data
    :param validation_generator: generator (keras sequence) of validation data
    :param model_path: Path to save model checkpoints
    :param batch_size: batch size to use in training
    :param num_epochs: number of epochs to use in training
    :return: model training history and the trained model
    '''
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'.AE{epoch:02d}-{val_mean_absolute_percentage_error:.4f}.h5',
                                                    save_best_only=True, monitor='mean_absolute_percentage_error')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../TBlogs/FSR', write_images=True, histogram_freq=1)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, min_delta=0.001, verbose=True)
    history = model.fit(x=training_generator, validation_data=validation_generator, batch_size=batch_size,
                        epochs=num_epochs, callbacks=[save_model, tensorboard, lr_reducer])
    return history, model