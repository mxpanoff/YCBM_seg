import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG19, EfficientNetB7, EfficientNetB4, MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.efficientnet import preprocess_input

import argparse
from tensorflow.keras.layers import BatchNormalization

def gen_FE_model(type = 'B0', unfreeze=0, desired_size=None):
    '''
    Make a feature extraction model
    :return: a frozen model
    '''
    if type == 'B4':
        if desired_size == None:
            desired_size = (380, 380, 3)
        model = EfficientNetB4(include_top=False, input_shape=desired_size, weights='imagenet')
    elif type == 'B0':
        if desired_size == None:
            desired_size = (224, 224, 3)
        model = EfficientNetB0(include_top=False, input_shape=desired_size, weights='imagenet')
    elif type == 'VGG19':
        if desired_size == None:
            desired_size = (224, 224, 3)
        model = VGG19(include_top=False, input_shape=desired_size, weights='imagenet')
    elif type == 'VGG16':
        if desired_size == None:
            desired_size = (224, 224, 3)
        model = VGG16(include_top=False, input_shape=desired_size, weights='imagenet')
    elif type == 'Res50':
        if desired_size == None:
            desired_size = (224, 224, 3)
        model = ResNet50V2(include_top=False, input_shape=desired_size, weights='imagenet')
    elif type == 'Mobilenet':
        if desired_size == None:
            desired_size = (224, 224, 3)
        model = MobileNetV2(include_top=False, input_shape=desired_size, weights='imagenet')
    else:
        raise ValueError

    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        layer.trainable = False

    if unfreeze > 0:
        for i in range(unfreeze):
            if isinstance(model.layers[-(i+1)], BatchNormalization):
                model.layers[-(i+1)].trainable = False
            model.layers[-(i+1)].trainable = True
    elif unfreeze == -1:
        for layer in model.layers:
            layer.trainable = True

    return model

def gen_feature_map(img, FE_model):
    '''
    convert a (keras preprocessed) image into an appropriate input
    :param img: preprocessed image
    :param FE_model: model to use for feature extraction
    :return: the feature map of the input as found by the model
    '''
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_map = FE_model.predict(img)
    return feature_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_image', type=str, help='path to test image')

    args = parser.parse_args()

    image_path = args.demo_image
    print(image_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = np.array(img)

    FE_model = gen_FE_model()
    feature_map = gen_feature_map(img, FE_model)
    print(feature_map.shape, feature_map.min(), feature_map.max(), feature_map.mean(), feature_map.sum())