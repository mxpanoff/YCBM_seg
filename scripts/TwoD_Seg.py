import argparse
import tensorflow as tf
import numpy as np



from tensorflow.keras import layers
from scripts.utils.Model_Utils import train_seg
from scripts.utils.Generators import ycbm_mask_generator, ycbm_mask_comb_generator

from scripts.utils.common_utils import Conv2d_layer, YCBM_CLASSES_LIST
from scripts.utils.metrics import mAR, mAP, dice_coef
from scripts.utils.Feature_Extraction import gen_FE_model
from scripts.utils.losses import cce_and_dice_loss

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

def orig_embedding_model_gen(feature_map_2nd_last, feature_map_last, feature_map, in_extrinsics=None, out_extrinsics=None):
    # first seg embedding (add the 2nd to last and last stages, embed to 64 with two conv layers)
    embedded_input1 = layers.Input(shape=feature_map_2nd_last.shape[1:], name='embed1')
    embedded_fms1 = layers.Conv2D(512, (1, 1), activation=None, kernel_regularizer=None)(embedded_input1)
    embedded_fms1 = layers.LeakyReLU()(embedded_fms1)
    # embedded_fms1 = layers.Conv2DTranspose(512, (2, 2), (2, 2), kernel_regularizer='l1')(embedded_fms1)
    embedded_fms1 = layers.SpatialDropout2D(0.05)(embedded_fms1)

    # second seg embedding (deconv last layer)
    embedded_input2 = layers.Input(shape=feature_map_last.shape[1:], name='embed2')
    embedded_fms2 = layers.Conv2D(256, (1, 1), activation=None, kernel_regularizer=None)(embedded_input2)
    embedded_fms2 = layers.LeakyReLU()(embedded_fms2)
    embedded_fms2 = layers.UpSampling2D(interpolation='bilinear')(embedded_fms2)
    # embedded_fms2 = layers.Conv2DTranspose(256, (2, 2), (2, 2), kernel_regularizer=None)(embedded_fms2)
    # embedded_fms2 = layers.LeakyReLU()(embedded_fms2)
    embedded_fms2 = layers.Conv2D(512, (1, 1), activation='relu', kernel_regularizer=None)(embedded_fms2)
    embedded_fms2 = layers.LeakyReLU()(embedded_fms2)
    embedded_fms2 = layers.SpatialDropout2D(0.05)(embedded_fms2)

    # experimental seg
    embedded_input3 = layers.Input(shape=feature_map.shape[1:], name='embed3')
    embedded_fms3 = layers.Conv2D(128, (1, 1), activation=None, kernel_regularizer=None)(embedded_input3)
    embedded_fms3 = layers.LeakyReLU()(embedded_fms3)
    embedded_fms3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(embedded_fms3)
    # embedded_fms3 = layers.Conv2DTranspose(128, (2, 2), (2, 2), kernel_regularizer=None)(embedded_fms3)
    #embedded_fms3 = layers.LeakyReLU()(embedded_fms3)

    embedded_fms3 = layers.Conv2D(256, (1, 1), activation=None, kernel_regularizer=None)(embedded_fms3)
    embedded_fms3 = layers.LeakyReLU()(embedded_fms3)
    embedded_fms3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(embedded_fms3)
    #embedded_fms3 = layers.Conv2DTranspose(256, (2, 2), (2, 2), kernel_regularizer=None)(embedded_fms3)
    # embedded_fms3 = layers.LeakyReLU()(embedded_fms3)
    embedded_fms3 = layers.Conv2D(512, (1, 1), activation='relu', kernel_regularizer=None)(embedded_fms3)
    embedded_fms3 = layers.LeakyReLU()(embedded_fms3)
    embedded_fms3 = layers.SpatialDropout2D(0.05)(embedded_fms3)

    # sum embeddings
    embedded_seg = layers.Add()([embedded_fms1, embedded_fms2, embedded_fms3])
    embedded_seg = layers.SpatialDropout2D(0.05)(embedded_seg)

    embedding_model = tf.keras.Model([embedded_input1, embedded_input2, embedded_input3], embedded_seg,
                                     name='embedding2d')
    return embedding_model


def img_to_seg_poseCNN(FE_model, num_models, optimizer):
    img_input_layer = layers.Input(shape=FE_model.input_shape[1:], name='seg_input')
    # img_input_layer1 = tf.keras.backend.print_tensor(img_input_layer)
    # noisy_in = vgg_preprocess(input_layer * 255)
    noisy_in = mobilenet_preprocess(img_input_layer)
    noisy_in = layers.GaussianNoise(0.05)(noisy_in)

    # base_model = tf.keras.Model(FE_model.input, [FE_model.output,
    #                                           FE_model.get_layer('block5_conv3').output,
    #                                           FE_model.get_layer('block4_conv3').output])

    base_model = tf.keras.Model(FE_model.input, [FE_model.output,
                                              FE_model.get_layer('block_13_expand_relu').output,
                                              FE_model.get_layer('block_6_expand_relu').output])


    feature_map, feature_map_last, feature_map_2nd_last = base_model(noisy_in)
    feature_map = layers.BatchNormalization()(feature_map)
    feature_map = layers.SpatialDropout2D(0.05)(feature_map)
    feature_map = layers.GaussianNoise(0.05)(feature_map)

    #feature_map_last = base_model(input_layer)[1]
    feature_map_last = layers.BatchNormalization()(feature_map_last)
    feature_map_last = layers.SpatialDropout2D(0.05)(feature_map_last)
    feature_map_last = layers.GaussianNoise(0.05)(feature_map_last)

    #feature_map_2nd_last = base_model(input_layer)[2]
    feature_map_2nd_last = layers.BatchNormalization()(feature_map_2nd_last)
    feature_map_2nd_last = layers.SpatialDropout2D(0.05)(feature_map_2nd_last)
    feature_map_2nd_last = layers.GaussianNoise(0.05)(feature_map_2nd_last)

    #FE_model.summary()

    embedding_model = orig_embedding_model_gen(feature_map_2nd_last, feature_map_last, feature_map)

    embedded_rep = embedding_model([feature_map_2nd_last, feature_map_last, feature_map])

    # expand to original image size
    embedded_rep_in = layers.Input(shape=embedded_rep.shape[1:])
    # embedded_expanded = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation=None)(embedded_rep_in)
    # embedded_expanded = layers.LeakyReLU()(embedded_expanded)
    # embedded_expanded = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation=None)(embedded_expanded)
    # embedded_expanded = layers.LeakyReLU()(embedded_expanded)
    # embedded_expanded = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation=None)(embedded_expanded)
    # embedded_expanded = layers.LeakyReLU()(embedded_expanded)
    # embedded_expanded = layers.UpSampling2D((2, 2))(embedded_expanded)
    # embedded_expanded = layers.AvgPool2D((4, 4), strides=(2, 2))(embedded_expanded)
    embedded_expanded = layers.UpSampling2D((8, 8), interpolation='bilinear')(embedded_rep_in)
    embedded_expanded = layers.SpatialDropout2D(0.05)(embedded_expanded)

    # classify pixels
    seg_out = layers.Conv2DTranspose(num_models+1, (1, 1), activation='softmax')(embedded_expanded)

    classify_model = tf.keras.Model(embedded_rep_in, seg_out, name='classify')

    embedded_rep = layers.BatchNormalization()(embedded_rep)
    embedded_rep = layers.SpatialDropout2D(0.05)(embedded_rep)
    embedded_rep = layers.GaussianNoise(0.05)(embedded_rep)

    segmented_image = classify_model(embedded_rep)

    model = tf.keras.Model([img_input_layer], segmented_image, name='seg')
    model.compile(optimizer=optimizer, loss=cce_and_dice_loss,
                  metrics=['categorical_accuracy', mAR, mAP, dice_coef], run_eagerly=True)

    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_image_path', type=str, help='path to test image')
    parser.add_argument('val_image_path', type=str, help='path to test image')
    parser.add_argument('model_path', type=str, help='path to save trained models')
    parser.add_argument('--batch_size', type=int, help='batch size when training autoencoder', default=3)
    parser.add_argument('--lr', type=float, help='learning rate when training autoencoder', default=0.001)
    parser.add_argument('--epochs', type=int, help='epochs when training autoencoder', default=99)
    parser.add_argument('--img_size', type=tuple, help='size and number of channels of input image', default=(224, 224, 3))

    args = parser.parse_args()

    train_path = args.train_image_path
    val_path = args.val_image_path
    model_path = args.model_path
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    img_size = args.img_size

    FE_model = gen_FE_model('Mobilenet', -1, img_size)
    #FE_model.summary()
    num_models = len(YCBM_CLASSES_LIST)

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, epsilon=0.01)

    seg_net = img_to_seg_poseCNN(FE_model, num_models, optimizer=opt)

    train_gen = ycbm_mask_generator(batch_size, train_path, img_size[:-1], shuffle=True, num_models=num_models,
                                    augment=True, camera_types=['astra', 'xtion', 'kinect2', 'ensenso', 'realsense_r200'])
    val_gen = ycbm_mask_generator(batch_size, val_path, img_size[:-1], shuffle=True, num_models=num_models,
                                  augment=False, camera_types=['astra', 'xtion', 'kinect2', 'ensenso', 'realsense_r200'])

    history, seg_net = train_seg(seg_net, train_gen, val_gen, model_path, batch_size, num_epochs, num_models=num_models)
