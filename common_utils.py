import numpy as np
import tensorflow as tf

YCBM_CLASS_WEIGHTS = np.asarray([4, 4, 5, 6, 5, 6, 7, 7, 6, 8, 3, 3, 4, 7, 2, 2, 8, 30, 6, 5, 2])
YCBM_CLASS_WEIGHTS_TF = tf.constant(YCBM_CLASS_WEIGHTS, dtype=tf.float32)

YCBM_CLASSES_LIST = [
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick'
]

YCBM_CLASSES_LOOKUP = {}
for num, item in enumerate(YCBM_CLASSES_LIST):
    YCBM_CLASSES_LOOKUP[item] = num

YCBM_CLASS_TO_SEG_VAL = np.ones(((len(YCBM_CLASSES_LIST)+1,)), dtype=np.uint8)
YCBM_CLASS_TO_SEG_VAL *= 12
YCBM_CLASS_TO_SEG_VAL[0] = 0
YCBM_CLASS_TO_SEG_VAL = YCBM_CLASS_TO_SEG_VAL.cumsum()

YCBM_SEGVALS_TO_CLASS = np.zeros((255,), dtype=np.uint8)
for class_num, seg_val in enumerate(YCBM_CLASS_TO_SEG_VAL):
    YCBM_SEGVALS_TO_CLASS[seg_val] = class_num


def depth_to_point_cloud(depth_image, K, px_to_m=1):
    cx = K[0, 2]
    cy = K[1, 2]
    focal_x = K[0, 0]
    focal_y = K[1, 1]

    depth_image *= px_to_m

    blank = np.ones(depth_image.shape)
    u = blank.cumsum(axis=0)
    v = blank.cumsum(axis=1)
    print(u)
    print(v)


    x_over_z = (cx - u) / focal_x
    y_over_z = (cy - v) / focal_y
    z = depth_image / np.sqrt(1. + x_over_z ** 2 + y_over_z ** 2)
    x = x_over_z * z
    y = y_over_z * z

    return x, y, z

class seg_out_to_img():
    def __init__(self, num_models):
        color_space = [int(255*i/num_models+1) for i in range(num_models+1)]
        color_space = np.asarray(color_space)

        R = color_space.copy()
        np.random.shuffle(R)

        G = color_space.copy()
        np.random.shuffle(G)

        B = color_space.copy()
        np.random.shuffle(B)

        self.RGB = np.stack([R, G, B])
        self.RGB.shape = (num_models+1, 3)
        self.num_models = num_models

    def transform_seg(self, seg_out):
        seg_as_sparse_img = seg_out.argmax(axis=3)

        out_image = np.zeros((*seg_as_sparse_img.shape, 3), dtype=np.uint8)
        for i in range(self.num_models+1):
            out_image[seg_as_sparse_img == i] = self.RGB[i]
        return out_image

def Conv2d_layer(input_layer, num_filters, kernel, strides=(1, 1), name_template=''):
    noisy_in = tf.keras.layers.GaussianNoise(0.1, name=name_template+'Gauss')(input_layer)
    #conv_in = tf.keras.layers.Conv2D(num_filters, (1, 1), strides, bias_initializer='glorot_uniform',
    #                           name=name_template + 'conv2d_in', activation=None, kernel_regularizer='l2'
    #                           )(noisy_in)
    #x = tf.keras.layers.Conv2DTranspose(num_filters*2, kernel, strides, bias_initializer='glorot_uniform',
    #                           name=name_template + 'conv2dTranspose', activation=None, #kernel_regularizer='l2'
    #                           )(conv_in)
    x = tf.keras.layers.Conv2D(num_filters, kernel, strides, bias_initializer='glorot_uniform',
                               name=name_template + 'conv2d', activation=None, kernel_regularizer='l2'
                               )(noisy_in)

    #x = tf.keras.layers.Add(name=name_template)([conv_in, x])

    #x = tf.keras.layers.BatchNormalization(name=name_template + 'BN')(x)
    #x = tf.keras.layers.Activation('tanh', name=name_template + 'Tanh')(x)
    x = tf.keras.layers.LeakyReLU(name=name_template + 'LReLU')(x)

    x = tf.keras.layers.SpatialDropout2D(0.1)(x)

    return x


