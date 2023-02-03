import numpy as np
import os
import cv2
import warnings
import json

from tensorflow.keras.utils import Sequence
from scripts.utils.Feature_Extraction import gen_feature_map
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from scripts.utils.common_utils import depth_to_point_cloud, YCBM_CLASS_WEIGHTS, YCBM_CLASS_TO_SEG_VAL

class ycbm_mask_generator(Sequence):
    def __init__(self, batch_size, ycbm_path, img_size, shuffle, num_models, augment, camera_types):
        #self.FE_model = FE_model
        #self.fm_shape = FE_model.output_shape[1:]
        self.augment = augment

        # find each rbg image in each subdirctory
        self.setups = {}
        self.num_images = 0
        for camera_type in camera_types:
            setup_path = ycbm_path + os.path.sep + camera_type
            for setup in os.listdir(setup_path):
                instances = []
                if self.setups.get(setup) == None:
                    self.setups[setup] = []
                snapshot_path = setup_path + os.path.sep + setup + os.path.sep + 'snapshots' + os.path.sep
                for filename in os.listdir(snapshot_path):
                    instance_num = filename.split('.')[0]
                    if instance_num in instances:
                        continue
                    elif '_' in instance_num:
                        continue
                    instances.append((camera_type, 'snapshots', instance_num))
                    self.num_images += 1

                snapshot_path = setup_path + os.path.sep + setup + os.path.sep + 'trajectory' + os.path.sep
                for filename in os.listdir(snapshot_path):
                    instance_num = filename.split('.')[0]
                    if instance_num in instances:
                        continue
                    elif '_' in instance_num:
                        continue
                    instances.append((camera_type, 'trajectory', instance_num))
                    self.num_images += 1
                self.setups[setup].extend(instances)
        self.ycbm_path = ycbm_path
        self.img_size = img_size

        self.max_trans = int(self.img_size[0]/10)
        self.min_scale = 0.75
        self.img_center = ((int(self.img_size[0]/2), int(self.img_size[1]/2)))

        self.shuffle = shuffle
        self.img_shape = (*img_size, 3)
        self.batch_size = batch_size
        self.num_models = num_models
        self.mask_shape = (*img_size, num_models)

        background_weight = np.ones((1,))
        class_weights = YCBM_CLASS_WEIGHTS
        self.class_weights = tf.constant(np.concatenate([background_weight, class_weights]))
        #self.class_weights = self.class_weights / tf.reduce_sum(self.class_weights)

        self.rng = np.random.default_rng()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = []
        id_num = -1
        for setup in self.setups:
            for item in self.setups[setup]:
                id_num += 1
                if id_num in indexes:
                    self.list_IDs_temp.append((setup, item))
                    if len(self.list_IDs_temp) == len(indexes):
                        break
        assert len(self.list_IDs_temp) == self.batch_size, "could not find item number {}".format(indexes)

        # Generate data
        X, y, sample_weights = self.__data_generation(self.list_IDs_temp)

        return X, y, sample_weights

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.zeros((self.batch_size, *self.img_size, 3))
        masks = np.zeros((self.batch_size, *self.img_size, self.num_models+1))
        sample_weights = np.zeros((self.batch_size, *self.img_size, 1))

        for sample_num, load_info in enumerate(img_paths_tmp):
            # load the rgb
            setup, item = load_info
            camera_type, snap_vs_traj, item = item
            img_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
            img_path += item + '.jpg'
            img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            img = np.array(img)
            img = img.astype(np.float32)
            # img /= 255


            mask_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
            mask_path += item + '.seg.png'
            mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
            mask = np.array(mask)

            done_augment = False
            if self.augment:
                while done_augment == False:
                    scale = 1
                    if self.rng.uniform(0, 1) > 0.5:
                        scale = np.random.uniform(self.min_scale, 0.5+self.min_scale)

                    X_trans = 0
                    Y_trans = 0
                    if self.rng.uniform(0, 1) > 0.5:
                        X_trans = self.rng.uniform(-self.max_trans, self.max_trans)
                        Y_trans = self.rng.uniform(-self.max_trans, self.max_trans)

                    theta = 0
                    if self.rng.uniform(0, 1) > 0.5:
                        theta = self.rng.uniform(-180, 180)

                    M = cv2.getRotationMatrix2D((self.img_center[0]+X_trans, self.img_center[1]+Y_trans),
                                                scale=scale,
                                                angle=theta)

                    mask = cv2.warpAffine(mask, M, self.img_size, flags=cv2.INTER_NEAREST)
                    img = cv2.warpAffine(img, M, self.img_size)

                    if (np.any(mask) and np.any(img)):
                        done_augment = True
                    else:
                        warnings.warn('Improper warp in '+img_path+' all zeros', RuntimeWarning)

                        img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
                        # img = np.array(img) / 255
                        mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
                        mask = np.array(mask)


            imgs[sample_num] = img

            for class_id in range(self.num_models+1):
                masks[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id], class_id] = 1
                sample_weights[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id]] = self.class_weights[class_id]

        return imgs, masks, sample_weights
