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

class lmp_AE_generator(Sequence):
    '''
    A generator (keras sequence) for making a feature map autoencoder using the linemod processed dataset
    '''
    def __init__(self, batch_size, dir_path, img_size, shuffle, FE_model, FM_shape):
        self.img_paths = []
        # get all images in each sub directory
        for set in os.listdir(dir_path):
            image_dir_path = dir_path + os.path.sep + set + os.path.sep + 'rgb' + os.path.sep
            image_paths = [image_dir_path + img_path for img_path in os.listdir(image_dir_path)]
            self.img_paths.extend(image_paths)
        self.img_size = img_size
        self.shuffle = shuffle
        self.FE_model = FE_model
        self.FM_shape = FM_shape
        self.num_images = len(self.img_paths)
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.FM_shape))

        for i, path in enumerate(img_paths_tmp):
            # load image into memory
            img = tf.keras.utils.load_img(path, target_size=self.img_size)
            # create a feature map using the supplied model to act as both input and expected output
            X[i] = gen_feature_map(img, self.FE_model)

        return X, X

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

class lm_AE_generator(Sequence):
    '''
        A generator (keras sequence) for making a feature map autoencoder using the linemod dataset
    '''
    def __init__(self, batch_size, lm_path, img_size, shuffle, FE_model, FM_shape):
        self.img_paths = []
        # get all images in each subdirectory
        for collection in os.listdir(lm_path):
            image_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'rgb' + os.path.sep
            image_paths = [image_dir_path + img_path for img_path in os.listdir(image_dir_path)]
            self.img_paths.extend(image_paths)
        self.img_size = img_size
        self.shuffle = shuffle
        self.FE_model = FE_model
        self.FM_shape = FM_shape
        self.num_images = len(self.img_paths)
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.empty((self.batch_size, *self.img_size, 3))

        for i, path in enumerate(img_paths_tmp):
            # load image into memory
            img = tf.keras.utils.load_img(path, target_size=self.img_size)
            # create a feature map using the supplied model to act as both input and expected output
            imgs[i] = np.array(img)
        X = gen_feature_map(imgs, self.FE_model)
        return X, X

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

class lm_mask_generator(Sequence):
    '''
        A generator (keras sequence) for reconstructing visible object segmentation
        masks using the linemod dataset
    '''
    def __init__(self, batch_size, lm_path, img_size, shuffle, num_models, fm_shape, FE_model):
        self.img_paths = []
        self.FE_model = FE_model
        self.fm_shape = fm_shape
        self.mask_paths = {}
        # find each rbg image in each subdirctory
        for collection in os.listdir(lm_path):
            image_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'rgb' + os.path.sep
            image_paths = [image_dir_path + img_path for img_path in os.listdir(image_dir_path)]
            # find the directory with the matching visible mask s
            mask_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'mask_visib' + os.path.sep
            # for each mask in that directory
            for path in image_paths:
                folder, file = os.path.split(path)
                # find the image number number
                file_num = file.split('.')[0]
                masks_for_img = []
                # then find all visible model masks for that number
                for model_num in range(num_models):
                    mask_path = mask_dir_path + file_num + '_{:06d}.png'.format(model_num)
                    if os.path.isfile(mask_path):
                        # save the path to the mask, as well as the model number it is for
                        masks_for_img.append([mask_path, model_num])
                self.mask_paths[path] = masks_for_img
            self.img_paths.extend(image_paths)
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_shape = (*img_size, 3)
        self.num_images = len(self.img_paths)
        self.batch_size = batch_size
        self.num_models = num_models
        self.mask_shape = (*img_size, num_models)

        background_weight = np.ones((1,))
        class_weights = np.ones((self.num_models,)) * 100
        self.class_weights = tf.constant(np.concatenate([background_weight, class_weights]))
        #self.class_weights = self.class_weights / tf.reduce_sum(self.class_weights)
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.zeros((self.batch_size, *self.img_size, 3))
        masks = np.zeros((self.batch_size, *self.img_size, self.num_models+1))
        masks[:, :, :, 0] = np.ones((self.batch_size, *self.img_size))
        sample_weights = np.zeros((self.batch_size, *self.img_size, 1))
        for image_num, path in enumerate(img_paths_tmp):
            # load the rgb
            img = load_img(path, target_size=self.img_size)
            #imgs[image_num] = gen_feature_map(img, self.FE_model)
            imgs[image_num] = np.array(img)
            # load a number corresponding to the model number into each pixel of the image covered by the mask
            for mask_path, model_num in self.mask_paths[path]:
                mask = load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
                mask = np.array(mask)
                masks[image_num, mask == 255.0, model_num+1] = 1
                masks[0, mask == 255.0, 0] = 0 # remove ones from background where the mask is
            mask_as_sparse = masks[image_num].argmax(axis=-1)
            sample_weights[image_num, :, :, 0] = tf.gather(self.class_weights, indices=mask_as_sparse.astype(np.int32))
            #sample_weights[image_num] = 100 * np.count_nonzero(masks[image_num]) / masks[image_num].size
        #X = gen_feature_map(imgs, self.FE_model)
        return imgs, masks, sample_weights

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y, sample_weights = self.__data_generation(list_IDs_temp)

        return X, y, sample_weights

class lm_direction_generator(Sequence):
    '''
        A generator (keras sequence) for reconstructing the direction and orientation of each image using the linemod dataset
    '''
    def __init__(self, batch_size, lm_path, img_size, shuffle, num_models, seg_model, intrinsic):
        self.K = intrinsic
        self.img_paths = []
        self.seg_model = seg_model
        self.seg_shape = (*img_size, num_models+1)
        self.mask_paths = {}
        #self.depth_paths = []
        # find each rbg image in each subdirctory
        for collection in os.listdir(lm_path):
            image_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'rgb' + os.path.sep
            image_paths = [image_dir_path + img_path for img_path in os.listdir(image_dir_path)]
            # find the directory with the matching visible masks
            mask_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'mask_visib' + os.path.sep
            # for each mask in that directory
            for path in image_paths:
                folder, file = os.path.split(path)
                # find the image number number
                file_num = file.split('.')[0]
                masks_for_img = []
                # then find all visible model masks for that number
                for model_num in range(num_models):
                    mask_path = mask_dir_path + file_num + '_{:06d}.png'.format(model_num)
                    if os.path.isfile(mask_path):
                        # save the path to the mask, as well as the model number it is for
                        masks_for_img.append([mask_path, model_num])
                self.mask_paths[path] = masks_for_img
            self.img_paths.extend(image_paths)

            # depth_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'depth' + os.path.sep
            # depth_paths = [depth_dir_path + depth_path for depth_path in os.listdir(depth_dir_path)]
            # self.depth_paths.extend(depth_paths)
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_shape = (*img_size, 3)
        self.num_images = len(self.img_paths)
        self.batch_size = batch_size
        self.num_models = num_models
        self.mask_shape = (*img_size, num_models)
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.zeros((self.batch_size, *self.img_size, 3))
        masks = np.zeros((self.batch_size, *self.img_size, 1))
        directions = np.zeros((self.batch_size, *self.img_size, self.num_models*3))

        for image_num, path in enumerate(img_paths_tmp):
            # load the rgb
            img = load_img(path, target_size=self.img_size)
            img = np.array(img)
            imgs[image_num] = img

            path_parts = path.split('/')
            depth_path = os.path.sep.join(path_parts[:-2]) + os.path.sep + 'depth' + os.path.sep + path_parts[-1]
            depth_img = tf.keras.utils.load_img(depth_path, target_size=self.img_size, color_mode='grayscale')
            point_cloud = depth_to_point_cloud(depth_img, point_cloud)
            # load a number corresponding to the model number into each pixel of the image covered by the mask
            for mask_path, model_num in self.mask_paths[path]:
                mask = load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
                mask = np.array(mask)
                masks[image_num, mask == 255.0] = model_num+1 # offset by 1 as 0 is background
                #masks[0, mask] = 0 # remove ones from background where the mask is

        return imgs, masks

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

class lm_complete_generator(Sequence):
    '''
        A generator (keras sequence) for reconstructing the direction and orientation of each image using the linemod dataset
    '''
    def __init__(self, batch_size, lm_path, img_size, shuffle, num_models):
        self.img_paths = []
        #self.FE_model = FE_model
        #self.fm_shape = fm_shape
        self.mask_paths = {}
        self.depth_paths = []
        # find each rbg image in each subdirctory
        for collection in os.listdir(lm_path):
            image_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'rgb' + os.path.sep
            image_paths = [image_dir_path + img_path for img_path in os.listdir(image_dir_path)]
            # find the directory with the matching visible masks
            mask_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'mask_visib' + os.path.sep
            # for each mask in that directory
            for path in image_paths:
                folder, file = os.path.split(path)
                # find the image number number
                file_num = file.split('.')[0]
                masks_for_img = []
                # then find all visible model masks for that number
                for model_num in range(num_models):
                    mask_path = mask_dir_path + file_num + '_{:06d}.png'.format(model_num)
                    if os.path.isfile(mask_path):
                        # save the path to the mask, as well as the model number it is for
                        masks_for_img.append([mask_path, model_num])
                self.mask_paths[path] = masks_for_img
            self.img_paths.extend(image_paths)

            mask_dir_path = lm_path + os.path.sep + collection + os.path.sep + 'depth' + os.path.sep
            depth_paths = [mask_dir_path + depth_path for depth_path in os.listdir(mask_dir_path)]
            self.depth_paths.extend(depth_paths)
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_shape = (*img_size, 3)
        self.num_images = len(self.img_paths)
        self.batch_size = batch_size
        self.num_models = num_models
        self.mask_shape = (*img_size, num_models)
        background_weight = np.ones((1,))
        class_weights = np.ones((self.num_models,)) * 100
        self.class_weights = tf.constant(np.concatenate([class_weights, background_weight]))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_tmp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.zeros((self.batch_size, *self.img_size, 3))
        masks = np.zeros((self.batch_size, *self.img_size, 1)) * (self.num_models)
        sample_weights = np.zeros((self.batch_size, *self.img_size, 1))

        for image_num, path in enumerate(img_paths_tmp):
            # load the rgb
            img = tf.keras.utils.load_img(path, target_size=self.img_size)
            img = np.array(img)
            imgs[image_num] = img
            # load a number corresponding to the model number into each pixel of the image covered by the mask
            for mask_path, model_num in self.mask_paths[path]:
                mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
                mask = np.array(mask)
                masks[image_num, mask == 255.0] = model_num+1 # offset by 1 as 0 is background
                #masks[0, mask] = 0 # remove ones from background where the mask is
            sample_weights[image_num] = tf.gather(self.class_weights, indices=masks[image_num].astype(np.int32))
        #masks.shape = (self.batch_size, -1)
        return imgs, masks, sample_weights

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y, sample_weights = self.__data_generation(list_IDs_temp)

        return X, y, sample_weights


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

class ycbm_dist_generator(Sequence):
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
        imgs = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        dists = np.zeros((self.batch_size, *self.img_size, 1))
        masks = np.zeros((self.batch_size, *self.img_size, self.num_models + 1))
        sample_weights = np.ones((self.batch_size, *self.img_size, 1))

        for sample_num, load_info in enumerate(img_paths_tmp):
            # load the rgb
            setup, item = load_info
            camera_type, snap_vs_traj, item = item
            img_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
            img_path += item + '.jpg'
            img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            img = np.array(img)
            img = img.astype(np.float32)


            dist_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
            dist_path += item + '.depth.png'
            dist = tf.keras.utils.load_img(dist_path, target_size=self.img_size, color_mode='grayscale')
            dist = np.array(dist)
            dist = dist.astype(np.float32)

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

                    dist = cv2.warpAffine(dist, M, self.img_size, flags=cv2.INTER_NEAREST)
                    mask = cv2.warpAffine(mask, M, self.img_size, flags=cv2.INTER_NEAREST)
                    img = cv2.warpAffine(img, M, self.img_size)

                    if (np.any(dist) and np.any(img)):
                        done_augment = True
                    else:
                        warnings.warn('Improper warp in '+img_path+' all zeros', RuntimeWarning)

                        img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
                        img = np.array(img)
                        img = img.astype(np.float32)

                        dist = tf.keras.utils.load_img(dist_path, target_size=self.img_size, color_mode='grayscale')
                        dist = np.array(dist)
                        dist = dist.astype(np.float32)

                        mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
                        mask = np.array(mask)


            imgs[sample_num] = img
            dists[sample_num, :, :, 0] = dist
            for class_id in range(self.num_models+1):
                masks[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id], class_id] = 1
                sample_weights[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id]] = self.class_weights[class_id]

        return imgs, [dists, masks], sample_weights


class ycbm_mask_comb_generator(Sequence):
    def __init__(self, batch_size, ycbm_path, img_size, shuffle, num_models, augment, camera_types,
                 base_model_path, num_views=1):
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
        self.num_views = num_views

        base_model = tf.keras.models.load_model(base_model_path, compile=False)
        base_model._name = 'base_model'

        FE_model = tf.keras.Model(base_model.get_layer('model').input,
                                  base_model.get_layer('model').output,
                                  name='GT')

        for layer in FE_model.layers:
            layer.trainable = False

        embedding_model = tf.keras.Model(base_model.get_layer('embedding').input,
                                         base_model.get_layer('embedding').output,
                                         name='embedded_rep')

        for layer in embedding_model.layers:
            layer.trainable = False

        img_input_layer = tf.keras.layers.Input(shape=base_model.input_shape[1:], name='gt_input')

        FM3, FM2, FM1 = FE_model(img_input_layer)
        embedded_2d = embedding_model([FM1, FM2, FM3])

        self.FE_model = tf.keras.Model(img_input_layer, embedded_2d, name='gt_gen')

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
        imgs_in = [np.zeros((self.batch_size, *self.img_size, 3))] * self.num_views
        masks = np.zeros((self.batch_size, *self.img_size, self.num_models+1))
        out_FM = np.zeros((self.batch_size, *self.FE_model.output_shape[1:]))
        sample_weights_img = np.zeros((self.batch_size, *self.img_size, 1))
        #sample_weights_fm = np.ones((self.batch_size, *self.FE_model.output_shape[1:]))

        in_extrinsics = [np.zeros((self.batch_size, 7))] * self.num_views
        out_extrinsics = np.zeros((self.batch_size, 7))

        for sample_num, load_info in enumerate(img_paths_tmp):
            # load the rgb
            setup, item = load_info
            camera_type, snap_vs_traj, item = item

            img_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
            included_views = []
            all_views_for_setup = os.listdir(img_path)
            for view in range(self.num_views):
                img_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep
                rand_view = '_'
                while '_' in rand_view or rand_view == item or rand_view in included_views:
                    rand_view = self.rng.choice(all_views_for_setup)
                    rand_view = rand_view.split('.')[0]

                img_path += rand_view + '.jpg'
                img_json_path = img_path[:-4] + '.json'
                img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
                img = np.array(img)/255

                imgs_in[view][sample_num] = img

                in_json_file = open(img_json_path, 'r')
                in_json = json.load(in_json_file)
                in_camera_loc = in_json['camera_data']['location_worldframe']
                in_camera_rot = in_json['camera_data']['quaternion_xyzw_worldframe']
                in_json_file.close()
                in_extrinsics[view][sample_num, :3] = in_camera_loc
                in_extrinsics[view][sample_num, 3:] = in_camera_rot


            mask_path = self.ycbm_path + os.path.sep + camera_type + os.path.sep + setup + os.path.sep + snap_vs_traj + os.path.sep

            mask_img_path = mask_path + item + '.jpg'
            mask_img = tf.keras.utils.load_img(mask_img_path, target_size=self.img_size)
            mask_img = np.array(mask_img)/255
            mask_img = mask_img[None, :, :, :]

            mask_fm = self.FE_model(mask_img)
            out_FM[sample_num] = mask_fm

            mask_path = mask_path + item + '.seg.png'
            mask_json_path = mask_path[:-8] + '.json'
            mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
            mask = np.array(mask)

            # done_augment = False
            # if self.augment:
            #     while done_augment == False:
            #         scale = 1
            #         if self.rng.uniform(0, 1) > 0.5:
            #             scale = np.random.uniform(self.min_scale, 0.5+self.min_scale)
            #
            #         X_trans = 0
            #         Y_trans = 0
            #         if self.rng.uniform(0, 1) > 0.5:
            #             X_trans = self.rng.uniform(-self.max_trans, self.max_trans)
            #             Y_trans = self.rng.uniform(-self.max_trans, self.max_trans)
            #
            #         theta = 0
            #         if self.rng.uniform(0, 1) > 0.5:
            #             theta = self.rng.uniform(-180, 180)
            #
            #         M = cv2.getRotationMatrix2D((self.img_center[0]+X_trans, self.img_center[1]+Y_trans),
            #                                     scale=scale,
            #                                     angle=theta)
            #
            #         mask = cv2.warpAffine(mask, M, self.img_size, flags=cv2.INTER_NEAREST)
            #         img = cv2.warpAffine(img, M, self.img_size)
            #
            #         if (np.any(mask) and np.any(img)):
            #             done_augment = True
            #         else:
            #             warnings.warn('Improper warp in '+img_path+' all zeros', RuntimeWarning)
            #
            #             img = tf.keras.utils.load_img(img_path, target_size=self.img_size)
            #             img = np.array(img) / 255
            #             mask = tf.keras.utils.load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
            #             mask = np.array(mask)

            for class_id in range(self.num_models+1):
                masks[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id], class_id] = 1
                sample_weights_img[sample_num, mask == YCBM_CLASS_TO_SEG_VAL[class_id]] = self.class_weights[class_id]

            out_json_file = open(mask_json_path, 'r')
            out_json = json.load(out_json_file)
            out_camera_loc = out_json['camera_data']['location_worldframe']
            out_camera_rot = out_json['camera_data']['quaternion_xyzw_worldframe']
            out_json_file.close()
            out_extrinsics[sample_num, :3] = out_camera_loc
            out_extrinsics[sample_num, 3:] = out_camera_rot

        out_FM_weighted = out_FM.copy()
        out_FM_weighted[out_FM == 0] = 1E-5
        out_FM_weighted[out_FM == np.nan] = 1E-5

        extrinsic_diff = np.inf
        for in_extr in in_extrinsics:
            loc = in_extr[sample_num, :3]
            dist = np.square(loc - out_camera_loc).sum()
            if dist < extrinsic_diff:
                extrinsic_diff = dist
                #print('extrinsic_diff', extrinsic_diff)
        extrinsic_diff /= 1000
        if extrinsic_diff == 0:
            extrinsic_diff = 1

        #print('extrinsic_diff', extrinsic_diff)

        sample_weights = {'classify': sample_weights_img * extrinsic_diff,
                          'decoded_2d': (np.abs(out_FM_weighted*10)+(out_FM_weighted.mean()/10)) * extrinsic_diff}

        #print('out_fm', out_FM_weighted.max(), out_FM_weighted.min())

        return [*imgs_in, *in_extrinsics, out_extrinsics], [masks, out_FM_weighted], sample_weights