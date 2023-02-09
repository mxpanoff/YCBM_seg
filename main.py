import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import os
from scripts.utils.common_utils import seg_out_to_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to model to load')
    parser.add_argument('img_dir_path', type=str, help='path to directory with images to segment')

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False)
    converter = seg_out_to_img(model.output_shape[-1])

    assert os.path.isdir(args.img_dir_path), 'Error {} is not a valid directory'.format(args.img_dir_path)

    for f in os.listdir(args.img_dir_path):
        if f[-4:] not in ['.png', '.PNG', '.jpg', '.JPG']:
            print('Skipping', f)
            continue
        img = tf.keras.utils.load_img(args.img_dir_path + os.path.sep + f, target_size=model.input_shape[1:])
        img = np.asarray(img) / 255
        img = np.expand_dims(img, 0)

        out = model(img)
        out = np.asarray(out)

        out = converter.transform_seg(out)

        plt.imshow(out[0])
        name = os.path.split(f)[-1]
        name = name.split('.')[0]
        plt.savefig('out_segs' + os.path.sep + name + '.png')
