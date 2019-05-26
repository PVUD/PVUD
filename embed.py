#!/usr/bin/env python3

import os
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import common
from aggregators import AGGREGATORS

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset', required=True,
    help='Path to the dataset csv file to be embedded.')


parser.add_argument(
    '--image_root', required=True, type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel data loading.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

parser.add_argument(
    '--filename', default=None,
    help='Name of the HDF5 file in which to store the embeddings, relative to'
         ' the `experiment_root` location. If omitted, appends `_embeddings.h5`'
         ' to the dataset name.')

parser.add_argument(
    '--flip_augment', action='store_true', default=True,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', choices=['center', 'avgpool', 'five'], default='five',
    help='When this flag is provided, crop augmentation is performed.'
         '`avgpool` means the full image at the precrop size is used and '
         'the augmentation is performed by the average pooling. `center` means'
         'only the center crop is used and `five` means the four corner and '
         'center crops are used. When not provided, by default the image is '
         'resized to network input size.')

parser.add_argument(
    '--gpu', default=0,
    help='ID of GPU which should be use for running.')

parser.add_argument(
    '--dropout', default=None, type=str,
    help='Layers that include dropout. '
         'Example: "blockNumber-dropoutProb" = "4-0.5/3-1"')

parser.add_argument(
    '--b4_layers', default= 1, type=int,
    help='Number of layers in the final block of ResNet (has to be in [1,2,3]')

def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    # I changed dimension with tf
    # return images, [fid]*2, [pid]*2
    return images, tf.stack([fid]*2), tf.stack([pid]*2)


def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.floor_div(crop_margin, 2)
        bottom_right = tf.add(top_left, crop_size)
    center = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left = image[:-crop_margin[0], :-crop_margin[1]]
    top_right = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right

def calculate_emb_for_fids(args, data_fids):
    '''
    Calculate embeddings

    :param args: input arguments
    :param data_fids: relative paths to the images
    :return: matrix with shape len(data_fids) x embedding_dim (embedding vector for each image - one row)
    '''
    ###################################################################################################################
    # LOAD DATA
    ###################################################################################################################

    # Check a proper aggregator is provided if augmentation is used.
    if args.flip_augment or args.crop_augment == 'five':
        if args.aggregator is None:
            args.aggregator = 'mean'

    print('Evaluating using the following parameters:')
    for key, value in sorted(vars(args).items()):
        print('{}: {}'.format(key, value))

    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)


    ###################################################################################################################
    # PREPARE DATA
    ###################################################################################################################
    # Setup a tf Dataset containing all images.
    dataset = tf.data.Dataset.from_tensor_slices(data_fids)

    # Convert filenames to actual image tensors.
    # dataset tensor: [image_resized, fid, pid]
    dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, tf.constant("dummy", dtype=tf.string), image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Augment the data if specified by the arguments.
    # `modifiers` is a list of strings that keeps track of which augmentations
    # have been applied, so that a human can understand it later on.
    modifiers = ['original']
    if args.flip_augment:
        dataset = dataset.map(flip_augment)
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

    if args.crop_augment == 'center':
        dataset = dataset.map(lambda im, fid, pid:(five_crops(im, net_input_size)[0], fid, pid))
        modifiers = [o + '_center' for o in modifiers]
    elif args.crop_augment == 'five':
        dataset = dataset.map(lambda im, fid, pid:(tf.stack(five_crops(im, net_input_size)), tf.stack([fid] * 5), tf.stack([pid] * 5)))
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for o in modifiers for m in ['_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
    elif args.crop_augment == 'avgpool':
        modifiers = [o + '_avgpool' for o in modifiers]
    else:
        modifiers = [o + '_resize' for o in modifiers]

    # Group it back into PK batches.
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(1)

    images, _, _ = dataset.make_one_shot_iterator().get_next()

    ###################################################################################################################
    # CREATE MODEL
    ###################################################################################################################
    # Create the model and an embedding head.

    model = import_module('nets.' + args.model_name)
    b4_layers = None
    try:
        b4_layers = int(args.b4_layers)
        if b4_layers not in [1,2,3]: raise ValueError()
    except: ValueError("Argument exception: b4_layers has to be in [1,2,3]")
    endpoints, body_prefix = model.endpoints(images, b4_layers, drops={}, is_training=False, resnet_stride=int(args.resnet_stride))

    endpoints['emb'] = endpoints['emb_raw'] = slim.fully_connected(
        endpoints['model_output'], args.embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')

    with h5py.File(args.filename, 'w') as f_out, tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        if args.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        else:
            checkpoint = os.path.join(args.experiment_root, args.checkpoint)
        print('Restoring from checkpoint: {}'.format(checkpoint))
        tf.train.Saver().restore(sess, checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros(
            (len(data_fids) * len(modifiers), args.embedding_dim), np.float32)
        for start_idx in count(step=args.batch_size):
            try:
                embd = endpoints if args.model_name == 'xception' else endpoints['emb']

                emb = sess.run(embd)


                print('\rEmbedded batch {}-{}/{}'.format(
                        start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        print()
        print("Done with embedding, aggregating augmentations...", flush=True)

        if len(modifiers) > 1:
            # Pull out the augmentations into a separate first dimension.
            emb_storage = emb_storage.reshape(len(data_fids), len(modifiers), -1)
            emb_storage = emb_storage.transpose((1,0,2))  # (Aug,FID,128D)

            # Aggregate according to the specified parameter.
            emb_storage = AGGREGATORS['mean'](emb_storage)

        # Store information about the produced augmentation and in case no crop
        # augmentation was used, if the images are resized or avg pooled.
        f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))

    tf.reset_default_graph()
    return emb_storage

def calculate_emb(args):
    # autogenerated name of embedding output file
    if args.filename is None:
        basename = os.path.basename(args.dataset)
        args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'
    args.filename = os.path.join(args.experiment_root, args.filename)

    # Load the data from the CSV file.
    # pids - person id (array corresponding to the images)
    # fids - array of the paths to the images ({str_})
    data_pids, data_fids = common.load_dataset(args.dataset, args.image_root, False)

    return calculate_emb_for_fids(args, data_fids)

def run_embedding(stored_args, dataset):
    # required: experiment_root, dataset, image_root
    stored_args.dataset = dataset

    return calculate_emb(stored_args)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    calculate_emb(args)
