#!/usr/bin/env python3
import json
import logging.config
import os
import sys
import time
from argparse import ArgumentParser
from datetime import timedelta
from importlib import import_module
from signal import SIGINT, SIGTERM
from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf

import common
import lbtoolbox as lb
import loss

parser = ArgumentParser(description='Train a ReID network.')

# Required.

parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_dataset', required=True,
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', required=True, type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional with same defaults.

parser.add_argument(
    '--resnet_stride', required=False, default=2, choices=('1','2'),
    help='Resnet stride on blocks 3 and 4')

parser.add_argument(
    '--query_dataset', type=common.writeable_directory,
    help='Csv file with query dataset')

parser.add_argument(
    '--gallery_dataset', type=common.writeable_directory,
    help='Csv file with test dataset')

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

parser.add_argument(
    '--lr_decay', default=0.001,
    help='The rate of the exponential decay of the learning rate')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default='resnet_v1_50.ckpt',
    help='Path to the checkpoint file of the pretrained network.')

parser.add_argument(
    '--batch_p', default=32, type=common.positive_int,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--batch_k', default=4, type=common.positive_int,
    help='The numberK used in the PK-batches')

parser.add_argument(
    '--net_input_height', default=256, type=common.positive_int,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=128, type=common.positive_int,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--pre_crop_height', default=288, type=common.positive_int,
    help='Height used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--pre_crop_width', default=144, type=common.positive_int,
    help='Width used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--alpha1', default=0.05, type=float,
    help='Constant which is used for quadruplet loss function. Alpha1 means'
         'ratio of the main part of the equation.')

parser.add_argument(
    '--alpha2', default=0.5, type=float,
    help='Constant which is used for quadruplet loss function. Alpha2 means'
         'ratio of the second part of the equation.')

parser.add_argument(
    '--alpha3', default=0.5, type=float,
    help='Constant which is used for new quadruplet loss function. Alpha3 means'
         'ratio of the third part of the equation.')

parser.add_argument(
    '--metric', default='euclidean', choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--learning_rate', default=3e-4, type=common.positive_float,
    help='The initial value of the learning-rate, before it kicks in.')

parser.add_argument(
    '--train_iterations', default=35000, type=common.positive_int,
    help='Number of training iterations.')

parser.add_argument(
    '--decay_start_iteration', default=8000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument(
    '--checkpoint_frequency', default=5000, type=common.nonnegative_int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')

parser.add_argument(
    '--flip_augment', action='store_false', default=True,
    help='When this flag is provided, flip augmentation is not performed.')

parser.add_argument(
    '--crop_augment', action='store_false', default=True,
    help='When this flag is provided, crop augmentation is not performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')

parser.add_argument(
    '--detailed_logs', action='store_true', default=False,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')

parser.add_argument(
    '--gpu', default=0, type=int,
    help='ID of GPU which should be use for running.')

parser.add_argument(
    '--dropout', default=None, type=str,
    help='Layers that include dropout. '
         'Example: "blockNumber-dropoutProb" = "4-0.5/3-1"')

parser.add_argument(
    '--b4_layers', default= 1,
    help='Number of layers in the final block of ResNet (has to be in [1,2,3]')

parser.add_argument(
    '--sgdr', default=False, action='store_true', help='Determines the use of SGDR: https://arxiv.org/pdf/1608.03983.pdf'
)

def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    # possible_fids = relative paths to the images of the same pid (same person)
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0] # We found "count" number of figures of the person with the given pid
    padded_count = tf.cast(tf.ceil(batch_k / count), tf.int32) * count  # Rounding to a multiple of "count", upwards (for batch=18 and count=10 it rounds to 20)
    full_range = tf.mod(tf.range(padded_count), count) #Stores the indices of our figures

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range) #Shuffling the indices
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k]) # Taking the first "batch" number of figures according to their shuffled indices
                                                                # Therefore we ensure that we always get "batch" number of shuffled figures of the same
                                                                # person (of course if count<batch then some will be duplicates

    return selected_fids, tf.fill([batch_k], pid)

def prepare_data(args):
    '''
    Data preparation for training

    :param args: all stored arguments
    :return: images: prepared images for training
            fid: figure id which means relative paths of images
            pid: person id (or car id) of each image
    '''
    # Load the data from the CSV file.
    # pids - person id (array corresponding to the images)
    # fids - array of the paths to the images ({str_})
    pids, fids = common.load_dataset(args.train_dataset, args.image_root, False)
    max_fid_len = max(map(len, fids))  # We'll need this later for logfiles.

    # Setup a tf.Dataset where one "epoch" loops over all PIDS.
    # PIDS are shuffled after every epoch and continue indefinitely.
    unique_pids = np.unique(pids)
    # dataset.output_types = float32, dataset.output_shape = len(unique_pids)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    # For every PID, get K images.
    # default is args.batch_k = 4 (so it takes 4 images per each person)
    # dataset has len(unique_pids) tensors
    # create each tensor = (tensor_of_k_fids, tensor_of_pid)
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))

    # Ungroup/flatten the batches for easy loading of the files.
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    dataset = dataset.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Augment the data if specified by the arguments.
    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    images, fids, pids = dataset.make_one_shot_iterator().get_next()
    return [images, fids, pids, max_fid_len]

def getDropoutProbs(args):
    drl = args.split("/")
    drops = dict()
    for i in drl:
        a, b = i.split("-")
        try:
            drops[int(a)] = float(b)
        except ValueError:
            raise Exception("Invalid dropout parameters")

    if not set(drops.keys()).issubset([3, 4]):
        raise Exception("Invalid dropout parameters")
    if all(x < 0 and x > 1 for x in drops.values()):
        raise Exception("Invalid dropout parameters: "+str(drops.values()))
    return drops

def train(args, images, fids, pids, max_fid_len, log):
    '''
    Creation model and training neural network

    :param args: all stored arguments
    :param images: prepared images for training
    :param fids: figure id (relative paths from image_root to images)
    :param pids: person id (or car id) for all images
    :param log: log file, where logs from training are stored
    :return: saved files (checkpoints, train log file)
    '''
    ###################################################################################################################
    # CREATE MODEL
    ###################################################################################################################
    # Create the model and an embedding head.

    model = import_module('nets.resnet_v1_50')
    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    drops = {}
    if args.dropout is not None:
        drops = getDropoutProbs(args.dropout)
    b4_layers = None
    try:
        b4_layers = int(args.b4_layers)
        if b4_layers not in [1, 2, 3]: raise ValueError()
    except: ValueError("Argument exception: b4_layers has to be in [1, 2, 3]")

    endpoints, body_prefix = model.endpoints(images, b4_layers, drops, is_training=True, resnet_stride=int(args.resnet_stride))
    endpoints['emb'] = endpoints['emb_raw'] = slim.fully_connected(
        endpoints['model_output'], args.embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')

    step_pl = tf.placeholder(dtype=tf.float32)

    features = endpoints['emb']

    # Create the loss in two steps:
    # 1. Compute all pairwise distances according to the specified metric.
    # 2. For each anchor along the first dimension, compute its loss.
    dists = loss.cdist(features, features, metric=args.metric)
    losses, train_top1, prec_at_k, _, probe_neg_dists, pos_dists, neg_dists = loss.loss_function(
        dists, pids, [args.alpha1, args.alpha2, args.alpha3], batch_precision_at_k=args.batch_k - 1)

    # Count the number of active entries, and compute the total batch loss.
    num_active = tf.reduce_sum(tf.cast(tf.greater(losses, 1e-5), tf.float32))
    loss_mean = tf.reduce_mean(losses)

    # Some logging for tensorboard.
    tf.summary.histogram('loss_distribution', losses)
    tf.summary.scalar('loss', loss_mean)
    tf.summary.scalar('batch_top1', train_top1)
    tf.summary.scalar('batch_prec_at_{}'.format(args.batch_k - 1), prec_at_k)
    tf.summary.scalar('active_count', num_active)
    tf.summary.scalar('embedding_pos_dists', tf.reduce_mean(pos_dists))
    tf.summary.scalar('embedding_probe_neg_dists', tf.reduce_mean(probe_neg_dists))
    tf.summary.scalar('embedding_neg_dists', tf.reduce_mean(neg_dists))
    tf.summary.histogram('embedding_dists', dists)
    tf.summary.histogram('embedding_pos_dists', pos_dists)
    tf.summary.histogram('embedding_probe_neg_dists', probe_neg_dists)
    tf.summary.histogram('embedding_neg_dists', neg_dists)
    tf.summary.histogram('embedding_lengths',
                         tf.norm(endpoints['emb_raw'], axis=1))


    # Create the mem-mapped arrays in which we'll log all training detail in
    # addition to tensorboard, because tensorboard is annoying for detailed
    # inspection and actually discards data in histogram summaries.
    batch_size = args.batch_p * args.batch_k
    if args.detailed_logs:
        log_embs = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'embeddings'),
            dtype=np.float32, shape=(args.train_iterations, batch_size, args.embedding_dim))
        log_loss = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'losses'),
            dtype=np.float32, shape=(args.train_iterations, batch_size))
        log_fids = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'fids'),
            dtype='S' + str(max_fid_len), shape=(args.train_iterations, batch_size))

    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    # Define the optimizer and the learning-rate schedule.
    # Unfortunately, we get NaNs if we don't handle no-decay separately.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if args.sgdr:
        learning_rate = tf.train.cosine_decay_restarts(
            learning_rate=args.learning_rate,
            global_step=global_step,
            first_decay_steps=4000,
            t_mul=1.5)
    else:
        if 0 <= args.decay_start_iteration < args.train_iterations:
            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                tf.maximum(0, global_step - args.decay_start_iteration),
                args.train_iterations - args.decay_start_iteration, float(args.lr_decay))
        else:
            learning_rate = args.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(tf.convert_to_tensor(learning_rate))


    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.
    checkpoint_saver = tf.train.Saver(max_to_keep=0)
    with tf.Session() as sess:
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
            log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
            checkpoint_saver.restore(sess, last_checkpoint)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            sess.run(tf.global_variables_initializer())
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver(model_variables)
                saver.restore(sess, args.initial_checkpoint)

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess.graph)

        start_step = sess.run(global_step)
        step = start_step
        log.info('Starting training from iteration {}.'.format(start_step))

        ###################################################################################################################
        # TRAINING
        ###################################################################################################################
        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):
                # Compute gradients, update weights, store logs!
                start_time = time.time()
                _, summary, step, b_prec_at_k, b_embs, b_loss, b_fids = \
                    sess.run([train_op, merged_summary, global_step,
                              prec_at_k, features, losses, fids], feed_dict={step_pl: step})
                elapsed_time = time.time() - start_time

                # Compute the iteration speed and add it to the summary.
                # We did observe some weird spikes that we couldn't track down.
                summary2 = tf.Summary()
                summary2.value.add(tag='secs_per_iter', simple_value=elapsed_time)
                summary_writer.add_summary(summary2, step)
                summary_writer.add_summary(summary, step)

                if args.detailed_logs:
                    log_embs[i], log_loss[i], log_fids[i] = b_embs, b_loss, b_fids

                # Do a huge print out of the current progress. Maybe steal from here.
                seconds_todo = (args.train_iterations - step) * elapsed_time
                log.info('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                         'batch-p@{}: {:.2%}, ETA: {} ({:.2f}s/it), lr={:.4g}'.format(
                    step,
                    float(np.min(b_loss)),
                    float(np.mean(b_loss)),
                    float(np.max(b_loss)),
                    args.batch_k - 1, float(b_prec_at_k),
                    timedelta(seconds=int(seconds_todo)),
                    elapsed_time,
                    sess.run(optimizer._lr)))
                sys.stdout.flush()
                sys.stderr.flush()


                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and
                        step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)

def main():
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.set_random_seed(0)
    ###################################################################################################################
    # RESTORE ARGUMENTS
    ###################################################################################################################
    # We store all arguments in a json file. This has two advantages:
    # 1. We can always get back and see what exactly
    # that experiment was
    # 2. We can resume an experiment as-is without needing to remember all flags.
    args_file = os.path.join(args.experiment_root, 'args.json')
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))

        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        args_resumed['resume'] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        # values from the file, but we also want to check for some possible
        # conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value:
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    args.__dict__[key] = resumed_value
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))

    else:
        # If the experiment directory exists already, we bail in fear.
        if os.path.exists(args.experiment_root):
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume to'
                      ' your call.'.format(args.experiment_root))
                exit(1)
        else:
            os.makedirs(args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))

    # Check them here, so they are not required when --resume-ing.
    if not args.train_dataset:
        parser.print_help()
        log.error("You did not specify the `train_set` argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You did not specify the required `image_root` argument!")
        sys.exit(1)

    images, fids, pids, max_fid_len = prepare_data(args)
    train(args, images, fids, pids,max_fid_len,  log)


if __name__ == '__main__':
    main()
