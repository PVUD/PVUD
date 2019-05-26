#!/usr/bin/env python3
import csv
import json
import os
import time
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

import common
import loss

parser = ArgumentParser(description='Evaluate a ReID embedding.')

# Required

parser.add_argument(
    '--excluder', required=True, choices=('market1501', 'diagonal', 'PVUD', 'veri'),
    help='Excluder function to mask certain matches. Especially for multi-'
         'camera datasets, one often excludes pictures of the query person from'
         ' the gallery if it is taken from the same camera. The `diagonal`'
         ' excluder should be used if this is *not* required.')

parser.add_argument(
    '--query_dataset', required=True,
    help='Path to the query dataset csv file.')

parser.add_argument(
    '--query_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset csv file.')

parser.add_argument(
    '--gallery_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

# Optional

parser.add_argument(
    '--metric', default='euclidean', choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--output_name', type=str, default=None,
    help='Optional name of the output files to store the results in. Do not add the extension of the file.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on your memory usage.')

parser.add_argument(
    '--image_root', default=None, type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

parser.add_argument(
    '--gpu', default=0,
    help='ID of GPU which should be use for running.')


def evaluate_embs(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs):
    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs.shape[1]
    gallery_dim = gallery_embs.shape[1]
    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    if 'vehicleid' in args.dataset:
        gallery_pids = np.asarray(gallery_pids)
        gallery_fids = np.asarray(gallery_fids)
    else:
        excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)

    # We go through the queries in batches, but we always need the whole gallery
    batch_pids, batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_pids, query_fids, query_embs)
    ).batch(args.batch_size).make_one_shot_iterator().get_next()

    batch_distances = loss.cdist(batch_embs, gallery_embs, metric=args.metric)
    # Loop over the query embeddings and compute their APs and the CMC curve.
    aps = []
    correct_rank = []
    results_images = []
    scores_of_queries = []
    pid_matches_all = np.zeros(shape=(1, len(gallery_fids)))
    num_of_NaN = 0
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    num_of_paired_img = len(query_pids)
    if args.output_name is not None:
        text_file = open(os.path.join(args.experiment_root, "accuracy_summary_" + args.output_name + ".txt"), "w")
    else:
        text_file = open(os.path.join(args.experiment_root, "accuracy_summary.txt"), "w")
    with tf.Session() as sess:
        for start_idx in count(step=args.batch_size):
            try:
                # Compute distance to all gallery embeddings for the batch of queries
                distances, pids, fids = sess.run([batch_distances, batch_pids, batch_fids])
                print('\rEvaluating batch {}-{}/{}'.format(
                   start_idx, start_idx + len(fids), len(query_fids)),
                   flush=True, end='')
            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

            # Convert the array of objects back to array of strings
            pids, fids = np.array(pids, '|U'), np.array(fids, '|U')

            # Compute the pid matches
            pid_matches = gallery_pids[None] == pids[:, None]

            # Get a mask indicating True for those gallery entries that should
            # be ignored for whatever reason (same camera, junk, ...) and
            # exclude those in a way that doesn't affect CMC and mAP.
            if 'vehicleid' not in args.dataset:
                mask = excluder(fids)
                distances[mask] = np.inf
                pid_matches[mask] = False
            pid_matches_all = np.concatenate((pid_matches_all, pid_matches), axis=0)

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            num_of_col = 10
            for i in range(len(distances)):
                ap = average_precision_score(pid_matches[i], scores[i])
                sorted_distances_inds = np.argsort(distances[i])

                if np.isnan(ap):
                    print()
                    print(str(num_of_NaN) + ". WARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(fids[i]))
                    print("I'm excluding this person from eval and carrying on.")
                    print()
                    text = (str(
                        num_of_NaN) + ". WARNING: encountered an AP of NaN! Probably a person only appears once - {}\n".format(
                        fids[i]))
                    text_file.write(text)
                    correct_rank.append(-1)
                    results_images.append(gallery_fids[sorted_distances_inds[0:num_of_col]])
                    num_of_NaN += 1
                    num_of_paired_img -= 1
                    scores_of_queries.append(-1)
                    continue

                aps.append(ap)
                scores_of_queries.append(ap)
                # Find the first true match and increment the cmc data from there on.
                rank_k = np.where(pid_matches[i, sorted_distances_inds])[0][0]
                cmc[rank_k:] += 1
                # Save five more similar images to each of image and correct rank of each image
                if (len(gallery_fids) < num_of_col):
                    num_of_col = len(gallery_fids)
                correct_rank.append(rank_k)
                results_images.append(gallery_fids[sorted_distances_inds[0:num_of_col]])

    # Compute the actual cmc and mAP values
    cmc = cmc / num_of_paired_img
    mean_ap = np.mean(aps)

    # Save important data
    saveResults(args, results_images, np.argsort(scores_of_queries)[::-1], query_fids, 10)
    if args.output_name is not None:
        out_file = open(os.path.join(args.experiment_root, "evaluation_" + args.output_name + ".json"), "w")
        json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)},  out_file)
        out_file.close()
    else:
        out_file = open(os.path.join(args.experiment_root, "evaluation.json"), "w")
        json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, out_file)
        out_file.close()

    # Print out a short summary and save summary accuracy.
    if len(cmc) > 9:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(mean_ap, cmc[0], cmc[1],cmc[4], cmc[9]))
        text_file.write('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(mean_ap, cmc[0], cmc[1],cmc[4], cmc[9]))
    elif len(cmc) > 5:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%}'.format(mean_ap, cmc[0], cmc[1], cmc[4]))
        text_file.write('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%}'.format(mean_ap, cmc[0], cmc[1], cmc[4]))
    elif len(cmc) > 2:
        print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%}'.format(mean_ap, cmc[0], cmc[1]))
        text_file.write('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%}'.format(mean_ap, cmc[0], cmc[1]))
    else:
        print('mAP: {:.2%} | top-1: {:.2%}'.format(mean_ap, cmc[0]))
        text_file.write('mAP: {:.2%} | top-1: {:.2%}'.format(mean_ap, cmc[0]))
    text_file.close()

    return [mean_ap, cmc[0]]

def saveResults(args, results_images, array_of_queries, query_fids, num_of_col):
    # make csv file with all queries and for each query save ten best images from gallery
    row = -1
    array_output = np.empty(shape=(len(array_of_queries), num_of_col+1), dtype=np.dtype('U256'))
    for i in array_of_queries:
        # write into the first column the query image
        row += 1
        array_output[row,0] = query_fids[i]
        for j in range(0, num_of_col):
            # next column would be the 10 most similar images from gallery
            array_output[row, j + 1] = results_images[i][j]


    if args.output_name is not None:
        output_file = "sorted_results_" + args.output_name + ".csv"
    else:
        output_file = "sorted_results.csv"
    with open(os.path.join(args.experiment_root, output_file), 'wt') as csv_query:
        writer_q = csv.writer(csv_query, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer_q.writerows(array_output)

def run_evaluation_with_args(args):

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, args.image_root, False)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, args.image_root, False)

    # Load the two datasets fully into memory.
    with h5py.File(os.path.join(args.experiment_root, args.query_embeddings), 'r') as f_query:
        query_embs = np.array(f_query['emb'])
    with h5py.File(os.path.join(args.experiment_root, args.gallery_embeddings), 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])

    [mAP, rank1] = evaluate_embs(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs)

    return [mAP, rank1]

def run_evaluation(args, query_embs, gallery_embs):

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, args.image_root, False)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, args.image_root, False)

    [mAP, rank1] = evaluate_embs(args, query_pids, query_fids, query_embs, gallery_pids, gallery_fids, gallery_embs)
    return [mAP, rank1]

if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_evaluation_with_args(args)
    end = time.time()
    print('Time: ' + str(end - start))
