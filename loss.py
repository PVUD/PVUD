import sys

import numpy as np
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def  cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            a1, a2 = tf.nn.top_k(a, tf.shape(a)[1])
            b1, b2 = tf.nn.top_k(b, tf.shape(b)[1])
            diffs2 = tf.cast(all_diffs(a2, b2), dtype=tf.float32)
            return tf.reduce_sum(tf.square(diffs2), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        elif metric == 'cosine':
            return tf.losses.cosine_distance(a, b, axis=1)
        elif metric == 'spearman':
            a1, a2 = tf.nn.top_k(a, tf.shape(a)[1])
            b1, b2 = tf.nn.top_k(b, tf.shape(b)[1])
            diffs2 = tf.cast(all_diffs(a2, b2), dtype=tf.float32)
            return tf.reduce_sum(tf.square(diffs2), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
    'spearman', #no gradient
    'cosine',
]

def return_min_negatives(neg_matrix):
    num_of_min_negs = len(neg_matrix)
    negatives = np.sort(neg_matrix[neg_matrix>0])
    return negatives[0:num_of_min_negs]

def return_max_pos(pos_matrix):
    num_of_min_negs = len(pos_matrix)
    positives = np.sort(pos_matrix[pos_matrix > 0])
    positives = positives[::-1]
    return positives[0:num_of_min_negs]

def sort_and_divide_mining(matrix, sortUpwardly, divisor):
    if sortUpwardly:
        matrix[matrix == 0] = sys.maxsize
        matrix = np.sort(matrix)
        matrix[matrix == sys.maxsize] = 0.
    else:
        matrix = np.sort(matrix)
        matrix = matrix[:,::-1]
    matrix = matrix[:,0:int(len(matrix)/divisor)]
    vector_means = np.mean(matrix, 1)
    return vector_means

def loss_function(dists, pids, alpha, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        alpha: Vector with two values. [Alpha1, Alpha2], for ratio in the quadruplet equation.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """


    # matrices pid x pid of all images from mini-match
    same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                  tf.expand_dims(pids, axis=0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(pids)[0], dtype=tf.bool))
    neg_upper_triang_mask = tf.matrix_band_part(negative_mask, 0, -1)

    # vector of the distances which are the biggest or the lowest in the row of the matrix
    furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask, tf.float32), axis=1)
    # closest_probe_vs_neg = tf.boolean_mask(dists, negative_mask)
    closest_probe_vs_neg = tf.map_fn(lambda x: tf.reduce_min(tf.cast(tf.boolean_mask(x[0], x[1]), tf.float32)),
                                     (dists, negative_mask), tf.float32)

    # all furthest negatives
    all_furth_pos = dists * tf.cast(tf.matrix_band_part(positive_mask, 0, -1), tf.float32)
    furthest_positive2 = tf.py_func(return_max_pos, [all_furth_pos], tf.float32)
    # negatives regardless of whether pairs contain the same probe
    all_negatives = dists * tf.cast(neg_upper_triang_mask, tf.float32)
    closest_negatives = tf.py_func(return_min_negatives, [all_negatives], tf.float32)

    # taking diff between the worst cases
    diff1 = tf.nn.softplus(furthest_positive - closest_probe_vs_neg)
    diff2 = tf.pow(1.1, furthest_positive2) - 1
    diff3 = tf.divide(30, (closest_negatives + 3))
    diff = tf.multiply(alpha[0], diff1) + tf.multiply(alpha[1], diff2) + tf.multiply(alpha[2], diff3)

    if batch_precision_at_k is None:
        return diff

    # For monitoring, compute the within-batch top-1 accuracy and the
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)

        # Drop the diagonal (distance to self is always least).
        indices = indices[:,1:]

        # Generate the index indexing into the batch dimension.
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
            (1, tf.shape(indices)[1]))

        # Stitch the above together with the argsort indices to get the
        # indices of the top-k of each row.
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:,0])
        prec_at_k = tf.reduce_mean(topk_is_same_f32)

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        probe_negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, positive_mask)
        negative_dists = tf.boolean_mask(dists, neg_upper_triang_mask)

        return diff, top1, prec_at_k, topk_is_same, probe_negative_dists, positive_dists, negative_dists


