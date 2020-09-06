import utils
import tensorflow as tf

def mmd_penalty(sample_qz, sample_pz, hps):
    sigma2_p = hps.pz_scale ** 2
    n = hps.batch_size
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size =  tf.cast((n * n - n) / 2, tf.int32)
    kernel = hps.wae_kernel

    # pairwise distances

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    if kernel == 'RBF':
        #index_to_take = half_size - 1 if hps.batch_size > 1 else 0
        if hps.batch_size == 1: # for testing purposes
            sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values
            sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values
        # Median heuristic for the sigma^2 of Gaussian kernel - prende la distanza mediana
        else: 
            sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel - prende la distanza maggiore
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif kernel == 'IMQ':
        # k(x, y) = C / (C + ||x - y||^2)
        Cbase = 2. * hps.z_size * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    return stat