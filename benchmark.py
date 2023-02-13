#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The deep equilibrium net benchmark model:

This script provides the code used to model and solve the benchmark model in the working paper by
Azinovic, Gaegauf, & Scheidegger (2021) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482)
(see section 3). For a more streamlined application, see
https://github.com/sischei/DeepEquilibriumNets/blob/master/code/jupyter-notebooks/analytic/Analytic_tf1.ipynb.

Note that, this script was originally programmed in TensorFlow 1. The current default version of
TensorFlow is now TensorFlow 2. This script is TensorFlow 2 compatible. To install the correct
version, use
> pip install tensorflow

-----------------------------------------------------------------------------------------------------
There are two modes to run this code: 1) the final network weights presented in the paper can be
loaded and used to output a host of plots; 2) the deep equilibrium net can be trained from scratch.
We have simplified the code such that the only user input is the desired running mode. To run, follow
these instructions:

In terminal:
> cd '/DeepEquilibriumNets/code/python-scripts/benchmark'

Mode 1: Load the trained network weights
> python benchmark.py
The results are saved to ./output/deqn_benchmark_restart

Mode 2: Train from scratch
> python benchmark.py --train_from_scratch
The results are saved to ./output/deqn_benchmark

Note: the results presented in the paper (see, section 5) were achieved by training the neural
network on 2 training schedules. Once the first training schedule is complete (after running the
above command), uncomment lines 1314-1320 and run the previous command again
(python benchmark.py --train_from_scratch). The results are saved to
./output/deqn_benchmark_2ndschedule.
"""

import numpy.matlib
import json
import codecs
from utils import random_mini_batches
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

matplotlib.use('pdf')

print('tf version:', tf.__version__)

plt.rcParams.update({'font.size': 12})


def train(path_wd, run_name,
          num_episodes, len_episodes, epochs_per_episode,
          batch_size, optimizer_name, lr,
          save_interval, num_hidden_nodes, activations_hidden_nodes,
          train_flag=True, load_flag=False, load_run_name=None,
          load_episode=None, seed=1, save_raw_plot_data=False):

    train_dict = {}
    load_dict = {}
    train_setup_dict = {}
    econ_setup_dict = {}
    net_setup_dict = {}
    result_dict = {}
    params_dict = {}

    train_dict['seed'] = seed
    train_dict['identifier'] = run_name

    save_base_path = os.path.join('./output', run_name)
    log_dir = os.path.join(save_base_path, 'tensorboard')
    plot_dir = os.path.join(save_base_path, 'plots')

    train_setup_dict['num_episodes'] = num_episodes
    train_setup_dict['len_episodes'] = len_episodes
    train_setup_dict['epochs_per_episode'] = epochs_per_episode
    train_setup_dict['optimizer'] = optimizer_name
    train_setup_dict['batch_size'] = batch_size
    train_setup_dict['lr'] = lr

    train_dict['train_setup'] = train_setup_dict

    net_setup_dict['num_hidden_nodes'] = num_hidden_nodes
    net_setup_dict['activations_hidden_nodes'] = activations_hidden_nodes

    train_dict['net_setup'] = net_setup_dict

    load_dict['load_flag'] = load_flag
    load_dict['load_run_name'] = load_run_name
    load_dict['load_episode'] = load_episode

    train_dict['load_info'] = load_dict

    from nn_utils import Neural_Net

    if 'output' not in os.listdir():
        os.mkdir('./output')

    if run_name not in os.listdir('./output/'):
        os.mkdir(save_base_path)
        os.mkdir(os.path.join(save_base_path, 'json'))
        os.mkdir(os.path.join(save_base_path, 'model'))
        os.mkdir(os.path.join(save_base_path, 'plots'))
        os.mkdir(os.path.join(save_base_path, 'plotdata'))
        os.mkdir(os.path.join(save_base_path, 'tensorboard'))

    if 'tensorboard' in os.listdir(save_base_path):
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))

    # Set the seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # if the fraction of constraint agents should be printed
    print_frac_constrained = False

    # Global data parameters ==================================================
    NUM_EX_SHOCKS = 2
    assert NUM_EX_SHOCKS == 2, 'Two shocks hardcoded'

    BETA = 0.95
    SIGMA = 0.2
    S = 0.5
    Y = np.array([[0.05], [0.2]])
    R = 1.03
    A_LB = 0.0
    Z_UB = 1.00
    Z_LB = 0.00
    Q2 = 0.22
    Q1 = 0.18

    # A = 1
    # Transition
    GAMMA = np.array([[0.4, 0.6],
                      [0.1, 0.9]])

    print('Beta = ', BETA)
    print('Sigma = ', SIGMA)
    print('S = ', S)
    print('Y = ', Y)
    print('R = ', R)
    print('A_LB = ', A_LB)
    print('Z_UB = ', Z_UB)
    print('Z_LB = ', Z_LB)
    print('Q2 = ', Q2)
    print('Q1 = ', Q1)
    print('Gamma =', GAMMA)

    econ_setup_dict['y'] = Y.tolist()
    econ_setup_dict['beta'] = BETA
    econ_setup_dict['sigma'] = SIGMA
    econ_setup_dict['s'] = S
    econ_setup_dict['y'] = Y
    econ_setup_dict['r'] = R
    econ_setup_dict['zub'] = Z_UB
    econ_setup_dict['zlb'] = Z_LB
    econ_setup_dict['q2'] = Q2
    econ_setup_dict['q1'] = Q1
    econ_setup_dict['gamma'] = GAMMA.tolist()

    train_dict['econ_params'] = econ_setup_dict

    for key in econ_setup_dict:
        print('{}: {}'.format(key, econ_setup_dict[key]))

    with tf.name_scope('econ_parameters'):
        y = tf.constant(Y, dtype=tf.float32, name='y')
        gamma = tf.constant(GAMMA, dtype=tf.float32, name='gamma')
        beta = tf.constant(BETA, dtype=tf.float32, name='beta')
        sigma = tf.constant(SIGMA, dtype=tf.float32, name='sigma')
        s = tf.constant(S, dtype=tf.float32, name='s')
        r = tf.constant(R, dtype=tf.float32, name='r')
        q2 = tf.constant(Q2, dtype=tf.float32, name='q2')
        q1 = tf.constant(Q1, dtype=tf.float32, name='q1')
        alb = tf.constant(A_LB, dtype=tf.float32, name='alb')
        zub = tf.constant(Z_UB, dtype=tf.float32, name='zub')
        zlb = tf.constant(Z_LB, dtype=tf.float32, name='zlb')

    with tf.name_scope('neural_net'):
        # financial income, labor income, income, old_capital
        n_input = 1 + 2 + NUM_EX_SHOCKS
        # n_input = 8 + 4 * A + NUM_EX_SHOCKS
        # capital holding, capital multipler, kappa * bond holding + capital, bond multipler, bond price
        n_output = 3

        num_nodes = [n_input] + num_hidden_nodes + [n_output]
        activation_list = activations_hidden_nodes + [tf.nn.softplus]

        nn = Neural_Net(num_nodes, activation_list)

    X = tf.placeholder(tf.float32, shape=(None, n_input), name='X')

    with tf.name_scope('compute_cost'):
        eps = 0.00001

        # get number samples
        m = tf.shape(X)[0]

        with tf.name_scope('todays_consumption'):
            with tf.name_scope('decompose_state'):
                # get current state
                with tf.name_scope('exog_shock'):
                    # exogenous shock
                    y_t = X[:, 0]

                with tf.name_scope('asset'):
                    # individual financial wealth
                    a_t = X[:, 1]
                with tf.name_scope('location'):
                    # individual labor income endowment
                    z_t = X[:, 2]

                with tf.name_scope('income'):
                    # individual labor income endowment
                    inc = X[:, 3]

                with tf.name_scope('probs_tomorrow'):
                    # probabilities for shocks tomorrow
                    probs_next = X[:, 4: 4 + NUM_EX_SHOCKS]

            with tf.name_scope('get_todays_savings'):
                # get todays savings by executing the neural net
                with tf.name_scope('NN'):
                    predprime = nn.predict(X)
                    aprime = predprime[:, 0]
                    qprime = predprime[:, 1]
                    lambdaprime = predprime[:, 2]

            with tf.name_scope('kprime_wakeup_all'):
                kprime_wakeup_all = tf.concat(
                    [A_LB * tf.ones([m, 1]), kprime], axis=1)
                bprime_wakeup_all = tf.concat(
                    [tf.zeros([m, 1]), bprime], axis=1)

            with tf.name_scope('kprime_wakeup_all'):
                k_saved_all = tf.concat([kprime, tf.zeros([m, 1])], axis=1)
                bond_spent_all = tf.multiply(
                    tf.concat([bprime, tf.zeros([m, 1])], axis=1), p_mat)

            with tf.name_scope('adjustment_cost_all'):
                adjustment_all = (
                    k_saved_all - tf.reshape(R, [m, 1]) * k_wakeup_all)
                adj_cost_all = (zeta / 2.) * adjustment_all**2

            with tf.name_scope('compute_todays_consumption'):
                tot_saved_all = k_saved_all + bond_spent_all

                c_all_orig = inc - tot_saved_all - adj_cost_all
                c_all = tf.maximum(c_all_orig, tf.ones_like(
                    c_all_orig) * eps, name='c_all_today')

            with tf.name_scope('get_tomorrows_state'):
                with tf.name_scope('tomorrows_exog_shock'):
                    # state tomorrow
                    z1prime = tf.zeros_like(z, name='zprime_1')
                    z2prime = tf.ones_like(z, name='zprime_2')
                    z3prime = 2 * tf.ones_like(z, name='zprime_3')
                    z4prime = 3 * tf.ones_like(z, name='zprime_4')

                with tf.name_scope('tomorrows_exog_param'):
                    # tfp tomorrow
                    tfpprime_1 = tf.gather(xi, tf.cast(
                        z1prime, tf.int32), name='tfpprime_1')
                    tfpprime_2 = tf.gather(xi, tf.cast(
                        z2prime, tf.int32), name='tfpprime_2')
                    tfpprime_3 = tf.gather(xi, tf.cast(
                        z3prime, tf.int32), name='tfpprime_3')
                    tfpprime_4 = tf.gather(xi, tf.cast(
                        z4prime, tf.int32), name='tfpprime_4')

                    # depreciation tomorrow
                    deprprime_1 = tf.gather(delta, tf.cast(
                        z1prime, tf.int32), name='deprprime_1')
                    deprprime_2 = tf.gather(delta, tf.cast(
                        z2prime, tf.int32), name='deprprime_2')
                    deprprime_3 = tf.gather(delta, tf.cast(
                        z3prime, tf.int32), name='deprprime_3')
                    deprprime_4 = tf.gather(delta, tf.cast(
                        z4prime, tf.int32), name='deprprime_4')

                # tomorrows aggregate capital
                Kprime_orig = tf.reduce_sum(
                    kprime, axis=1, keepdims=True, name='Kprime_orig_tomorrow')
                Kprime = tf.maximum(Kprime_orig, tf.ones_like(
                    Kprime_orig) * eps, name='Kprime_tomorrow')

                with tf.name_scope('tomorrows_labor'):
                    # labor tomorrow
                    lprime_1 = tf.gather(l_mat, tf.cast(
                        z1prime, tf.int32), name='lprime_1')
                    Lprime_1 = tf.reduce_sum(
                        lprime_1, axis=1, keepdims=True, name='Lprime_1')
                    lprime_2 = tf.gather(l_mat, tf.cast(
                        z2prime, tf.int32), name='lprime_2')
                    Lprime_2 = tf.reduce_sum(
                        lprime_2, axis=1, keepdims=True, name='Lprime_2')
                    lprime_3 = tf.gather(l_mat, tf.cast(
                        z3prime, tf.int32), name='lprime_3')
                    Lprime_3 = tf.reduce_sum(
                        lprime_3, axis=1, keepdims=True, name='Lprime_3')
                    lprime_4 = tf.gather(l_mat, tf.cast(
                        z4prime, tf.int32), name='lprime_4')
                    Lprime_4 = tf.reduce_sum(
                        lprime_4, axis=1, keepdims=True, name='Lprime_4')

                with tf.name_scope('tomorrows_prices'):
                    # prices tomorrow
                    with tf.name_scope('R_tomorrow'):
                        Rprime_1 = alpha * tfpprime_1 * \
                            Kprime**(alpha - 1) * Lprime_1**(1 -
                                                             alpha) + (1 - deprprime_1)
                        Rprime_2 = alpha * tfpprime_2 * \
                            Kprime**(alpha - 1) * Lprime_2**(1 -
                                                             alpha) + (1 - deprprime_2)
                        Rprime_3 = alpha * tfpprime_3 * \
                            Kprime**(alpha - 1) * Lprime_3**(1 -
                                                             alpha) + (1 - deprprime_3)
                        Rprime_4 = alpha * tfpprime_4 * \
                            Kprime**(alpha - 1) * Lprime_4**(1 -
                                                             alpha) + (1 - deprprime_4)

                        # prepare prices
                        Rprime_1_mat = Rprime_1 * tf.ones([1, A])
                        Rprime_2_mat = Rprime_2 * tf.ones([1, A])
                        Rprime_3_mat = Rprime_3 * tf.ones([1, A])
                        Rprime_4_mat = Rprime_4 * tf.ones([1, A])

                    with tf.name_scope('w_tomorrow'):
                        wprime_1 = (1 - alpha) * tfpprime_1 * \
                            Kprime**alpha * Lprime_1**(-alpha)
                        wprime_2 = (1 - alpha) * tfpprime_2 * \
                            Kprime**alpha * Lprime_2**(-alpha)
                        wprime_3 = (1 - alpha) * tfpprime_3 * \
                            Kprime**alpha * Lprime_3**(-alpha)
                        wprime_4 = (1 - alpha) * tfpprime_4 * \
                            Kprime**alpha * Lprime_4**(-alpha)

                        wprime_1_mat = wprime_1 * tf.ones([1, A])
                        wprime_2_mat = wprime_2 * tf.ones([1, A])
                        wprime_3_mat = wprime_3 * tf.ones([1, A])
                        wprime_4_mat = wprime_4 * tf.ones([1, A])

                with tf.name_scope('tomorrows_production'):
                    Yprime_1 = tfpprime_1 * Kprime ** alpha * \
                        Lprime_1 ** (1 - alpha) + (1 - deprprime_1) * Kprime
                    Yprime_2 = tfpprime_2 * Kprime ** alpha * \
                        Lprime_2 ** (1 - alpha) + (1 - deprprime_2) * Kprime
                    Yprime_3 = tfpprime_3 * Kprime ** alpha * \
                        Lprime_3 ** (1 - alpha) + (1 - deprprime_3) * Kprime
                    Yprime_4 = tfpprime_4 * Kprime ** alpha * \
                        Lprime_4 ** (1 - alpha) + (1 - deprprime_4) * Kprime

                with tf.name_scope('tomorrows_fin_wealth'):
                    # individual financial wealth capital
                    f_wkprime_1 = kprime_wakeup_all * Rprime_1_mat
                    f_wkprime_2 = kprime_wakeup_all * Rprime_2_mat
                    f_wkprime_3 = kprime_wakeup_all * Rprime_3_mat
                    f_wkprime_4 = kprime_wakeup_all * Rprime_4_mat

                    # individual financial wealth bond
                    bond_payprime_1 = tf.minimum(
                        kappa * Rprime_1_mat, tf.ones_like(Rprime_1_mat))
                    bond_payprime_2 = tf.minimum(
                        kappa * Rprime_2_mat, tf.ones_like(Rprime_2_mat))
                    bond_payprime_3 = tf.minimum(
                        kappa * Rprime_3_mat, tf.ones_like(Rprime_3_mat))
                    bond_payprime_4 = tf.minimum(
                        kappa * Rprime_4_mat, tf.ones_like(Rprime_4_mat))

                    f_wbprime_1 = bprime_wakeup_all * bond_payprime_1
                    f_wbprime_2 = bprime_wakeup_all * bond_payprime_2
                    f_wbprime_3 = bprime_wakeup_all * bond_payprime_3
                    f_wbprime_4 = bprime_wakeup_all * bond_payprime_4

                    # individual total financial wealth
                    f_wprime_1 = f_wkprime_1 + f_wbprime_1
                    f_wprime_2 = f_wkprime_2 + f_wbprime_2
                    f_wprime_3 = f_wkprime_3 + f_wbprime_3
                    f_wprime_4 = f_wkprime_4 + f_wbprime_4

                with tf.name_scope('tomorrows_labor_wealth'):
                    # individual labor wealth
                    l_wprime_1 = lprime_1 * wprime_1_mat
                    l_wprime_2 = lprime_2 * wprime_2_mat
                    l_wprime_3 = lprime_3 * wprime_3_mat
                    l_wprime_4 = lprime_4 * wprime_4_mat

                with tf.name_scope('tomorrows_total_wealth'):
                    # individuals total wealth
                    incprime_1 = f_wprime_1 + l_wprime_1
                    incprime_2 = f_wprime_2 + l_wprime_2
                    incprime_3 = f_wprime_3 + l_wprime_3
                    incprime_4 = f_wprime_4 + l_wprime_4

                with tf.name_scope('tomorrows_transition_probabilities'):
                    pi_transprime_1 = tf.gather(pi, tf.cast(z1prime, tf.int32))
                    pi_trans_to1prime_1 = tf.expand_dims(
                        pi_transprime_1[:, 0], -1)
                    pi_trans_to2prime_1 = tf.expand_dims(
                        pi_transprime_1[:, 1], -1)
                    pi_trans_to3prime_1 = tf.expand_dims(
                        pi_transprime_1[:, 2], -1)
                    pi_trans_to4prime_1 = tf.expand_dims(
                        pi_transprime_1[:, 3], -1)

                    pi_transprime_2 = tf.gather(pi, tf.cast(z2prime, tf.int32))
                    pi_trans_to1prime_2 = tf.expand_dims(
                        pi_transprime_2[:, 0], -1)
                    pi_trans_to2prime_2 = tf.expand_dims(
                        pi_transprime_2[:, 1], -1)
                    pi_trans_to3prime_2 = tf.expand_dims(
                        pi_transprime_2[:, 2], -1)
                    pi_trans_to4prime_2 = tf.expand_dims(
                        pi_transprime_2[:, 3], -1)

                    pi_transprime_3 = tf.gather(pi, tf.cast(z3prime, tf.int32))
                    pi_trans_to1prime_3 = tf.expand_dims(
                        pi_transprime_3[:, 0], -1)
                    pi_trans_to2prime_3 = tf.expand_dims(
                        pi_transprime_3[:, 1], -1)
                    pi_trans_to3prime_3 = tf.expand_dims(
                        pi_transprime_3[:, 2], -1)
                    pi_trans_to4prime_3 = tf.expand_dims(
                        pi_transprime_3[:, 3], -1)

                    pi_transprime_4 = tf.gather(pi, tf.cast(z4prime, tf.int32))
                    pi_trans_to1prime_4 = tf.expand_dims(
                        pi_transprime_4[:, 0], -1)
                    pi_trans_to2prime_4 = tf.expand_dims(
                        pi_transprime_4[:, 1], -1)
                    pi_trans_to3prime_4 = tf.expand_dims(
                        pi_transprime_4[:, 2], -1)
                    pi_trans_to4prime_4 = tf.expand_dims(
                        pi_transprime_4[:, 3], -1)

                with tf.name_scope('concatenate_for_tomorrows_state'):
                    xprime_1 = tf.concat([tf.expand_dims(z1prime, -1),
                                          tfpprime_1,
                                          deprprime_1,
                                          Kprime,
                                          Lprime_1,
                                          Rprime_1,
                                          wprime_1,
                                          Yprime_1,
                                          f_wprime_1,
                                          l_wprime_1,
                                          incprime_1,
                                          kprime_wakeup_all,
                                          pi_trans_to1prime_1,
                                          pi_trans_to2prime_1,
                                          pi_trans_to3prime_1,
                                          pi_trans_to4prime_1],
                                         axis=1, name='state_tomorrow_1')

                    xprime_2 = tf.concat([tf.expand_dims(z2prime, -1),
                                          tfpprime_2,
                                          deprprime_2,
                                          Kprime,
                                          Lprime_2,
                                          Rprime_2,
                                          wprime_2,
                                          Yprime_2,
                                          f_wprime_2,
                                          l_wprime_2,
                                          incprime_2,
                                          kprime_wakeup_all,
                                          pi_trans_to1prime_2,
                                          pi_trans_to2prime_2,
                                          pi_trans_to3prime_2,
                                          pi_trans_to4prime_2],
                                         axis=1, name='state_tomorrow_2')

                    xprime_3 = tf.concat([tf.expand_dims(z3prime, -1),
                                          tfpprime_3,
                                          deprprime_3,
                                          Kprime,
                                          Lprime_3,
                                          Rprime_3,
                                          wprime_3,
                                          Yprime_3,
                                          f_wprime_3,
                                          l_wprime_3,
                                          incprime_3,
                                          kprime_wakeup_all,
                                          pi_trans_to1prime_3,
                                          pi_trans_to2prime_3,
                                          pi_trans_to3prime_3,
                                          pi_trans_to4prime_3],
                                         axis=1, name='state_tomorrow_3')

                    xprime_4 = tf.concat([tf.expand_dims(z4prime, -1),
                                          tfpprime_4,
                                          deprprime_4,
                                          Kprime,
                                          Lprime_4,
                                          Rprime_4,
                                          wprime_4,
                                          Yprime_4,
                                          f_wprime_4,
                                          l_wprime_4,
                                          incprime_4,
                                          kprime_wakeup_all,
                                          pi_trans_to1prime_4,
                                          pi_trans_to2prime_4,
                                          pi_trans_to3prime_4,
                                          pi_trans_to4prime_4],
                                         axis=1, name='state_tomorrow_4')

        with tf.name_scope('get_tomorrows_consumption'):
            with tf.name_scope('get_tomorrows_saving'):
                with tf.name_scope('NN'):
                    predprimeprime_1 = nn.predict(xprime_1)
                    kprimeprime_1 = predprimeprime_1[:, 0:A-1]
                    # lambdprime_1 = predprimeprime_1[:, A-1:2*(A-1)]
                    coll_req_primeprime_1 = predprimeprime_1[:,
                                                             2*(A-1):3*(A-1)]
                    bprimeprime_1 = (coll_req_primeprime_1 -
                                     kprimeprime_1) / kappa
                    # muprime_1 = predprimeprime_1[:, 3*(A-1): 4*(A-1)]
                    pprime_1 = tf.reshape(predprimeprime_1[:, 4*(A-1)], [m, 1])
                    pprime_1_mat = tf.tile(pprime_1, [1, A])

                    predprimeprime_2 = nn.predict(xprime_2)
                    kprimeprime_2 = predprimeprime_2[:, 0:A-1]
                    # lambdprime_2 = predprimeprime_2[:, A-1:2*(A-1)]
                    coll_req_primeprime_2 = predprimeprime_2[:,
                                                             2*(A-1):3*(A-1)]
                    bprimeprime_2 = (coll_req_primeprime_2 -
                                     kprimeprime_2) / kappa
                    # muprime_2 = predprimeprime_2[:, 3*(A-1): 4*(A-1)]
                    pprime_2 = tf.reshape(predprimeprime_2[:, 4*(A-1)], [m, 1])
                    pprime_2_mat = tf.tile(pprime_2, [1, A])

                    predprimeprime_3 = nn.predict(xprime_3)
                    kprimeprime_3 = predprimeprime_3[:, 0:A-1]
                    # lambdprime_3 = predprimeprime_3[:, A-1:2*(A-1)]
                    coll_req_primeprime_3 = predprimeprime_3[:,
                                                             2*(A-1):3*(A-1)]
                    bprimeprime_3 = (coll_req_primeprime_3 -
                                     kprimeprime_3) / kappa
                    # muprime_3 = predprimeprime_3[:, 3*(A-1): 4*(A-1)]
                    pprime_3 = tf.reshape(predprimeprime_3[:, 4*(A-1)], [m, 1])
                    pprime_3_mat = tf.tile(pprime_3, [1, A])

                    predprimeprime_4 = nn.predict(xprime_4)
                    kprimeprime_4 = predprimeprime_4[:, 0:A-1]
                    # lambdprime_4 = predprimeprime_4[:, A-1:2*(A-1)]
                    coll_req_primeprime_4 = predprimeprime_4[:,
                                                             2*(A-1):3*(A-1)]
                    bprimeprime_4 = (coll_req_primeprime_4 -
                                     kprimeprime_4) / kappa
                    # muprime_4 = predprimeprime_4[:, 3*(A-1): 4*(A-1)]
                    pprime_4 = tf.reshape(predprimeprime_4[:, 4*(A-1)], [m, 1])
                    pprime_4_mat = tf.tile(pprime_4, [1, A])

            with tf.name_scope('kprimeprime_save_all'):
                kprimeprime_save_all_1 = tf.concat(
                    [kprimeprime_1, tf.zeros([m, 1])], axis=1)
                kprimeprime_save_all_2 = tf.concat(
                    [kprimeprime_2, tf.zeros([m, 1])], axis=1)
                kprimeprime_save_all_3 = tf.concat(
                    [kprimeprime_3, tf.zeros([m, 1])], axis=1)
                kprimeprime_save_all_4 = tf.concat(
                    [kprimeprime_4, tf.zeros([m, 1])], axis=1)

                bond_spent_allprime_1 = tf.multiply(
                    tf.concat([bprimeprime_1, tf.zeros([m, 1])], axis=1), pprime_1_mat)
                bond_spent_allprime_2 = tf.multiply(
                    tf.concat([bprimeprime_2, tf.zeros([m, 1])], axis=1), pprime_2_mat)
                bond_spent_allprime_3 = tf.multiply(
                    tf.concat([bprimeprime_3, tf.zeros([m, 1])], axis=1), pprime_3_mat)
                bond_spent_allprime_4 = tf.multiply(
                    tf.concat([bprimeprime_4, tf.zeros([m, 1])], axis=1), pprime_4_mat)

                tot_save_prime_1 = kprimeprime_save_all_1 + bond_spent_allprime_1
                tot_save_prime_2 = kprimeprime_save_all_2 + bond_spent_allprime_2
                tot_save_prime_3 = kprimeprime_save_all_3 + bond_spent_allprime_3
                tot_save_prime_4 = kprimeprime_save_all_4 + bond_spent_allprime_4

            with tf.name_scope('adjustment_cost_prime_all'):
                adjustment_all_prime_1 = (
                    kprimeprime_save_all_1 - tf.reshape(Rprime_1, [m, 1]) * kprime_wakeup_all)
                adjustment_all_prime_2 = (
                    kprimeprime_save_all_2 - tf.reshape(Rprime_2, [m, 1]) * kprime_wakeup_all)
                adjustment_all_prime_3 = (
                    kprimeprime_save_all_3 - tf.reshape(Rprime_3, [m, 1]) * kprime_wakeup_all)
                adjustment_all_prime_4 = (
                    kprimeprime_save_all_4 - tf.reshape(Rprime_4, [m, 1]) * kprime_wakeup_all)

                adj_cost_all_prime_1 = (zeta / 2.0) * adjustment_all_prime_1**2
                adj_cost_all_prime_2 = (zeta / 2.0) * adjustment_all_prime_2**2
                adj_cost_all_prime_3 = (zeta / 2.0) * adjustment_all_prime_3**2
                adj_cost_all_prime_4 = (zeta / 2.0) * adjustment_all_prime_4**2

            with tf.name_scope('compute_tomorrows_consumption'):
                c_all_origprime_1 = incprime_1 - tot_save_prime_1 - adj_cost_all_prime_1
                c_allprime_1 = tf.maximum(c_all_origprime_1, tf.ones_like(
                    c_all_origprime_1) * eps, name='c_all_todayprime_1')

                c_all_origprime_2 = incprime_2 - tot_save_prime_2 - adj_cost_all_prime_2
                c_allprime_2 = tf.maximum(c_all_origprime_2, tf.ones_like(
                    c_all_origprime_2) * eps, name='c_all_todayprime_2')

                c_all_origprime_3 = incprime_3 - tot_save_prime_3 - adj_cost_all_prime_3
                c_allprime_3 = tf.maximum(c_all_origprime_3, tf.ones_like(
                    c_all_origprime_3) * eps, name='c_all_todayprime_3')

                c_all_origprime_4 = incprime_4 - tot_save_prime_4 - adj_cost_all_prime_4
                c_allprime_4 = tf.maximum(c_all_origprime_4, tf.ones_like(
                    c_all_origprime_4) * eps, name='c_all_todayprime_4')

            with tf.name_scope('optimality_conditions'):
                # optimality conditions
                with tf.name_scope('rel_ee'):
                    # prepare transitions
                    pi_trans_to1 = tf.expand_dims(
                        probs_next[:, 0], -1) * tf.ones([1, A-1])
                    pi_trans_to2 = tf.expand_dims(
                        probs_next[:, 1], -1) * tf.ones([1, A-1])
                    pi_trans_to3 = tf.expand_dims(
                        probs_next[:, 2], -1) * tf.ones([1, A-1])
                    pi_trans_to4 = tf.expand_dims(
                        probs_next[:, 3], -1) * tf.ones([1, A-1])

                    # euler equation
                    opt_euler_cap = - 1 \
                        + (((
                            (beta *
                             (pi_trans_to1 * (Rprime_1_mat[:, 0:A-1] * (1. + zeta * adjustment_all_prime_1[:, 1:A])) * c_allprime_1[:, 1:A]**(- gamma)
                              + pi_trans_to2 * (Rprime_2_mat[:, 0:A-1] * (
                                  1. + zeta * adjustment_all_prime_2[:, 1:A])) * c_allprime_2[:, 1:A]**(- gamma)
                                 + pi_trans_to3 * (Rprime_3_mat[:, 0:A-1] * (
                                     1. + zeta * adjustment_all_prime_3[:, 1:A])) * c_allprime_3[:, 1:A]**(- gamma)
                                 + pi_trans_to4 * (Rprime_4_mat[:, 0:A-1] * (1. + zeta * adjustment_all_prime_4[:, 1:A])) * c_allprime_4[:, 1:A]**(- gamma))
                                + lambd + mu) / (1.0 + zeta * adjustment_all[:, 0:A-1])
                        ) ** (-1.0 / gamma))
                            / c_all[:, 0:A-1])

                    opt_euler_bond = - 1 \
                        + ((((beta *
                              (pi_trans_to1 * bond_payprime_1[:, 0:A-1] * c_allprime_1[:, 1:A]**(- gamma)
                               + pi_trans_to2 *
                               bond_payprime_2[:, 0:A-1] *
                               c_allprime_2[:, 1:A]**(- gamma)
                               + pi_trans_to3 *
                               bond_payprime_3[:, 0:A-1] *
                               c_allprime_3[:, 1:A]**(- gamma)
                               + pi_trans_to4 * bond_payprime_4[:, 0:A-1] * c_allprime_4[:, 1:A]**(- gamma))
                            + mu * kappa) / p_mat[:, 0:A-1]) ** (-1.0 / gamma))
                            / c_all[:, 0:A-1])

                    opt_euler = tf.concat(
                        [opt_euler_cap, opt_euler_bond], axis=1)

                    # KKT condition
                    # The condition that kprime >= 0 and lambd >= 0 are enforced by softplus activation in the output layer
                    opt_KKT_cap = tf.multiply(kprime, lambd)
                    opt_KKT_bond = tf.multiply(coll_req_prime, mu)

                    opt_KKT = tf.concat([opt_KKT_cap, opt_KKT_bond], axis=1)

                with tf.name_scope('punishments'):
                    # punishment for negative cons
                    orig_cons = tf.concat(
                        [c_all_orig, c_all_origprime_1, c_all_origprime_2, c_all_origprime_3, c_all_origprime_4], axis=1)

                    opt_punish_cons = (1.0/eps) * tf.maximum(-1 * orig_cons,
                                                             tf.zeros_like(orig_cons), name='opt_punish_cons')

                    # Punishment for negative aggregate capital
                    opt_punish_ktotprime = (
                        1.0/eps) * tf.maximum(-Kprime_orig, tf.zeros_like(Kprime_orig), name='opt_punish_aggr_saving')

                    # Punishment for excessdemand
                    tot_bond_dem = tf.reduce_sum(bprime, axis=1, keepdims=True)
                    tot_bond_dem_prime_1 = tf.reduce_sum(
                        bprimeprime_1, axis=1, keepdims=True)
                    tot_bond_dem_prime_2 = tf.reduce_sum(
                        bprimeprime_2, axis=1, keepdims=True)
                    tot_bond_dem_prime_3 = tf.reduce_sum(
                        bprimeprime_3, axis=1, keepdims=True)
                    tot_bond_dem_prime_4 = tf.reduce_sum(
                        bprimeprime_4, axis=1, keepdims=True)
                    opt_punish_bond_clear = tf.concat([tot_bond_dem, tot_bond_dem_prime_1, tot_bond_dem_prime_2,
                                                       tot_bond_dem_prime_3, tot_bond_dem_prime_4], axis=1)

                # Put together
                combined_opt = [opt_euler, opt_punish_cons,
                                opt_punish_ktotprime, opt_KKT, opt_punish_bond_clear]
                opt_predict = tf.concat(
                    combined_opt, axis=1, name='combined_opt_cond')

                with tf.name_scope('compute_cost'):
                    # define the correct output
                    opt_correct = tf.zeros_like(opt_predict, name='target')

                    # define the cost function
                    cost = tf.losses.mean_squared_error(
                        opt_correct, opt_predict)

    with tf.name_scope('train_setup'):
        if optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='adam')
        else:
            raise NotImplementedError

        with tf.name_scope('gradients'):
            gvs = optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var)
                          for grad, var in gvs]
            train_step = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('simulate_episode'):
        def simulate_episodes(sess, x_start, episode_length=10**4, print_flag=True):
            sim_start_time = datetime.now()
            if print_flag:
                print('Start simulating {} periods.'.format(episode_length))

            num_state_var = np.shape(x_start)[1]

            X_episodes = np.zeros([episode_length, num_state_var])
            X_episodes[0, :] = x_start
            X_old = x_start
            rand_num = np.random.rand(episode_length, 1)

            for t in range(1, episode_length):
                if rand_num[t - 1] <= PI[int(X_old[0, 0]), 0]:
                    X_new = sess.run(xprime_1, feed_dict={X: X_old})
                elif rand_num[t - 1] <= PI[int(X_old[0, 0]), 0] + PI[int(X_old[0, 0]), 1]:
                    X_new = sess.run(xprime_2, feed_dict={X: X_old})
                elif rand_num[t - 1] <= PI[int(X_old[0, 0]), 0] + PI[int(X_old[0, 0]), 1] + PI[int(X_old[0, 0]), 2]:
                    X_new = sess.run(xprime_3, feed_dict={X: X_old})
                else:
                    X_new = sess.run(xprime_4, feed_dict={X: X_old})

                X_episodes[t, :] = X_new
                X_old = X_new.copy()

            sim_end_time = datetime.now()
            sim_duration = sim_end_time - sim_start_time
            if print_flag:
                print('Finished simulation. Time for simulation: {}.'.format(
                    sim_duration))

            return X_episodes

        def simulate_batch_episodes(sess, x_start, episode_length=10**4, print_flag=True):
            sim_start_time = datetime.now()
            num_state_var = np.shape(x_start)[1]
            batch_size = np.shape(x_start)[0]

            if print_flag:
                print('Start simulating {} tracks with {} periods.'.format(
                    batch_size, episode_length))

            X_episodes = np.zeros([batch_size * episode_length, num_state_var])
            X_old = x_start
            rand_num = np.random.rand(batch_size, episode_length)

            for t in range(0, episode_length):
                temp_rand = rand_num[:, t]
                X_new = np.zeros((batch_size, num_state_var))
                trans_probs_to1 = X_old[:, 8 + 4 * A]
                trans_probs_to2 = X_old[:, 8 + 4 * A+1]
                trans_probs_to3 = X_old[:, 8 + 4 * A+2]
                #trans_probs_to4 = X_old[:, 8 + 4 * A+3]

                to_1 = temp_rand <= trans_probs_to1
                to_2 = np.logical_and(
                    temp_rand > trans_probs_to1, temp_rand <= trans_probs_to1 + trans_probs_to2)
                to_3 = np.logical_and(temp_rand > trans_probs_to1 + trans_probs_to2,
                                      temp_rand <= trans_probs_to1 + trans_probs_to2 + trans_probs_to3)
                to_4 = temp_rand > trans_probs_to1 + trans_probs_to2 + trans_probs_to3

                X_new[to_1, :] = sess.run(
                    xprime_1, feed_dict={X: X_old[to_1, :]})
                X_new[to_2, :] = sess.run(
                    xprime_2, feed_dict={X: X_old[to_2, :]})
                X_new[to_3, :] = sess.run(
                    xprime_3, feed_dict={X: X_old[to_3, :]})
                X_new[to_4, :] = sess.run(
                    xprime_4, feed_dict={X: X_old[to_4, :]})

                X_episodes[t * batch_size: (t+1) * batch_size, :] = X_new
                X_old = X_new.copy()

            sim_end_time = datetime.now()
            sim_duration = sim_end_time - sim_start_time
            if print_flag:
                print('Finished simulation. Time for simulation: {}.'.format(
                    sim_duration))

            return X_episodes

    sess = tf.Session()

    with tf.name_scope('get_starting_point'):
        if not(load_flag):
            X_data_train = np.random.rand(1, n_input)
            X_data_train[:, 0] = (X_data_train[:, 0] > 0.5)
            X_data_train[:, 1:] = X_data_train[:, 1:] + 0.1
            assert np.min(np.sum(X_data_train, axis=1, keepdims=True) >
                          0) == True, 'starting point has negative aggregate capital'
            print('Calculated a valid starting point')

        else:
            load_base_path = os.path.join('./output',  load_run_name)
            load_params_nm = load_run_name + '-episode' + str(load_episode)
            load_params_path = os.path.join(
                load_base_path, 'model', load_params_nm)
            load_data_path = os.path.join(
                load_base_path,  'model', load_params_nm + '_LastData.npy')
            X_data_train = np.load(load_data_path)

            print('Loaded initial data from ' + load_data_path)

    with tf.name_scope('training'):
        minibatch_size = int(batch_size)
        num_minibatches = int(len_episodes / minibatch_size)
        train_seed = 0

        cost_store = np.zeros(num_episodes)
        mov_ave_cost_store = np.zeros(num_episodes)
        mov_ave_len = 100

        time_store = np.zeros(num_episodes)
        ee_store = np.zeros((num_episodes, 2*(num_agents - 1)))
        max_ee_store = np.zeros((num_episodes, 2*(num_agents - 1)))

        start_time = datetime.now()
        print('start time: {}'.format(start_time))

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        init = tf.global_variables_initializer()

        print_interval = 50

        sim_batch_size = 1000
        sim_len = int(len_episodes/sim_batch_size)

        # run the initializer
        sess.run(init)

        if load_flag:
            saver = tf.train.Saver(nn.param_dict)
            saver.restore(sess, load_params_path)
            print('Weights loaded from: ' + load_params_path)

        for ep in range(load_episode, num_episodes + load_episode):
            if ep == load_episode:
                if ep <= 2:
                    X_data_train = np.matlib.repmat(
                        X_data_train, sim_batch_size, 1)

            print_flag = (ep % print_interval == 0) or ep == load_episode

            if print_flag:
                print('Episode {}'.format(ep))
            start_time_learn = datetime.now()

            X_episodes = simulate_batch_episodes(
                sess, X_data_train, episode_length=sim_len, print_flag=print_flag)
            X_data_train = X_episodes[len_episodes -
                                      sim_batch_size: len_episodes, :]

            if print_flag:
                print('Starting learning on episode')

            for epoch in range(epochs_per_episode):
                if print_flag:
                    print('Epoch {} on this episode.'.format(epoch))
                train_seed = train_seed + 1

                minibatches = random_mini_batches(
                    X_episodes, minibatch_size, train_seed)
                minibatch_cost = 0

                if epoch == 0:
                    ee_error = np.zeros((1, 2 * (num_agents - 1)))
                    max_ee = np.zeros((1, 2 * (num_agents - 1)))

                for minibatch in minibatches:
                    (minibatch_X) = minibatch

                    # Run optimization
                    minibatch_cost += sess.run(cost,
                                               feed_dict={X: minibatch_X}) / num_minibatches
                    if epoch == 0:
                        ee_error += np.mean(np.abs(sess.run(opt_euler, feed_dict={
                                            X: minibatch_X})), axis=0) / num_minibatches
                        temp_max_ee = np.max(
                            np.abs(sess.run(opt_euler, feed_dict={X: minibatch_X})), axis=0, keepdims=True)
                        max_ee = np.maximum(max_ee, temp_max_ee)

                if epoch == 0:
                    cost_store[ep-load_episode] = minibatch_cost

                if print_flag:
                    print('Epoch {}, log10(Cost)= {:.4f}'.format(
                        epoch, np.log10(minibatch_cost)))

                if train_flag:
                    for minibatch in minibatches:
                        (minibatch_X) = minibatch

                        # Run train step
                        sess.run(train_step, feed_dict={X: minibatch_X})

            end_time_learn = datetime.now()
            if print_flag:
                print('Finished learning on episode. Time for learning: {}.'.format(
                    end_time_learn - start_time_learn))

            if ep-load_episode > mov_ave_len + 10:
                mov_ave_cost_store[ep-load_episode] = np.mean(
                    cost_store[ep-load_episode-mov_ave_len:ep-load_episode])
            else:
                mov_ave_cost_store[ep -
                                   load_episode] = np.mean(cost_store[0:ep-load_episode])

            ee_store[ep-load_episode, :] = ee_error
            max_ee_store[ep-load_episode, :] = max_ee
            cur_time = datetime.now() - start_time
            time_store[ep-load_episode] = cur_time.seconds

            # Calculate cost
            print('\nEpisode {}, log10(Cost)= {:.4f}'.format(
                ep, np.log10(cost_store[ep-load_episode])))
            print('Time: {}; time since start: {}'.format(
                datetime.now(), datetime.now() - start_time))

            if ep % save_interval == 0 or ep == 1:
                plot_dict = {}
                plot_epi_length = 2000

                # simulate new episodes to plot
                X_data_train_plot = X_episodes[-1, :].reshape([1, -1])
                X_episodes = simulate_episodes(
                    sess, X_data_train_plot, episode_length=plot_epi_length, print_flag=print_flag)
                plot_period = np.arange(1, plot_epi_length+1)
                len_plot_episodes = plot_epi_length

                plot_age_all = np.arange(25, 25+A)
                plot_age_exceptlast = np.arange(25, 25+A - 1)

                plt.rc('font', family='serif')
                plt.rc('xtick', labelsize='small')
                plt.rc('ytick', labelsize='small')

                std_figsize = (4, 4)
                percentiles_dict = {50: {'ls': ':', 'label': '50'}, 10: {'ls': '-.', 'label': '10'}, 90: {
                    'ls': '-.', 'label': '90'}, 99.9: {'ls': '--', 'label': '99.9'}, 0.1: {'ls': '--', 'label': '0.1'}}

                shock1_dict = {'label': 'shock 1', 'color': 'r'}
                shock2_dict = {'label': 'shock 2', 'color': 'b'}
                shock3_dict = {'label': 'shock 3', 'color': 'y'}
                shock4_dict = {'label': 'shock 4', 'color': 'g'}
                shock_dict = {1: shock1_dict, 2: shock2_dict,
                              3: shock3_dict, 4: shock4_dict}

                if ep < load_episode + save_interval + 1:

                    fig = plt.figure(figsize=std_figsize)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(
                        plot_age_all, LABOR_ENDOW[1, :], 'k-', label='labor endowment over life')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Labor endowment')
                    plt.savefig(plot_dir + '/' + run_name +
                                '_laborendowment_' + str(ep)+'.pdf', bbox_inches='tight')
                    plt.close()

                    labor_endowment_dict = {
                        'x': plot_age_all.tolist(), 'y': LABOR_ENDOW[1, :].tolist()}
                    plot_dict['laborendowment'] = labor_endowment_dict

                # run stuff
                c_all_orig_ = sess.run(c_all_orig, feed_dict={X: X_episodes})
                c_all_origprime_1_ = sess.run(
                    c_all_origprime_1, feed_dict={X: X_episodes})
                c_all_origprime_2_ = sess.run(
                    c_all_origprime_2, feed_dict={X: X_episodes})
                c_all_origprime_3_ = sess.run(
                    c_all_origprime_3, feed_dict={X: X_episodes})
                c_all_origprime_4_ = sess.run(
                    c_all_origprime_4, feed_dict={X: X_episodes})

                bond_price_ = sess.run(p, feed_dict={X: X_episodes})
                bprime_ = sess.run(bprime, feed_dict={X: X_episodes})

                bond_spent_all_ = sess.run(
                    bond_spent_all, feed_dict={X: X_episodes})
                k_saved_all_ = sess.run(k_saved_all, feed_dict={X: X_episodes})
                tot_saved_all_ = sess.run(
                    tot_saved_all, feed_dict={X: X_episodes})
                inc_ = sess.run(inc, feed_dict={X: X_episodes})
                l_w_ = sess.run(l_w, feed_dict={X: X_episodes})
                fin_w_ = inc_ - l_w_
                adj_cost_all_ = sess.run(
                    adj_cost_all, feed_dict={X: X_episodes})

                lambd_ = sess.run(lambd, feed_dict={X: X_episodes})
                mu_ = sess.run(mu, feed_dict={X: X_episodes})
                opt_euler_cap_ = sess.run(
                    opt_euler_cap, feed_dict={X: X_episodes})
                opt_euler_bond_ = sess.run(
                    opt_euler_bond, feed_dict={X: X_episodes})
                opt_KKT_cap_ = sess.run(opt_KKT_cap, feed_dict={X: X_episodes})
                opt_KKT_bond_ = sess.run(
                    opt_KKT_bond, feed_dict={X: X_episodes})
                opt_euler_ = sess.run(opt_euler, feed_dict={X: X_episodes})

                tot_bond_dem_ = sess.run(
                    tot_bond_dem, feed_dict={X: X_episodes})

                a_cond1 = (X_episodes[:plot_epi_length, 0] == 0)
                a_cond2 = (X_episodes[:plot_epi_length, 0] == 1)
                a_cond3 = (X_episodes[:plot_epi_length, 0] == 2)
                a_cond4 = (X_episodes[:plot_epi_length, 0] == 3)

                if print_frac_constrained:

                    # to construct fraction of households with binding collateral constraint
                    coll_req_prime_ = sess.run(
                        coll_req_prime, feed_dict={X: X_episodes})

                    # mean value of kappa * k + b
                    meancollreqprime = np.mean(coll_req_prime_)

                    # construct fraction constrained households
                    # compare: (kappa * b + k) / c to (kappa * mu) / (u'(c))

                    frac_constr_bond_overall = np.mean((coll_req_prime_ / c_all_orig_[:, :-1]) < (
                        (KAPPA * mu_) / c_all_orig_[:, :-1] ** (- GAMMA)), axis=0)

                    frac_constr_bond_1 = np.mean((coll_req_prime_[a_cond1] / c_all_orig_[a_cond1, :-1]) < (
                        KAPPA * mu_[a_cond1]) / c_all_orig_[a_cond1, :-1] ** (- GAMMA), axis=0)
                    frac_constr_bond_2 = np.mean((coll_req_prime_[a_cond2] / c_all_orig_[a_cond2, :-1]) < (
                        KAPPA * mu_[a_cond2]) / c_all_orig_[a_cond2, :-1] ** (- GAMMA), axis=0)
                    frac_constr_bond_3 = np.mean((coll_req_prime_[a_cond3] / c_all_orig_[a_cond3, :-1]) < (
                        KAPPA * mu_[a_cond3]) / c_all_orig_[a_cond3, :-1] ** (- GAMMA), axis=0)
                    frac_constr_bond_4 = np.mean((coll_req_prime_[a_cond4] / c_all_orig_[a_cond4, :-1]) < (
                        KAPPA * mu_[a_cond4]) / c_all_orig_[a_cond4, :-1] ** (- GAMMA), axis=0)

                    print('#' + '=' * 50)
                    print('#' + '=' * 50)
                    print('fraction collateral constrained by age:')
                    print(frac_constr_bond_overall)
                    print('fraction collateral constrained by shock and age:')
                    print('shock 1:')
                    print(frac_constr_bond_1)
                    print('shock 2:')
                    print(frac_constr_bond_2)
                    print('shock 3:')
                    print(frac_constr_bond_3)
                    print('shock 4:')
                    print(frac_constr_bond_4)
                    print('#' + '=' * 50)

                    for ageidx in range(plot_age_exceptlast.shape[0]):
                        print('age = ', plot_age_exceptlast[ageidx])
                        print('fraction collateral constrained:')
                        print(frac_constr_bond_overall[ageidx])
                        print('fraction collateral constrained by shock and age:')
                        print('shock 1:')
                        print(frac_constr_bond_1[ageidx])
                        print('shock 2:')
                        print(frac_constr_bond_2[ageidx])
                        print('shock 3:')
                        print(frac_constr_bond_3[ageidx])
                        print('shock 4:')
                        print(frac_constr_bond_4[ageidx])
                        print('#' + '=' * 50)

                ### plots needed for paper ###
                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                lines, labs1, labs2 = [], ['mean'], ['capital', 'bond']
                l1, = plt.plot(plot_age_exceptlast, np.mean(
                    lambd_, axis=0), 'k-', label='capital, mean')
                l2, = plt.plot(plot_age_exceptlast, np.mean(
                    mu_, axis=0), 'r-', label='bond, mean')
                lines.append([l1, l2])
                for perc_key in percentiles_dict:
                    l1, = plt.plot(plot_age_exceptlast, np.percentile(lambd_, perc_key, axis=0), 'k' +
                                   percentiles_dict[perc_key]['ls'], label='capital, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l2, = plt.plot(plot_age_exceptlast, np.percentile(mu_, perc_key, axis=0), 'r' +
                                   percentiles_dict[perc_key]['ls'], label='bond, ' + percentiles_dict[perc_key]['label']+' percentile')
                    lines.append([l1, l2])
                    labs1.append(
                        percentiles_dict[perc_key]['label'] + ' percentile')
                ax.set_xlabel('Age')
                ax.set_ylabel('KKT mult')
                legend1 = plt.legend(
                    lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1,
                           bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name +
                            '_KKTmult_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                KKTmult_dict = {'x': plot_age_exceptlast.tolist(
                ), 'y1': lambd_.tolist(), 'y2': mu_.tolist()}
                plot_dict['KKTmult_dict'] = KKTmult_dict

                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(plot_age_exceptlast, np.mean(
                    bprime_, axis=0), 'k-', label='mean')
                for perc_key in percentiles_dict:
                    plt.plot(plot_age_exceptlast, np.percentile(bprime_, perc_key, axis=0), 'k' +
                             percentiles_dict[perc_key]['ls'], label=percentiles_dict[perc_key]['label']+' percentile')
                ax.set_xlabel('Age')
                ax.set_ylabel('bond bought')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name +
                            '_bondbought_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                bondbought_dict = {
                    'x': plot_age_exceptlast.tolist(), 'y': bprime_.tolist()}
                plot_dict['bondbought'] = bondbought_dict

                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                lines, labs1, labs2 = [], ['mean'], ['bond', 'capital', 'adj. cost', 'total']
                l1, = plt.plot(plot_age_all, np.mean(
                    bond_spent_all_, axis=0), 'r-')
                l2, = plt.plot(plot_age_all, np.mean(
                    k_saved_all_, axis=0), 'k-')
                l3, = plt.plot(plot_age_all, np.mean(
                    adj_cost_all_, axis=0), 'g-')
                l4, = plt.plot(plot_age_all, np.mean(
                    tot_saved_all_ + adj_cost_all_, axis=0), 'b-')
                lines.append([l1, l2, l3, l4])
                for perc_key in percentiles_dict:
                    l1, = plt.plot(plot_age_all, np.percentile(bond_spent_all_, perc_key, axis=0), 'r' +
                                   percentiles_dict[perc_key]['ls'], label='bond, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l2, = plt.plot(plot_age_all, np.percentile(k_saved_all_, perc_key, axis=0), 'k' +
                                   percentiles_dict[perc_key]['ls'], label='capital, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l3, = plt.plot(plot_age_all, np.percentile(adj_cost_all_, perc_key, axis=0), 'g' +
                                   percentiles_dict[perc_key]['ls'], label='adj. cost, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l4, = plt.plot(plot_age_all, np.percentile(tot_saved_all_ + adj_cost_all_, perc_key, axis=0), 'b' +
                                   percentiles_dict[perc_key]['ls'], label='total, ' + percentiles_dict[perc_key]['label']+' percentile')
                    lines.append([l1, l2, l3, l4])
                    labs1.append(
                        percentiles_dict[perc_key]['label'] + ' percentile')
                ax.set_xlabel('Age')
                ax.set_ylabel('cons invested')
                legend1 = plt.legend(
                    lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[1] for l in lines], labs1,
                           bbox_to_anchor=(1.05, 0.6), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name +
                            '_inv_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                inv_dict = {'x1': plot_age_all.tolist(), 'x2': plot_age_all.tolist(), 'x3': plot_age_all.tolist(),
                            'y1': bond_spent_all_.tolist(), 'y2': k_saved_all_.tolist(), 'y3': tot_saved_all_.tolist()}
                plot_dict['inv'] = inv_dict

                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(plot_age_all, np.mean(
                    bond_spent_all_, axis=0), 'k--', label='bond')
                plt.plot(plot_age_all, np.mean(
                    k_saved_all_, axis=0), 'k:', label='capital')
                plt.plot(plot_age_all, np.mean(adj_cost_all_, axis=0),
                         'k-.', label='adj. cost')
                plt.plot(plot_age_all, np.mean(tot_saved_all_ +
                         adj_cost_all_, axis=0), 'k-', label='total')

                plt.plot(plot_age_all, np.mean(bond_spent_all_[
                         a_cond1, :], axis=0), shock_dict[1]['color'] + '--')
                plt.plot(plot_age_all, np.mean(k_saved_all_[
                         a_cond1, :], axis=0), shock_dict[1]['color'] + ':')
                plt.plot(plot_age_all, np.mean(adj_cost_all_[
                         a_cond1, :], axis=0), shock_dict[1]['color'] + '-.')
                plt.plot(plot_age_all, np.mean(tot_saved_all_[
                         a_cond1, :] + adj_cost_all_[a_cond1, :], axis=0), shock_dict[1]['color'] + '-')

                plt.plot(plot_age_all, np.mean(bond_spent_all_[
                         a_cond2, :], axis=0), shock_dict[2]['color'] + '--')
                plt.plot(plot_age_all, np.mean(k_saved_all_[
                         a_cond2, :], axis=0), shock_dict[2]['color'] + ':')
                plt.plot(plot_age_all, np.mean(adj_cost_all_[
                         a_cond2, :], axis=0), shock_dict[2]['color'] + '-.')
                plt.plot(plot_age_all, np.mean(tot_saved_all_[
                         a_cond2, :] + adj_cost_all_[a_cond2, :], axis=0), shock_dict[2]['color'] + '-')

                plt.plot(plot_age_all, np.mean(bond_spent_all_[
                         a_cond3, :], axis=0), shock_dict[3]['color'] + '--')
                plt.plot(plot_age_all, np.mean(k_saved_all_[
                         a_cond3, :], axis=0), shock_dict[3]['color'] + ':')
                plt.plot(plot_age_all, np.mean(adj_cost_all_[
                         a_cond3, :], axis=0), shock_dict[3]['color'] + '-.')
                plt.plot(plot_age_all, np.mean(tot_saved_all_[
                         a_cond3, :] + adj_cost_all_[a_cond3, :], axis=0), shock_dict[3]['color'] + '-')

                plt.plot(plot_age_all, np.mean(bond_spent_all_[
                         a_cond4, :], axis=0), shock_dict[4]['color'] + '--')
                plt.plot(plot_age_all, np.mean(k_saved_all_[
                         a_cond4, :], axis=0), shock_dict[4]['color'] + ':')
                plt.plot(plot_age_all, np.mean(adj_cost_all_[
                         a_cond4, :], axis=0), shock_dict[4]['color'] + '-.')
                plt.plot(plot_age_all, np.mean(tot_saved_all_[
                         a_cond4, :] + + adj_cost_all_[a_cond4, :], axis=0), shock_dict[4]['color'] + '-')
                ax.set_xlabel('Age')
                ax.set_ylabel('cons invested')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name +
                            '_invshock_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                invshock_dict = {'x': plot_age_all.tolist(),
                                 'y1': bond_spent_all_.tolist(), 'y2': k_saved_all_.tolist(), 'y3': adj_cost_all_.tolist(), 'y4': (tot_saved_all_+adj_cost_all_).tolist(),
                                 'y5': bond_spent_all_[a_cond1, :].tolist(), 'y6': k_saved_all_[a_cond1, :].tolist(), 'y7': adj_cost_all_[a_cond1, :].tolist(), 'y8': (tot_saved_all_[a_cond1, :]+adj_cost_all_[a_cond1, :]).tolist(),
                                 'y9': bond_spent_all_[a_cond2, :].tolist(), 'y10': k_saved_all_[a_cond2, :].tolist(), 'y11': adj_cost_all_[a_cond2, :].tolist(), 'y12': (tot_saved_all_[a_cond2, :]+adj_cost_all_[a_cond2, :]).tolist(),
                                 'y13': bond_spent_all_[a_cond3, :].tolist(), 'y14': k_saved_all_[a_cond3, :].tolist(), 'y15': adj_cost_all_[a_cond3, :].tolist(), 'y16': (tot_saved_all_[a_cond3, :]+adj_cost_all_[a_cond3, :]).tolist(),
                                 'y17': bond_spent_all_[a_cond4, :].tolist(), 'y18': k_saved_all_[a_cond4, :].tolist(), 'y19': adj_cost_all_[a_cond4, :].tolist(), 'y20': tot_saved_all_[a_cond4, :].tolist()}
                plot_dict['invshock'] = invshock_dict

                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(plot_period[a_cond1], bond_price_[
                         a_cond1, 0], shock_dict[1]['color'] + '*', label=shock_dict[1]['label'])
                plt.plot(plot_period[a_cond2], bond_price_[
                         a_cond2, 0], shock_dict[2]['color'] + 'o', label=shock_dict[2]['label'])
                plt.plot(plot_period[a_cond3], bond_price_[
                         a_cond3, 0], shock_dict[3]['color'] + '*', label=shock_dict[3]['label'])
                plt.plot(plot_period[a_cond4], bond_price_[
                         a_cond4, 0], shock_dict[4]['color'] + 'o', label=shock_dict[4]['label'])
                plt.plot(np.ones_like(bond_price_) *
                         np.mean(bond_price_), 'k-')
                for perc_key in percentiles_dict:
                    price_perc = np.percentile(bond_price_, perc_key, axis=0)
                    plt.plot(np.ones_like(bond_price_) * price_perc,
                             'k' + percentiles_dict[perc_key]['ls'])
                ax.set_ylabel('bond price')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name +
                            '_bondprice_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                bondprice_dict = {'x1': plot_period[a_cond1].tolist(), 'x2': plot_period[a_cond2].tolist(), 'x3': plot_period[a_cond3].tolist(), 'x4': plot_period[a_cond4].tolist(),
                                  'y1': bond_price_[a_cond1, 0].tolist(), 'y2': bond_price_[a_cond2, 0].tolist(), 'y3': bond_price_[a_cond3, 0].tolist(), 'y4': bond_price_[a_cond4, 0].tolist()}
                plot_dict['bondprice'] = bondprice_dict

                plt.figure(figsize=std_figsize)
                ax4 = plt.subplot(1, 1, 1)
                lines, labs1, labs2 = [], ['mean'], ['capital', 'bond']
                l1, = ax4.plot(plot_age_exceptlast, np.log10(
                    np.mean(np.abs(opt_euler_cap_), axis=0)), 'k-', label='capital, mean')
                l2, = ax4.plot(plot_age_exceptlast, np.log10(
                    np.mean(np.abs(opt_euler_bond_), axis=0)), 'r-', label='bond, mean')
                lines.append([l1, l2])
                for perc_key in percentiles_dict:
                    l1, = ax4.plot(plot_age_exceptlast, np.log10(np.percentile(np.abs(opt_euler_cap_), perc_key, axis=0)), 'k' +
                                   percentiles_dict[perc_key]['ls'], label='capital, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l2, = ax4.plot(plot_age_exceptlast, np.log10(np.percentile(np.abs(opt_euler_bond_), perc_key, axis=0)),
                                   'r'+percentiles_dict[perc_key]['ls'], label='bond, ' + percentiles_dict[perc_key]['label']+' percentile')
                    lines.append([l1, l2])
                    labs1.append(
                        percentiles_dict[perc_key]['label'] + ' percentile')
                ax4.set_xlabel('Age')
                ax4.set_ylabel('rel Ee error [log10]')
                legend1 = plt.legend(
                    lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1,
                           bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name +
                            '_relee_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                relee_dict = {'x': plot_age_exceptlast.tolist(),
                              'y': opt_euler_.tolist()}
                plot_dict['relee'] = relee_dict

                fig = plt.figure(figsize=std_figsize)
                ax = fig.add_subplot(1, 1, 1)
                lines, labs1, labs2 = [], ['mean'], ['capital', 'bond']
                l1, = plt.plot(plot_age_exceptlast, np.log10(
                    np.mean(opt_KKT_cap_, axis=0)), 'k-', label='capital, mean')
                l2, = plt.plot(plot_age_exceptlast, np.log10(
                    np.mean(opt_KKT_bond_, axis=0)), 'r-', label='bond, mean')
                lines.append([l1, l2])
                for perc_key in percentiles_dict:
                    l1, = plt.plot(plot_age_exceptlast, np.log10(np.percentile(opt_KKT_cap_, perc_key, axis=0)), 'k' +
                                   percentiles_dict[perc_key]['ls'], label='capital, ' + percentiles_dict[perc_key]['label']+' percentile')
                    l2, = plt.plot(plot_age_exceptlast, np.log10(np.percentile(opt_KKT_bond_, perc_key, axis=0)), 'r' +
                                   percentiles_dict[perc_key]['ls'], label='bond, ' + percentiles_dict[perc_key]['label']+' percentile')
                    lines.append([l1, l2])
                    labs1.append(
                        percentiles_dict[perc_key]['label'] + ' percentile')
                ax.set_xlabel('Age')
                ax.set_ylabel('KKT constr viol [log10]')
                legend1 = plt.legend(
                    lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1,
                           bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name +
                            '_opt_KKT_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                opt_KKT_dict = {'x': plot_age_exceptlast.tolist(
                ), 'y1': opt_KKT_cap_.tolist(), 'y2': opt_KKT_bond_.tolist()}
                plot_dict['opt_KKT'] = opt_KKT_dict

                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1, 1, 1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(c_all_orig_, perc_key, axis=0), 'k' +
                             percentiles_dict[perc_key]['ls'], label=percentiles_dict[perc_key]['label']+' percentile')
                ax1.plot(plot_age_all, np.mean(
                    c_all_orig_, axis=0), 'k-', label='mean')
                ax1.set_ylabel('c')
                ax1.set_xlabel('Age')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name +
                            '_cons_today_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cons_today_dict = {
                    'x': plot_age_all.tolist(), 'y': c_all_orig_.tolist()}
                plot_dict['cons_today'] = cons_today_dict

                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1, 1, 1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(c_all_origprime_1_, perc_key,
                             axis=0), shock_dict[1]['color']+percentiles_dict[perc_key]['ls'])
                    ax1.plot(plot_age_all, np.percentile(c_all_origprime_2_, perc_key,
                             axis=0), shock_dict[2]['color']+percentiles_dict[perc_key]['ls'])
                    ax1.plot(plot_age_all, np.percentile(c_all_origprime_3_, perc_key,
                             axis=0), shock_dict[3]['color']+percentiles_dict[perc_key]['ls'])
                    ax1.plot(plot_age_all, np.percentile(c_all_origprime_4_, perc_key,
                             axis=0), shock_dict[4]['color']+percentiles_dict[perc_key]['ls'])
                ax1.plot(plot_age_all, np.mean(c_all_origprime_1_, axis=0),
                         shock_dict[1]['color'], label=shock_dict[1]['label'])
                ax1.plot(plot_age_all, np.mean(c_all_origprime_2_, axis=0),
                         shock_dict[2]['color'], label=shock_dict[2]['label'])
                ax1.plot(plot_age_all, np.mean(c_all_origprime_3_, axis=0),
                         shock_dict[3]['color'], label=shock_dict[3]['label'])
                ax1.plot(plot_age_all, np.mean(c_all_origprime_4_, axis=0),
                         shock_dict[4]['color'], label=shock_dict[4]['label'])
                ax1.set_ylabel("c'")
                ax1.set_xlabel('Age')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name +
                            '_cons_tomorrow_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cons_tomorrow_dict = {'x1': plot_age_all.tolist(), 'x2': plot_age_all.tolist(), 'x3': plot_age_all.tolist(), 'x4': plot_age_all.tolist(),
                                      'y1': c_all_origprime_1_.tolist(), 'y2': c_all_origprime_2_.tolist(), 'y3': c_all_origprime_3_.tolist(), 'y4': c_all_origprime_4_.tolist()}
                plot_dict['cons_tomorrow'] = cons_tomorrow_dict

                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1, 1, 1)
                ax1.hist(np.abs(tot_bond_dem_), color='k', alpha=0.5)
                ax1.set_xlabel('Error aggregate bond demand')
                ax1.set_ylabel('Count')
                plt.savefig(plot_dir + '/' + run_name +
                            '_bond_clear_error_hist_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                # Show graph
                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1, 1, 1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(k_saved_all_, perc_key, axis=0), 'k' +
                             percentiles_dict[perc_key]['ls'], label=percentiles_dict[perc_key]['label']+' percentile')
                ax1.plot(plot_age_all, np.mean(
                    k_saved_all_, axis=0), 'k-', label='mean')
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Saving in capital')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name +
                            '_kpercs_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                kpercs_dict = {
                    'x': plot_age_all.tolist(), 'y': k_saved_all_.tolist()}
                plot_dict['kpercs'] = kpercs_dict

                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1, 1, 1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(fin_w_, perc_key, axis=0), 'k' +
                             percentiles_dict[perc_key]['ls'], label=percentiles_dict[perc_key]['label']+' percentile')
                ax1.plot(plot_age_all, np.mean(
                    fin_w_, axis=0), 'k-', label='mean')
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Financial wealth (wakeup)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name +
                            '_finwealthpercs_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                finwealthpercs_dict = {
                    'x': plot_age_all.tolist(), 'y': fin_w_.tolist()}
                plot_dict['finwealth'] = finwealthpercs_dict

                plt.figure(figsize=std_figsize)
                ax6 = plt.subplot(1, 1, 1)
                ax6.plot(np.arange(load_episode, ep+1),
                         np.log10(cost_store[0:ep-load_episode+1]), 'k-', label='evolution')
                ax6.plot(np.arange(load_episode, ep+1), np.log10(
                    mov_ave_cost_store[0:ep-load_episode+1]), 'r--', label='moving mean')
                ax6.set_xlabel('Episode')
                ax6.set_ylabel('log10(cost)')
                plt.savefig(plot_dir + '/' + run_name +
                            '_cost_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cost_dict = {'x': np.arange(
                    load_episode, ep+1).tolist(), 'y': cost_store[0:ep-load_episode+1].tolist()}
                plot_dict['cost'] = cost_dict

                if ep - load_episode > 1100:

                    plt.figure(figsize=std_figsize)
                    ax6 = plt.subplot(1, 1, 1)
                    ax6.plot(np.arange(ep+1 - 1000, ep+1), np.log10(
                        cost_store[ep-load_episode+1 - 1000:ep-load_episode+1]), 'k-', label='evolution')
                    ax6.plot(np.arange(ep+1 - 1000, ep+1), np.log10(
                        mov_ave_cost_store[ep-load_episode+1 - 1000:ep-load_episode+1]), 'r-', label='moving mean')
                    ax6.set_xlabel('Episode')
                    ax6.set_ylabel('log10(cost)')
                    plt.legend()
                    plt.savefig(plot_dir + '/' + run_name +
                                '_costLAST_episode' + str(ep)+'.pdf', bbox_inches='tight')
                    plt.close()

                    costLAST_dict = {'x': np.arange(ep+1 - 1000, ep+1).tolist(
                    ), 'y': cost_store[ep-load_episode+1 - 1000:ep-load_episode+1].tolist()}
                    plot_dict['costLAST'] = costLAST_dict

                saver = tf.train.Saver(nn.param_dict)
                save_param_path = save_base_path + \
                    '/model/' + run_name + '-episode' + str(ep)
                save_data_path = save_base_path + '/model/' + \
                    run_name + '-episode' + str(ep) + '_LastData.npy'
                saver.save(sess, save_param_path)
                print('Model saved in path: %s' % save_param_path)
                np.save(save_data_path, X_data_train)
                print('Last points saved at: %s' % save_data_path)

                if save_raw_plot_data:
                    save_plot_dict_path = path_wd + '/output/' + run_name + \
                        '/plotdata/' + run_name + \
                        'plot_dict_ep_'+str(ep)+'.json'
                    json.dump(plot_dict, codecs.open(
                        save_plot_dict_path, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
                    print('plot data saved to ' + save_plot_dict_path)

        params_dict = sess.run(nn.param_dict)
        for param_key in params_dict:
            params_dict[param_key] = params_dict[param_key].tolist()

        train_dict['params'] = params_dict

        result_dict['cost'] = cost_store.tolist()
        result_dict['time'] = time_store.tolist()
        result_dict['rel_Ee'] = ee_store.tolist()

        train_dict['results'] = result_dict

        end_time = datetime.now()
        print('Optimization Finished!')
        print('end time: {}'.format(end_time))
        print('total training time: {}'.format(end_time - start_time))

        train_writer.close()

        return train_dict


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_from_scratch',
                        dest='load_flag', action='store_false')
    parser.set_defaults(load_flag=True)
    args = parser.parse_args()
    load_flag = args.load_flag

    print('##### input arguments #####')
    seed = 1
    path_wd = '.'
    run_name = 'deqn_benchmark_restart' if args.load_flag else 'deqn_benchmark'
    num_hidden_nodes = [1000, 1000]
    activations_hidden_nodes = [tf.nn.relu, tf.nn.relu]
    optimizer = 'adam'
    batch_size = 64
    num_episodes = 60000
    len_episodes = 10000
    epochs_per_episode = 1
    save_interval = 100
    lr = 1e-5
    load_run_name = 'deqn_benchmark_final' if args.load_flag else None
    load_episode = 200000 if args.load_flag else 1
    save_raw_plot_data = False

    # For the 2nd training schedule: first train 60000 episodes from scratch, then uncomment the next 7 lines ######################
    # batch_size = 1000
    # num_episodes = 140000
    # lr = 1e-6
    # run_name = 'deqn_benchmark_2ndschedule'
    # load_flag = True
    # load_run_name = 'deqn_benchmark'
    # load_episode = 60000
    # ################################################################

    print('seed: {}'.format(seed))
    print('working directory: ' + path_wd)
    print('run_name: {}'.format(run_name))
    print('num_agents: {}'.format(num_agents))
    if not(save_raw_plot_data):
        print('raw plot data will not be saved. To change set "save_raw_plot_data" in line 1618 to True.')
    print('hidden nodes: [1000, 1000]')
    print('activation hidden nodes: [relu, relu]')

    if args.load_flag:
        train_flag = False
        num_episodes = 1
        print('loading weights from deqn_benchmark_final')
        print('loading from episode {}'.format(load_episode))
    else:
        train_flag = True
        print('optimizer: {}'.format(batch_size))
        print('batch_size: {}'.format(batch_size))
        print('num_episodes: {}'.format(num_episodes))
        print('len_episodes: {}'.format(len_episodes))
        print('epochs_per_episode: {}'.format(epochs_per_episode))
        print('save_interval: {}'.format(save_interval))
        print('lr: {}'.format(lr))

    print('###########################')

    train_dict = train(path_wd, run_name,
                       num_episodes, len_episodes, epochs_per_episode,
                       batch_size, optimizer, lr,
                       save_interval, num_hidden_nodes,
                       activations_hidden_nodes, train_flag=train_flag,
                       load_flag=load_flag, load_run_name=load_run_name,
                       load_episode=load_episode, seed=seed, save_raw_plot_data=save_raw_plot_data)

    # Save outputs
    train_dict['net_setup']['activations_hidden_nodes'] = ['relu', 'relu']
    save_train_dict_path = os.path.join(
        '.', 'output', run_name, 'json', 'train_dict.json')
    json.dump(train_dict, codecs.open(save_train_dict_path, 'w',
              encoding='utf-8'), separators=(',', ':'), indent=4)
    print('Saved dictionary to:' + save_train_dict_path)


if __name__ == '__main__':
    main()
