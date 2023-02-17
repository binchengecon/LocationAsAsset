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
    save_root_path = os.path.join(path_wd,'output')
    save_base_path = os.path.join(path_wd,'output', run_name)
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
    
    os.makedirs(path_wd,exist_ok=True)
    
    if 'output' not in os.listdir(path_wd):
        os.makedirs(save_root_path,exist_ok=True)

    if run_name not in os.listdir(save_root_path):
        os.makedirs(save_base_path,exist_ok=True)
        os.makedirs(os.path.join(save_base_path, 'json'),exist_ok=True)
        os.makedirs(os.path.join(save_base_path, 'model'),exist_ok=True)
        os.makedirs(os.path.join(save_base_path, 'plots'),exist_ok=True)
        os.makedirs(os.path.join(save_base_path, 'plotdata'),exist_ok=True)
        os.makedirs(os.path.join(save_base_path, 'tensorboard'),exist_ok=True)

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

    econ_setup_dict['beta'] = BETA
    econ_setup_dict['sigma'] = SIGMA
    econ_setup_dict['s'] = S
    econ_setup_dict['y'] = Y.tolist()
    econ_setup_dict['r'] = R
    econ_setup_dict['alb'] = A_LB
    econ_setup_dict['zub'] = Z_UB
    econ_setup_dict['zlb'] = Z_LB
    econ_setup_dict['q2'] = Q2
    econ_setup_dict['q1'] = Q1
    econ_setup_dict['gamma'] = GAMMA.tolist()

    train_dict['econ_params'] = econ_setup_dict

    for key in econ_setup_dict:
        print('{}: {}'.format(key, econ_setup_dict[key]))

    with tf.name_scope('econ_parameters'):

        beta = tf.constant(BETA, dtype=tf.float32, name='beta')
        sigma = tf.constant(SIGMA, dtype=tf.float32, name='sigma')
        s = tf.constant(S, dtype=tf.float32, name='s')
        y = tf.constant(Y, dtype=tf.float32, name='y')
        r = tf.constant(R, dtype=tf.float32, name='r')
        alb = tf.constant(A_LB, dtype=tf.float32, name='alb')
        zub = tf.constant(Z_UB, dtype=tf.float32, name='zub')
        zlb = tf.constant(Z_LB, dtype=tf.float32, name='zlb')
        q2 = tf.constant(Q2, dtype=tf.float32, name='q2')
        q1 = tf.constant(Q1, dtype=tf.float32, name='q1')
        gamma = tf.constant(GAMMA, dtype=tf.float32, name='gamma')

    with tf.name_scope('neural_net'):
        # shock number, shock value (labor income), last period capital, location income, capital income, total income, transition probibility
        n_input = 6 + NUM_EX_SHOCKS 
        n_output = 3 # saving, KKT capital multiplier, location choice

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
                    num = X[:, 0]
                with tf.name_scope('labor_income'):
                    inc_y = X[:, 1]
                with tf.name_scope('old_capital'):
                    a = X[:, 2]
                with tf.name_scope('location_income'):
                    inc_z = X[:, 3]
                with tf.name_scope('capital_income'):
                    inc_a = X[:, 4]
                with tf.name_scope('total_income'):
                    inc_all = X[:, 5]
                with tf.name_scope('probs_tomorrow'):
                    # probabilities for shocks tomorrow
                    probs_next = X[:, 6: 6 + NUM_EX_SHOCKS]

            with tf.name_scope('get_todays_control'):
                # get todays savings by executing the neural net
                with tf.name_scope('NN'):
                    predprime = nn.predict(X)
                    aprime = predprime[:, 0] # a_hat(X)
                    lambd = predprime[:, 1] # lambda_hat(X)
                    zprime = predprime[:, 2] # z_hat(X)
                    
            with tf.name_scope('LHS_aprime_qzprime'):
                
                aprime_wakeup = tf.expand_dims(aprime,-1)
                zprime_wakeup = tf.expand_dims(zprime,-1)
                lambd_wakeup = tf.expand_dims(lambd,-1)
                
                cap_saved_wakeup = aprime_wakeup
                loc_spent_wakeup = zprime_wakeup**2*q2 + zprime_wakeup*q1


            with tf.name_scope('compute_todays_consumption'):
                tot_saved_wakeup = cap_saved_wakeup + loc_spent_wakeup
                inc_all_wakeup = tf.expand_dims(inc_all,-1)
                
                c_orig = inc_all_wakeup - tot_saved_wakeup
                c = tf.maximum(c_orig, tf.ones_like(
                    c_orig) * eps, name='c_today') # make sure non-trivial consumption # c_hat(X)

            with tf.name_scope('get_tomorrows_state'): # build X+
                with tf.name_scope('tomorrows_exog_shock'):
                    # state tomorrow
                    num1prime = tf.zeros_like(num, name='num_yprime_1')
                    num2prime = tf.ones_like(num, name='num_yprime_2')

                with tf.name_scope('tomorrows_exog_param'):
                    # inc_y tomorrow
                    inc_y_prime_1 = tf.gather(y, tf.cast(
                        num1prime, tf.int32), name='inc_y_prime_1')
                    inc_y_prime_2 = tf.gather(y, tf.cast(
                        num2prime, tf.int32), name='inc_y_prime_2')
                    
                # with tf.name_scope('tomorrows_capital'):
                #     aprime_wakeup = tf.maximum(aprime_wakeup, tf.ones_like(aprime_wakeup) * eps, name='aprime_tomorrow')


                with tf.name_scope('tomorrows_location_wealth'):
                    inc_z_prime = zprime_wakeup * s
                
                with tf.name_scope('tomorrows_capital_wealth'):
                    inc_a_prime = aprime_wakeup * r
                
                with tf.name_scope('tomorrows_total_wealth'):
                    # individuals total wealth
                    inc_all_prime_1 = inc_y_prime_1 +  inc_z_prime + inc_a_prime
                    inc_all_prime_2 = inc_y_prime_2 +  inc_z_prime + inc_a_prime

                
                with tf.name_scope('tomorrows_transition_probabilities'):
                    gamma_transprime_1 = tf.gather(gamma, tf.cast(num1prime, tf.int32)) # defualt as gathering rows, and stack rows
                    gamma_trans_to1prime_1 = tf.expand_dims(
                        gamma_transprime_1[:, 0], -1)
                    gamma_trans_to2prime_1 = tf.expand_dims(
                        gamma_transprime_1[:, 1], -1)

                    gamma_transprime_2 = tf.gather(gamma, tf.cast(num2prime, tf.int32))
                    gamma_trans_to1prime_2 = tf.expand_dims(
                        gamma_transprime_2[:, 0], -1)
                    gamma_trans_to2prime_2 = tf.expand_dims(
                        gamma_transprime_2[:, 1], -1)



                with tf.name_scope('concatenate_for_tomorrows_state'):
                    xprime_1 = tf.concat([tf.expand_dims(num1prime, -1),
                                           inc_y_prime_1,
                                           aprime_wakeup,
                                           inc_z_prime,
                                           inc_a_prime,
                                           inc_all_prime_1,
                                          gamma_trans_to1prime_1,
                                          gamma_trans_to2prime_1,],
                                         axis=1, name='state_tomorrow_1')

                    xprime_2 = tf.concat([tf.expand_dims(num2prime, -1),
                                           inc_y_prime_2,
                                           aprime_wakeup,
                                           inc_z_prime,
                                           inc_a_prime,
                                           inc_all_prime_2,
                                          gamma_trans_to1prime_2,
                                          gamma_trans_to2prime_2,],
                                         axis=1, name='state_tomorrow_2')
                    


        with tf.name_scope('get_tomorrows_consumption'):
            with tf.name_scope('get_tomorrows_saving'):
                with tf.name_scope('NN'):
                    
                    predprimeprime_1 = nn.predict(xprime_1)
                    aprimeprime_1 = predprimeprime_1[:, 0]
                    lambdaprime_1 = predprimeprime_1[:, 1]
                    zprimeprime_1 = predprimeprime_1[:, 2]

                    predprimeprime_2 = nn.predict(xprime_2)
                    aprimeprime_2 = predprimeprime_2[:, 0]
                    lambdaprime_2 = predprimeprime_2[:, 1]
                    zprimeprime_2 = predprimeprime_2[:, 2]

            with tf.name_scope('aprimeprime_wakeup_all'):
                aprimeprime_wakeup_1 = tf.expand_dims(aprimeprime_1,-1)
                zprimeprime_wakeup_1 = tf.expand_dims(zprimeprime_1,-1)
                
                aprimeprime_wakeup_2 = tf.expand_dims(aprimeprime_2,-1)
                zprimeprime_wakeup_2 = tf.expand_dims(zprimeprime_2,-1)
                
                cap_saved_prime_wakeup_1 = aprimeprime_wakeup_1
                loc_spent_prime_wakeup_1 = zprimeprime_wakeup_1**2 * q2 + zprimeprime_wakeup_1*q1
                loc_spent_deriv_prime_wakeup_1 = zprimeprime_wakeup_1 * 2 * q2  +  q1

                cap_saved_prime_wakeup_2 = aprimeprime_wakeup_2
                loc_spent_prime_wakeup_2 = zprimeprime_wakeup_2**2 * q2 + zprimeprime_wakeup_2 *q1
                loc_spent_deriv_prime_wakeup_2 = zprimeprime_wakeup_2 * 2 * q2  +  q1

            with tf.name_scope('kprimeprime_save_all'):

                tot_saved_prime_wakeup_1 = cap_saved_prime_wakeup_1 + loc_spent_prime_wakeup_1
                tot_saved_prime_wakeup_2 = cap_saved_prime_wakeup_2 + loc_spent_prime_wakeup_2


            with tf.name_scope('compute_tomorrows_consumption'):
                c_origprime_1 = inc_all_prime_1 - tot_saved_prime_wakeup_1
                c_prime_1 = tf.maximum(c_origprime_1, tf.ones_like(
                    c_origprime_1) * eps, name='c_tmr_prime_1')
                
                c_origprime_2 = inc_all_prime_2 - tot_saved_prime_wakeup_2
                c_prime_2 = tf.maximum(c_origprime_2, tf.ones_like(
                    c_origprime_2) * eps, name='c_tmr_prime_2')


            with tf.name_scope('optimality_conditions'):
                # optimality conditions
                with tf.name_scope('rel_ee'):
                    # prepare transitions
                    pi_trans_to1 = tf.expand_dims(
                        probs_next[:, 0], -1) 
                    pi_trans_to2 = tf.expand_dims(
                        probs_next[:, 1], -1) 
                    
                    
                    # euler equation
                    
                    cap_expect = pi_trans_to1 * c_prime_1**(- 1/sigma) + pi_trans_to2 * c_prime_2**(- 1/sigma)
                    disc_cap_expect = beta * r * cap_expect
                    cap_utility = disc_cap_expect + lambd_wakeup
                    inv_cap_utility = cap_utility**(-sigma)
                    opt_euler_cap = inv_cap_utility/c-1 


                    loc_expect_upper = pi_trans_to1 *  c_prime_1**(- 1/sigma) + pi_trans_to2 * c_prime_2**(- 1/sigma)
                    loc_expect_lower = pi_trans_to1 *  loc_spent_deriv_prime_wakeup_1+ pi_trans_to2 * loc_spent_deriv_prime_wakeup_2
                    loc_expect_ratio =     loc_expect_upper/loc_expect_lower 
                    disc_loc_expect_ratio = beta * s * loc_expect_ratio
                    inv_loc_utility = disc_loc_expect_ratio**(-sigma)
                    opt_euler_loc = inv_loc_utility/c - 1
                    

                    opt_euler = tf.concat(
                        [opt_euler_cap, opt_euler_loc], axis=1)

                    # KKT condition
                    # The condition that aprime >= 0 and lambd >= 0 are enforced by softplus activation in the output layer
                    # The condition that zprime >= 0 is  enforced by softplus activation in the output layer
                    opt_KKT_cap = aprime_wakeup * lambd_wakeup
                    
                    opt_KKT = opt_KKT_cap

                with tf.name_scope('punishments'):
                    # punishment for negative cons
                    # orig_cons = tf.concat(
                    #     [tf.expand_dims(c_orig,-1), tf.reshape(c_origprime_1,[m,1]), tf.reshape(c_origprime_2,[m,1])], axis=1)
                    orig_cons = tf.concat(
                        [c_orig, c_origprime_1, c_origprime_2], axis=1)

                    opt_punish_cons = (1.0/eps) * tf.maximum(-1 * orig_cons,
                                                             tf.zeros_like(orig_cons), name='opt_punish_cons')

                    # # punishment for loc larger than upper bound z_ub
                    # opt_punish_loc = (1.0/eps) * tf.maximum(zprime-1,
                    #                                          tf.zeros_like(zprime), name='opt_punish_loc')


                # Put together
                # combined_opt = [opt_euler, opt_punish_cons,
                #                 tf.expand_dims(opt_punish_loc, -1), opt_KKT]
                combined_opt = [opt_euler, opt_punish_cons, opt_KKT]
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
            # capped_gvs = [(tf.clip_by_value(grad, -2.0, 2.0), var)
            #               for grad, var in gvs]
            # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var)
            #             for grad, var in gvs]
            capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var)
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
                if rand_num[t - 1] <= GAMMA[int(X_old[0, 0]), 0]:
                    X_new = sess.run(xprime_1, feed_dict={X: X_old})
                else :
                    X_new = sess.run(xprime_2, feed_dict={X: X_old})
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
                trans_probs_to1 = X_old[:, 6]

                to_1 = temp_rand <= trans_probs_to1
                to_2 = temp_rand > trans_probs_to1

                # print(to_1.shape)
                # print(X_old[to_1, :].shape)
                # print(sess.run(
                #     tf.shape(num1prime), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(inc_y_prime_1), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(aprime_wakeup), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(inc_z_prime), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(inc_y_prime_1), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(inc_all_prime_1), feed_dict={X: X_old[to_1, :]}))               
                # print(sess.run(
                #     tf.shape(gamma_trans_to1prime_1), feed_dict={X: X_old[to_1, :]}))                
                # print(sess.run(
                #     tf.shape(gamma_trans_to1prime_2), feed_dict={X: X_old[to_1, :]}))
                # print(sess.run(
                #     tf.shape(xprime_1), feed_dict={X: X_old[to_1, :]}))
                # WORK WELL UNTIL xprime_1
                
                X_new[to_1, :] = sess.run(
                    xprime_1, feed_dict={X: X_old[to_1, :]})
                X_new[to_2, :] = sess.run(
                    xprime_2, feed_dict={X: X_old[to_2, :]})

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
        ee_store = np.zeros((num_episodes, 2))
        max_ee_store = np.zeros((num_episodes, 2))

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
                    ee_error = np.zeros((1, 2))
                    max_ee = np.zeros((1,2))

                for minibatch in minibatches:
                    (minibatch_X) = minibatch

                    # Run optimization
                    minibatch_cost += sess.run(cost,
                                               feed_dict={X: minibatch_X}) / num_minibatches
                    if epoch == 0:
                        
                        # print(minibatch_X.shape)

                        # print(sess.run(c, feed_dict={
                        #                     X: minibatch_X}).shape)
                        # print(sess.run(c, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(c, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(c, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(c, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(c, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(opt_euler, feed_dict={
                        #                 X: minibatch_X}).shape)
                        # print(sess.run(aprimeprime_1, feed_dict={X: minibatch_X}).shape)
                        
                        
                        
                        
                        
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

                # plot_age_all = np.arange(25, 25+A)
                # plot_age_exceptlast = np.arange(25, 25+A - 1)

                plt.rc('font', family='serif')
                plt.rc('xtick', labelsize='small')
                plt.rc('ytick', labelsize='small')

                std_figsize = (4, 4)
                # percentiles_dict = {50: {'ls': ':', 'label': '50'}, 10: {'ls': '-.', 'label': '10'}, 90: {
                #     'ls': '-.', 'label': '90'}, 99.9: {'ls': '--', 'label': '99.9'}, 0.1: {'ls': '--', 'label': '0.1'}}

                shock1_dict = {'label': 'shock 1', 'color': 'r'}
                shock2_dict = {'label': 'shock 2', 'color': 'b'}
                # shock3_dict = {'label': 'shock 3', 'color': 'y'}
                # shock4_dict = {'label': 'shock 4', 'color': 'g'}
                # shock_dict = {1: shock1_dict, 2: shock2_dict,
                #               3: shock3_dict, 4: shock4_dict}
                shock_dict = {1: shock1_dict, 2: shock2_dict}
                


                # run stuff
                c_orig_ = sess.run(c_orig, feed_dict={X: X_episodes})
                c_origprime_1_ = sess.run(
                    c_origprime_1, feed_dict={X: X_episodes})
                c_origprime_2_ = sess.run(
                    c_origprime_2, feed_dict={X: X_episodes})

                zprime_ = sess.run(zprime, feed_dict={X: X_episodes})


                # tot_saved_all_ = sess.run(
                #     tot_saved_all, feed_dict={X: X_episodes})
                # inc_ = sess.run(inc, feed_dict={X: X_episodes})

                lambd_ = sess.run(lambd, feed_dict={X: X_episodes})
                # mu_ = sess.run(mu, feed_dict={X: X_episodes})
                
                opt_euler_cap_ = sess.run(
                    opt_euler_cap, feed_dict={X: X_episodes})
                opt_euler_loc_ = sess.run(
                    opt_euler_loc, feed_dict={X: X_episodes})
                opt_KKT_cap_ = sess.run(opt_KKT_cap, feed_dict={X: X_episodes})


                a_cond1 = (X_episodes[:plot_epi_length, 0] == 0)
                a_cond2 = (X_episodes[:plot_epi_length, 0] == 1)
                
                cost_dict = {'x': np.arange(
                    load_episode, ep+1).tolist(), 'y': cost_store[0:ep-load_episode+1].tolist()}
                plot_dict['cost'] = cost_dict



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
    # path_wd = "/scratch/bincheng/DeepLearning/"
    path_wd = "."
    run_name = 'deqn_benchmark_restart' if args.load_flag else 'deqn_benchmark'
    num_agents = 1
    num_hidden_nodes = [30, 30]
    activations_hidden_nodes = [tf.nn.relu, tf.nn.relu]
    optimizer = 'adam'
    batch_size = 128
    num_episodes = 60000
    len_episodes = 10000
    epochs_per_episode = 1
    save_interval = 100
    lr = 1e-5
    load_run_name = 'deqn_benchmark_final' if args.load_flag else None
    load_episode = 200000 if args.load_flag else 1
    save_raw_plot_data = False

    # For the 2nd training schedule: first train 60000 episodes from scratch, then uncomment the next 7 lines ######################
    batch_size = 1000
    num_episodes = 140000
    lr = 1e-6
    run_name = 'deqn_benchmark_2ndschedule'
    load_flag = True
    load_run_name = 'deqn_benchmark'
    load_episode = 60000
    # ################################################################

    print('seed: {}'.format(seed))
    print('working directory: ' + path_wd)
    print('run_name: {}'.format(run_name))
    print('num_agents: {}'.format(num_agents))
    if not(save_raw_plot_data):
        print('raw plot data will not be saved. To change set "save_raw_plot_data" in line 1618 to True.')
    print('hidden nodes: {}'.format(num_hidden_nodes))
    print('activation hidden nodes: [relu, relu]')

    if args.load_flag:
        train_flag = False
        num_episodes = 1
        print('loading weights from deqn_benchmark_final')
        print('loading from episode {}'.format(load_episode))
    else:
        train_flag = True
        print('optimizer: {}'.format(optimizer))
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
        path_wd, 'output', run_name, 'json', 'train_dict.json')
    json.dump(train_dict, codecs.open(save_train_dict_path, 'w',
              encoding='utf-8'), separators=(',', ':'), indent=4)
    print('Saved dictionary to:' + save_train_dict_path)


if __name__ == '__main__':
    main()
