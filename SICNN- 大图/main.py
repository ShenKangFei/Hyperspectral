# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:11:22 2018
Thanks for my girlfriend's support
@author: Jiantong Chen
"""
import time
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from skimage import color, io
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, windowFeature, one_hot, disorder, next_batch
from processing_library import first_layer, conv_layer_same, conv_layer_valid, contrary_one_hot
from processing_library import order_weight_fixed, index_band_selection, save_result, DrawResult
from pre_color import plot_label
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


###############################################################################
class RET:
    def __init__(self, data_name="Indian_pines", num_band_seclection=60):
        self.data_name = data_name
        data_norm, labels_ori, x_train, y_train, train_loc, x_test, y_test, test_loc = load_data(self.data_name)
        nrows = data_norm.shape[0]
        ncols = data_norm.shape[1]
        dim_input = np.shape(data_norm)[2]
        batch_size = 128
        display_step = 1000
        step = 1
        index = batch_size
        LR_RATE = 0.01
        num_classification = int(np.max(labels_ori))  # 类别数
        w = 15  # 图像块大小
        global_step = tf.Variable(step)
        learn_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.8, staircase=False)  # 学习率
        num_epoch = 20  # 训练循环次数
        #         num_band_seclection=60#要选择的波段数
        # REGULARIZATION_RATE = 0.0001 # 正则化项的权重系数
        ###############################################################################
        X_train = windowFeature(data_norm, train_loc, w)
        X_test = windowFeature(data_norm, test_loc, w)

        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        Y_train = one_hot(y_train, num_classification)
        Y_test = one_hot(y_test, num_classification)

        X_train, Y_train = disorder(X_train, Y_train)
        X_test, Y_test = disorder(X_test, Y_test)
        X_data = windowFeature(data_norm, np.transpose(list(itertools.product(list(range(145)), repeat=2)), [1, 0]), w)

        ###############################################################################
        # create_variable

        def l1_loss(params):
            '''L1损失函数'''
            return tf.reduce_sum(tf.abs(params))

        def create_variable(name, shape, weight_decay=None, loss=l1_loss):
            with tf.device("/gpu:0"):
                var = tf.get_variable(name, dtype=tf.float32, shape=shape,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05))

            if weight_decay:
                wd = loss(var) * weight_decay
                tf.add_to_collection("weight_decay", wd)

            return var

        def elastic_loss(params, alpha=0.5):
            '''elastic损失函数'''
            return tf.add((1 - alpha) * 0.5 * tf.nn.l2_loss(params), alpha * l1_loss(params))

        def b_loss(params, num_band_selection=30):
            '''||sum(sigma)-num_bandselection||约束'''
            return tf.nn.l2_loss(tf.reduce_sum(tf.abs(params)) - num_band_selection)

        ###############################################################################
        weights = {'W1': tf.Variable(tf.random_uniform([1, 1, dim_input, 1], minval=0.48, maxval=0.5)),

                   'W2': tf.Variable(tf.truncated_normal([1, 1, dim_input, 32], stddev=0.1)),

                   'W3': tf.Variable(tf.truncated_normal([3, 3, dim_input, 32], stddev=0.1)),

                   'W4': tf.Variable(tf.truncated_normal([4, 4, 2 * 32, 64], stddev=0.1)),

                   'W5': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),

                   'W6': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),

                   'W7': tf.Variable(tf.truncated_normal([256, 512], stddev=0.1)),

                   'W8': tf.Variable(tf.truncated_normal([512, num_classification], stddev=0.1)),

                   'W9': tf.Variable(tf.truncated_normal([2 * 32, num_classification], stddev=0.1)),

                   'W10': tf.Variable(tf.truncated_normal([128, num_classification], stddev=0.1)),

                   'W11': tf.Variable(tf.truncated_normal([256, num_classification], stddev=0.1)),

                   'W12': tf.Variable(tf.truncated_normal([512, num_classification], stddev=0.1))
                   }

        bias = {'B1': tf.Variable(tf.constant(0.1, shape=[dim_input])),

                'B2': tf.Variable(tf.constant(0.1, shape=[32])),

                'B3': tf.Variable(tf.constant(0.1, shape=[32])),

                'B4': tf.Variable(tf.constant(0.1, shape=[64])),

                'B5': tf.Variable(tf.constant(0.1, shape=[128])),

                'B6': tf.Variable(tf.constant(0.1, shape=[256])),

                'B7': tf.Variable(tf.constant(0.1, shape=[512])),

                'B8': tf.Variable(tf.constant(0.1, shape=[num_classification])),

                'B9': tf.Variable(tf.constant(0.1, shape=[num_classification])),

                'B10': tf.Variable(tf.constant(0.1, shape=[num_classification])),

                'B11': tf.Variable(tf.constant(0.1, shape=[num_classification])),

                'B12': tf.Variable(tf.constant(0.1, shape=[num_classification]))
                }
        ###############################################################################
        x = tf.placeholder(tf.float32, [None, w, w, dim_input], name='x_input')
        y = tf.placeholder(tf.float32, [None, num_classification], name='y_output')
        x_reshape = tf.reshape(x, shape=[-1, w, w, dim_input])
        keep_prob = tf.placeholder(tf.float32)

        WWW = tf.placeholder(tf.float32, [1, 1, dim_input, 1], name='x_input')

        conv1 = first_layer(x_reshape, WWW, bias['B1'], [1, 1, 1, 1])

        conv2 = conv_layer_same(conv1, weights['W2'], bias['B2'], [1, 1, 1, 1])
        conv3 = conv_layer_same(conv1, weights['W3'], bias['B3'], [1, 1, 1, 1])
        youknow = tf.concat([conv2, conv3], 3)

        conv4 = conv_layer_valid(youknow, weights['W4'], bias['B4'], [1, 1, 1, 1])

        pool5 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv6 = conv_layer_same(pool5, weights['W5'], bias['B5'], [1, 1, 1, 1])

        dpt7 = tf.nn.dropout(conv6, keep_prob)

        pool8 = tf.nn.max_pool(dpt7, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv9 = conv_layer_valid(pool8, weights['W6'], bias['B6'], [1, 1, 1, 1])

        dpt10 = tf.nn.dropout(conv9, keep_prob)
        reshape = tf.reshape(dpt10, [-1, weights['W7'].get_shape().as_list()[0]])

        f11 = tf.nn.relu(tf.add(tf.matmul(reshape, weights['W7']), bias['B7']))

        f12 = tf.add(tf.matmul(f11, weights['W8']), bias['B8'])

        y_ = tf.nn.softmax(f12)

        out1 = tf.add(tf.matmul(
            tf.reshape(tf.slice(youknow, [0, int(w / 2), int(w / 2), 0], [batch_size, 1, 1, 2 * 32]),
                       shape=[-1, 2 * 32]), weights['W9']), bias['B9'])
        out2 = tf.add(tf.matmul(
            tf.reshape(tf.slice(pool8, [0, int(3 / 2), int(3 / 2), 0], [batch_size, 1, 1, 128]), shape=[-1, 128]),
            weights['W10']), bias['B10'])
        out3 = tf.add(tf.matmul(reshape, weights['W11']), bias['B11'])
        out4 = tf.add(tf.matmul(f11, weights['W12']), bias['B12'])

        # writer = tf.summary.FileWriter('D:/ten',tf.get_default_graph())
        # writer.close()
        ###############################################################################
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f12, name=None))
        cross_entropy_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out1, name=None))
        cross_entropy_yy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out2, name=None))
        cross_entropy_yyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out3, name=None))
        cross_entropy_yyyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out4, name=None))

        #        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #        regularization = regularizer(regularizer(weights['W2'])+regularizer(weights['W3'])+regularizer(weights['W4'])+regularizer(weights['W5'])+regularizer(weights['W6']))
        loss = tf.add(tf.add(tf.add(tf.add(cross_entropy, 0.5 * cross_entropy_y), 0.5 * cross_entropy_yy),
                             0.5 * cross_entropy_yy), 0.5 * cross_entropy_yyyy)
        weight_decay_loss = LR_RATE * b_loss(WWW, num_band_seclection)
        regularized_loss = cross_entropy
        # train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(regularized_loss,global_step)
        train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(regularized_loss, global_step)

        g = tf.where(tf.abs(weights['W1']) <= 1, tf.multiply(learn_rate, tf.gradients(regularized_loss, WWW)[0]),
                     tf.zeros_like(weights['W1']))
        # g=tf.multiply(learn_rate,tf.gradients(regularized_loss,WWW)[0])

        op = tf.assign(weights['W1'], tf.subtract(weights['W1'], g))
        # op = tf.cond(tf.abs(weights['W1']) < 1, lambda: ass, lambda: c)

        ###############################################################################
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        init = tf.global_variables_initializer()

        ###############################################################################
        def get_oa(X_valid, Y_valid):
            size = np.shape(X_valid)
            num = size[0]
            index_all = 0
            step_ = 3000
            y_pred = []
            while index_all < num:
                if index_all + step_ > num:
                    input = X_valid[index_all:, :, :, :]
                else:
                    input = X_valid[index_all:(index_all + step_), :, :, :]
                index_all += step_
                temp1 = y_.eval(
                    feed_dict={x: input, keep_prob: 1.0, WWW: order_weight_fixed(weights['W1'], num_band_seclection)})
                y_pred1 = contrary_one_hot(temp1).astype('int32')
                y_pred.extend(y_pred1)
            y = contrary_one_hot(Y_valid).astype('int32')
            return y_pred, y

        ###############################################################################
        import os
        randtime = 1
        f = open(os.getcwd() + '/实验结果.txt', 'a')
        f.writelines(
            "###############################################################################\n" + "Indian_pines_" + str(
                num_band_seclection) + "\n")
        f.close()
        for r in range(0, randtime):
            t1 = time.time()
            with tf.Session() as sess:

                sess.run(init)
                epoch = 0
                time_train_start = time.clock()
                start = time.time()
                a = 1
                b = 1
                while epoch < num_epoch:
                    if epoch == 0:
                        if a == 1:
                            t = time.time()
                            a += 1
                    if epoch == 1:
                        if b == 1:
                            b += 1
                            T = time.time()
                            print('训练一轮时间' + str(T - t))
                    batch_x, batch_y = next_batch(X_train, Y_train, index, batch_size)
                    sess.run([op, train_step], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5,
                                                          WWW: order_weight_fixed(weights['W1'], num_band_seclection)})

                    #        print(index_band_selection(order_weight_fixed(weights['W1'],num_band_seclection)))
                    #        print(order_weight_fixed(weights['W1'],num_band_seclection)[0,0,:,0])
                    #        print(conv1.eval(feed_dict={x: batch_x,W1:order_weight_fixed(weights['W1'],num_band_seclection)})[0,:,:,0])

                    if step % display_step == 0:
                        cros, cros_yy, los, acc, regula_loss = sess.run(
                            [cross_entropy, cross_entropy_yy, loss, accuracy, weight_decay_loss],
                            feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0,
                                       WWW: order_weight_fixed(weights['W1'], num_band_seclection)})
                        #            acc=sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,keep_prob:1.0,WWW:order_weight_fixed(weights['W1'],num_band_seclection)})
                        #                print(weights["W1"].eval())
                        print('step %d,training accuracy %f' % (step, acc))
                        print('loss %f,cross_entropy %f,cross_entropy_yy %f,reg_loss %f' % (
                        los, cros, cros_yy, regula_loss))
                        y_pr, y_tr = get_oa(X_test, Y_test)
                        oa = accuracy_score(y_tr, y_pr)
                        print('valid accuracy %f' % (oa))
                    index = index + batch_size
                    step += 1
                    if index > X_train.shape[0]:
                        index = batch_size
                        epoch = epoch + 1
                time_train_end = time.clock()
                t2 = time.time()
                print("Optimization Finished!")
                band_loction = index_band_selection(order_weight_fixed(weights['W1'], num_band_seclection))
                print(band_loction)

                time_test_start = time.clock()
                y_pr, y_real = get_oa(X_test, Y_test)
                oa = accuracy_score(y_real, y_pr)
                per_class_acc = recall_score(y_real, y_pr, average=None)
                aa = np.mean(per_class_acc)
                kappa = cohen_kappa_score(y_real, y_pr)
                time_test_end = time.clock()
                print("Time_train")
                num_band_seclection_now = len(band_loction)
                save_result(data_name, oa, aa, kappa, num_band_seclection_now, band_loction, per_class_acc,
                            (time_train_end - time_train_start), (time_test_end - time_test_start))

                print(per_class_acc)
                print(oa, aa, kappa)
                print((time_train_end - time_train_start), (time_test_end - time_test_start))
                # print("train accuracy %g"%acc)
                self.results = [oa, aa, kappa] + per_class_acc.tolist() + [time_train_end - time_train_start,
                                                                           time_test_end - time_test_start]
                # plot

                plot_loc = np.array([[i, j] for i in range(nrows) for j in range(ncols)])
                order = np.random.permutation(range(plot_loc.shape[0]))
                plot_loc = plot_loc[order]
                plot_loc = plot_loc.transpose()

                idx_plot = 0
                y_plot = []
                batch_size = 10000
                while (idx_plot + batch_size) < plot_loc.shape[1]:
                    X_plot = windowFeature(data_norm, plot_loc[:, idx_plot:idx_plot + batch_size], w)
                    plot_labela = np.zeros([X_plot.shape[0]])
                    y_plot_t, _ = get_oa(X_plot, plot_labela)
                    y_plot.extend(y_plot_t)
                    idx_plot += batch_size
                    print(idx_plot, "/", plot_loc.shape[1])
                if idx_plot < plot_loc.shape[1] - 1:
                    X_plot = windowFeature(data_norm, plot_loc[:, idx_plot:], w)
                    plot_labela = np.zeros([X_plot.shape[0]])
                    y_plot_t, _ = get_oa(X_plot, plot_labela)
                    y_plot.extend(y_plot_t)

                plot = np.zeros([nrows, ncols])
                for idx, item in enumerate(order.tolist()):
                    plot[item // ncols, item % ncols] = y_plot[idx]
                #            y_plot = np.array(y_plot).reshape([nrows,ncols])
                for i in range(labels_ori.shape[0]):
                    for j in range(labels_ori.shape[1]):
                        if labels_ori[i, j] == 0:
                            if self.data_name != "Houston":
                                plot[i, j] = 0
                        # else:
                        #     plot[i,j] = 1
                plot_model = plot_label(data_name)
                img = plot_model.plot_color(plot)
                plt.imsave(data_name + "_" + str(oa) + ".tiff", img)


if __name__ == '__main__':
    ret = RET("PaviaU")
    print(ret.results)
