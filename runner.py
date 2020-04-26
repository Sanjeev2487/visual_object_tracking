from __future__ import division

import os
import random
import sys
import logging
import argparse

import cv2
#import fire
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

import time
import math

import commons
from boundingbox import BoundingBox, Coordinate
from configs import ADNetConf
from networks import ADNetwork
from pystopwatch import StopWatchManager

_log_level = logging.DEBUG
_logger = logging.getLogger('ADNetRunner')
_logger.setLevel(_log_level)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(_log_level)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)


class ADNetRunner:
    MAX_BATCHSIZE = 4

    def __init__(self, model_path, verbose=True, load_baseline=False):
        self.tensor_input = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='patch')
        self.tensor_action_history = tf.placeholder(tf.float32, shape=(None, 1, 1, 110), name='action_history')
        self.tensor_lb_action = tf.placeholder(tf.int32, shape=(None, ), name='lb_action')
        self.tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')
        self.reward = tf.placeholder(tf.float32, shape=(None, ), name='reward')

        self.tensor_is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads=1
        config.intra_op_parallelism_threads=1
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        #self.persistent_sess = tf.Session(config=tf.ConfigProto(
        #    inter_op_parallelism_threads=1,
        #    intra_op_parallelism_threads=1
        #))
        self.persistent_sess = tf.Session(config=config)
        self.model_path = model_path
        self.verbose = verbose
        self.adnet = ADNetwork(self.learning_rate_placeholder)
        self.adnet.create_network(self.tensor_input, self.tensor_lb_action, self.tensor_lb_class, self.tensor_action_history, self.tensor_is_training, self.reward)

        model_path = os.path.join(self.model_path, 'model.ckpt')

        load_baseline = False
        if load_baseline:
            print('Loading baseline model')
            if 'ADNET_MODEL_PATH' in os.environ.keys():
                self.adnet.read_original_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
            else:
                self.adnet.read_original_weights(self.persistent_sess)
        elif 1:#os.path.exists(model_path):
            saver = tf.train.Saver()
            saver.restore(self.persistent_sess, model_path)
            print('Loading model from: ', model_path)
        else:
            print("Traning Model from scratch")
            #read_vgg_weights
            if 'ADNET_MODEL_PATH' in os.environ.keys():
                self.adnet.read_vgg_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
            else:
                self.adnet.read_vgg_weights(self.persistent_sess)

        self.action_histories = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.action_histories_old = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.histories = []
        self.iteration = 0
        self.imgwh = None

        self.callback_redetection = self.redetection_by_sampling
        self.failed_cnt = 0
        self.latest_score = 0

        self.stopwatch = StopWatchManager()
        self.loss_logger = open(os.path.join(self.model_path, 'loss.log'), 'a+')

    def get_image_path(self, vid_path, idx):
        im_path = os.path.join(vid_path, 'img', '%04d.jpg' % (idx + 1))
        if os.path.exists(im_path):
            return im_path
        return os.path.join(vid_path, 'img', '%08d.jpg' % (idx + 1))

    def by_dataset(self, vid_path='./data/freeman1/'):
        assert os.path.exists(vid_path)

        gt_boxes = BoundingBox.read_vid_gt(vid_path)

        curr_bbox = None
        self.stopwatch.start('total')
        _logger.info('---- start dataset l=%d' % (len(gt_boxes)))
        for idx, gt_box in enumerate(gt_boxes):
            im_path = self.get_image_path(vid_path, idx)
            print('im_path: {}'.format(im_path))
            img = commons.imread(im_path)
            self.imgwh = Coordinate.get_imgwh(img)
            if idx == 0:
                # initialization : initial fine-tuning
                self.initial_finetune(img, gt_box)
                curr_bbox = gt_box

            # tracking
            predicted_box = self.tracking(img, curr_bbox)
            self.show(img, gt_box=gt_box, predicted_box=predicted_box)
            # cv2.imwrite('/Users/ildoonet/Downloads/aaa/%d.jpg' % self.iteration, img)
            curr_bbox = predicted_box
        self.stopwatch.stop('total')

        _logger.info('----')
        _logger.info(self.stopwatch)
        _logger.info('%.3f FPS' % (len(gt_boxes) / self.stopwatch.get_elapsed('total')))

    def train(self, vid_path='./data/freeman1/'):
        assert os.path.exists(vid_path)

        gt_boxes = BoundingBox.read_vid_gt(vid_path)

        curr_bbox = None
        self.stopwatch.start('total')
        _logger.info('---- start dataset l=%d' % (len(gt_boxes)))

        gt_box_tuples = []
        for idx, gt_box in enumerate(gt_boxes):
            gt_box_tuples.append((idx, gt_box))

        random.shuffle(gt_box_tuples)
        num_frames = len(gt_box_tuples)

        for i in range(num_frames):
            idx = gt_box_tuples[i][0]
            gt_box = gt_box_tuples[i][1]
            im_path = self.get_image_path(vid_path, idx)
            #print('{}/{}: im_path: {}'.format(i, num_frames, im_path))
            img = commons.imread(im_path)
            self.imgwh = Coordinate.get_imgwh(img)
            # if idx == 0:
            # initialization : initial fine-tuning
            self.finetune_all(img, gt_box)
            # curr_bbox = gt_box

            # tracking
            # predicted_box = self.tracking(img, curr_bbox)
            # self.show(img, gt_box=gt_box, predicted_box=predicted_box)
            # cv2.imwrite('/Users/ildoonet/Downloads/aaa/%d.jpg' % self.iteration, img)
            # curr_bbox = predicted_box
        self.stopwatch.stop('total')

        self.save_model()

        _logger.info('----')
        _logger.info(self.stopwatch)
        _logger.info('%.3f FPS' % (len(gt_boxes) / self.stopwatch.get_elapsed('total')))

    def show(self, img, delay=1, predicted_box=None, gt_box=None):
        if isinstance(img, str):
            img = commons.imread(img)

        if gt_box is not None:
            gt_box.draw(img, BoundingBox.COLOR_GT)
        if predicted_box is not None:
            predicted_box.draw(img, BoundingBox.COLOR_PREDICT)

        cv2.imshow('result', img)
        cv2.waitKey(delay)

    def _get_features(self, samples):
        feats = []
        for batch in commons.chunker(samples, ADNetRunner.MAX_BATCHSIZE):
            feats_batch = self.persistent_sess.run(self.adnet.layer_feat, feed_dict={
                self.adnet.input_tensor: batch
            })
            feats.extend(feats_batch)
        return feats

    def initial_finetune(self, img, detection_box):
        self.stopwatch.start('initial_finetune')
        t = time.time()

        # generate samples
        pos_num, neg_num = ADNetConf.g()['initial_finetune']['pos_num'], ADNetConf.g()['initial_finetune']['neg_num']
        pos_boxes, neg_boxes = detection_box.get_posneg_samples(self.imgwh, pos_num, neg_num, use_whole=True)
        pos_lb_action = BoundingBox.get_action_labels(pos_boxes, detection_box)

        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
        for box, feat in zip(pos_boxes, feats):
            box.feat = feat
        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
        for box, feat in zip(neg_boxes, feats):
            box.feat = feat

        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['initial_finetune']['learning_rate'],
            ADNetConf.get()['initial_finetune']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        #_logger.info('ADNetRunner.initial_finetune t=%.3f' % t)
        self.stopwatch.stop('initial_finetune')

    def finetune_all(self, img, detection_box):
        self.stopwatch.start('finetune_all')
        t = time.time()

        # generate samples
        pos_num, neg_num = ADNetConf.g()['sl_train']['pos_num'], ADNetConf.g()['sl_train']['neg_num']


        #print('img.shape: ', img.shape)
        #print('gt_box: {}, {}, {}, {}'.format(detection_box.xy.x, detection_box.xy.y, detection_box.wh.x, detection_box.wh.y))
        pos_boxes, neg_boxes = detection_box.get_posneg_samples(self.imgwh, pos_num, neg_num, use_whole=True)
        pos_lb_action = BoundingBox.get_action_labels(pos_boxes, detection_box)

        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
        for box, feat in zip(pos_boxes, feats):
            box.feat = feat
        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
        for box, feat in zip(neg_boxes, feats):
            box.feat = feat

        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['sl_train']['learning_rate'],
            ADNetConf.get()['sl_train']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        #_logger.info('ADNetRunner.finetune_all t=%.3f' % t)
        self.stopwatch.stop('finetune_all')


    def _finetune_fc(self, img, pos_boxes, neg_boxes, pos_lb_action, learning_rate, iter, iter_score=1):
        BATCHSIZE = ADNetConf.g()['minibatch_size']

        def get_img(idx, posneg):
            if isinstance(img, tuple):
                return img[posneg][idx]
            return img

        pos_samples = [commons.extract_region(get_img(i, 0), box) for i, box in enumerate(pos_boxes)]
        neg_samples = [commons.extract_region(get_img(i, 1), box) for i, box in enumerate(neg_boxes)]
        # pos_feats, neg_feats = self._get_features(pos_samples), self._get_features(neg_samples)

        if self.verbose:
            commons.imshow_grid('pos', pos_samples[-50:], 10, 5)
            commons.imshow_grid('neg', neg_samples[-50:], 10, 5)
            cv2.waitKey(1)

        num_cls = 0
        num_actions = 0
        loss_actions = 0.0
        loss_cls = 0.0
        for i in range(iter):
            batch_idxs = commons.random_idxs(len(pos_boxes), BATCHSIZE)
            batch_feats = [x.feat for x in commons.choices_by_idx(pos_boxes, batch_idxs)]
            batch_lb_action = commons.choices_by_idx(pos_lb_action, batch_idxs)
            grad_action, loss_actions_ = self.persistent_sess.run(
                [self.adnet.weighted_grads_op1, self.adnet.loss_actions],
                feed_dict={
                    self.adnet.layer_feat: batch_feats,
                    self.adnet.label_tensor: batch_lb_action,
                    self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 1, 1, 110)),
                    self.learning_rate_placeholder: learning_rate,
                    self.tensor_is_training: True
                }
            )
            loss_actions += sum(loss_actions_)/BATCHSIZE
            num_actions += 1

            if i % iter_score == 0:
                # training score auxiliary(fc2)
                # -- hard score example mining
                scores = []
                for batch_neg in commons.chunker([x.feat for x in neg_boxes], ADNetRunner.MAX_BATCHSIZE):
                    scores_batch = self.persistent_sess.run(
                        self.adnet.layer_scores,
                        feed_dict={
                            self.adnet.layer_feat: batch_neg,
                            self.adnet.action_history_tensor: np.zeros(shape=(len(batch_neg), 1, 1, 110)),
                            self.learning_rate_placeholder: learning_rate,
                            self.tensor_is_training: False
                        }
                    )
                    scores.extend(scores_batch)
                desc_order_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x:x[1][1])]

                # -- train
                batch_feats_neg = [x.feat for x in commons.choices_by_idx(neg_boxes, desc_order_idx[:BATCHSIZE])]
                a = batch_feats + batch_feats_neg
                grads, loss_cls_ = self.persistent_sess.run(
                    [self.adnet.weighted_grads_op2, self.adnet.loss_cls],
                    feed_dict={
                        self.adnet.layer_feat: batch_feats + batch_feats_neg,
                        self.adnet.class_tensor: [1]*len(batch_feats) + [0]*len(batch_feats_neg),
                        self.adnet.action_history_tensor: np.zeros(shape=(len(batch_feats)+len(batch_feats_neg), 1, 1, 110)),
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True
                    }
                )
                loss_cls += sum(loss_cls_)/BATCHSIZE
                num_cls += 1
        s = "loss_actions: {}, Loss cls: {}".format(loss_actions/num_actions, loss_cls/num_cls)
        #print(s)
        self.loss_logger.write(s + '\n')
        self.loss_logger.flush()

    def train_rl_tracking(self, vid_path='./data/freeman1/'):
        ## TODO: Reset action history for each video
        assert os.path.exists(vid_path)

        gt_boxes = BoundingBox.read_vid_gt(vid_path)

        curr_bbox = None
        self.stopwatch.start('total')
        _logger.info('---- start dataset l=%d' % (len(gt_boxes)))

        gt_box_tuples = []
        for idx, gt_box in enumerate(gt_boxes):
            gt_box_tuples.append((idx, gt_box))

        # random.shuffle(gt_box_tuples)
        num_frames = len(gt_box_tuples)
        num_frame_step = ADNetConf.get()['rl_episode']['frame_steps']
        startFrames_ = list(range(num_frames - num_frame_step))  ## check if number of frames are sufficient
        endFrames_ = [i + num_frame_step for i in startFrames_]
        numOfClips = min(ADNetConf.get()['rl_episode']['num_frames_per_video'], len(startFrames_))
        randomIndex = list(range(len(startFrames_)))
        random.shuffle(randomIndex)
        startFrames = [startFrames_[i] for i in randomIndex]
        endFrames = [endFrames_[i] for i in randomIndex] 
        '''
        onehots_pos = []
        onehots_neg = []
        imgs_pos = []
        imgs_neg = []
        action_labels_pos = []
        action_labels_neg = []
        '''
        onehots = []
        imgs = []
        action_labels = []
        rewards = []
        for i in range(numOfClips):
            img_boxes = []
            action_labels_ = []
            one_hots = []
            reward_episode = []
            for frame_num in range(startFrames[i], endFrames[i] + 1):  # for each frame in the seq of clips
                img_index = gt_box_tuples[frame_num][0]
                im_path = self.get_image_path(vid_path, img_index)
                img = commons.imread(im_path)
                self.imgwh = Coordinate.get_imgwh(img)
                curr_gt_bbox = gt_box_tuples[frame_num][1]
                ##TODO: if black n white photo append to create 3 channel image
                #print('frame num:', frame_num)
                curr_bbox, boxes, actions_seq, onehot_seq, reward_ = self.tracking4training(img, curr_gt_bbox)
                print('reward: ', reward_)
                if len(actions_seq) != len(boxes):
                    print('+++++++++++++++++++++++++++++Mismatch in size: action_seq lentgh: {}, boxes length: {}'.format(len(actions_seq), len(boxes)))

                img_boxes.extend([(img_index, box) for box in boxes])
                #print('type: onehot_seq', type(onehot_seq))
                #print('len: onehot_seq', len(onehot_seq))
                #print('len boxes: ', len(boxes))
                one_hots.extend(onehot_seq)
                action_labels_.extend(actions_seq)
                reward_episode.extend(reward_)

            '''
            if curr_gt_bbox.iou(curr_bbox)>0.7:
                ##append to pos data
                if len(action_labels) != len(img_boxes):
                    print('====================Mismatch in size: action_labels lentgh: {}, img_boxes length: {}'.format(len(action_labels), len(img_boxes)))
                onehots_pos.extend(one_hots)
                imgs_pos.extend(img_boxes)
                action_labels_pos.extend(action_labels)

            else:
                ## append to neg data
                onehots_neg.extend(one_hots)
                #print('img_boxes: ', img_boxes)
                imgs_neg.extend(img_boxes)
                action_labels_neg.extend(action_labels)
            ''' 
            assert(len(action_labels_) == len(img_boxes))
            assert(len(action_labels_) == len(one_hots))
            assert(len(action_labels_) == len(reward_episode))
            onehots.extend(one_hots)
            #print('img_boxes: ', img_boxes)
            imgs.extend(img_boxes)
            action_labels.extend(action_labels_)
            rewards.extend(reward_episode)          

        ## Policy Gradient Training
        #num_pos = len(action_labels_pos)
        #num_neg = len(action_labels_neg)
        
        num = len(rewards)
        num_pos = len(list(filter(lambda x: (x > 0), rewards)))
        print("Number of positive action: {} out of total: {} actions".format(num_pos, num))
        batch_size = ADNetConf.get()['rl_episode']['batch_size']
        indices = list(range(num))
        random.shuffle(indices)
        ## training
        s = ''
        num_batches = int(num/batch_size)
        #print('len action_labels: ', len(action_labels))
        for batch_idx in range(num_batches):
            if num!=[]:
                examples = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]
                #print(examples)
                imgs_patches = []
                for i, ex_index in enumerate(examples):
                    #print(neg_ex_index)
                    img = commons.imread(self.get_image_path(vid_path, imgs[ex_index][0]))
                    imgs_patches.append(commons.extract_region(img, imgs[ex_index][1]))

                #print("type imgs_patches: ", type(imgs_patches))
                imgs_patches_feat = self._get_features(imgs_patches)
                action_labels_ = [action_labels[i] for i in examples]
                action_histories = [np.reshape(onehots[i], (-1, 1, 110)) for i in examples]
                reward = [rewards[i] for i in examples]
                if False:
                    print('type imgs_patches_feat: ', type(imgs_patches_feat))
                    print('type action_labels: ', type(action_labels))
                    print('type reward: ', type(reward))
                    print('type action_histories: ', type(action_histories))

                    print('size imgs_patches_feat: ', len(imgs_patches_feat))
                    print('size action_labels: ', len(action_labels))
                    print('size reward: ', len(reward))
                    print('size action_histories: ', len(action_histories))


                    print('type imgs_patches_feat[0]: ', type(imgs_patches_feat[0]))
                    print('type action_labels[0]: ', type(action_labels[0]))
                    print('type reward[0]: ', type(reward[0]))
                    print('type action_histories[0]: ', type(action_histories[0]))

                    print('type imgs_patches_feat[0]: ', imgs_patches_feat[0].shape)
                    print('action_labels[0]: ', action_labels[0])
                    print('reward[0]: ', reward[0])
                    print('action_histories[0]: ', action_histories[0])
                    print('shape action_histories[0]: ', action_histories[0].shape)


                grad_rl, loss_ = self.persistent_sess.run(
                    [self.adnet.weighted_grads_rl, self.adnet.loss_rl],
                    feed_dict={
                        self.adnet.layer_feat: imgs_patches_feat,
                        self.adnet.label_tensor: action_labels_,
                        self.adnet.reward: reward,
                        self.adnet.action_history_tensor: action_histories,  ## TODO reshape to np.zeros(shape=(BATCHSIZE, 1, 1, 110))
                        self.learning_rate_placeholder: ADNetConf.get()['rl_episode']['lr'],
                        self.tensor_is_training: True
                    }
                )
                loss = sum(loss_)/batch_size
                s = 'loss:{}\n'.format(loss)
                print("Loss: ", loss)
                self.loss_logger.write(s)
                self.loss_logger.flush()
        '''
        #print('len imgs_pos: ', len(imgs_pos))
        print('num_pos: ', num_pos)
        #print('len imgs_neg: ', len(imgs_neg))
        print('num_neg: ', num_neg)
        train_pos_cnt = 0
        train_pos = []
        train_neg_cnt = 0
        train_neg = []
        if num_pos>batch_size/2:
            remain = batch_size*numOfClips
            while(remain>0):
                if train_pos_cnt==0:
                    train_pos_list = list(range(num_pos))
                    #print('Lenght train_pos_list: ', len(train_pos_list))
                    random.shuffle(train_pos_list)

                train_pos.extend(train_pos_list[train_pos_cnt:min(len(train_pos_list), train_pos_cnt + remain)])
                train_pos_cnt = min(len(train_pos_list), train_pos_cnt + remain)
                train_pos_cnt = train_pos_cnt%len(train_pos_list)
                remain = batch_size*numOfClips - len(train_pos)

        if num_neg>batch_size/2:
            remain = batch_size*numOfClips
            while(remain>0):
                if train_neg_cnt==0:
                    train_neg_list = list(range(num_neg))
                    random.shuffle(train_neg_list)

                train_neg.extend(train_neg_list[train_neg_cnt:min(len(train_neg_list), train_neg_cnt + remain)])
                train_neg_cnt = min(len(train_neg_list), train_neg_cnt + remain)
                train_neg_cnt = train_neg_cnt%len(train_neg_list)
                remain = batch_size*numOfClips - len(train_neg)


        ## training
        print('train_pos shape: ', len(train_pos))
        #print('train_pos[0]: ', train_pos[0])
        print('train_neg shape: ', len(train_neg))
        print("numOfClips: ", numOfClips)
        s = ''
        for batch_idx in range(numOfClips):
            if train_pos!=[]:
                pos_examples = train_pos[batch_idx*batch_size:(batch_idx+1)*batch_size]
                imgs_patches = []
                for i, pos_ex_index in enumerate(pos_examples):
                    #print("len imgs_pos: ", len(imgs_pos))
                    #print("imgs_pos[0]: ", imgs_pos[0])
                    #print('pos_ex_index: ', pos_ex_index)
                    #print('imgs_pos[pos_ex_index]', imgs_pos[pos_ex_index])
                    img = commons.imread(self.get_image_path(vid_path, imgs_pos[pos_ex_index][0]))
                    imgs_patches.append(commons.extract_region(img, imgs_pos[pos_ex_index][1]))

                imgs_patches_feat = self._get_features(imgs_patches)
                action_labels = [action_labels_pos[i] for i in pos_examples]
                action_histories = [np.reshape(onehots_pos[i], (-1, 1, 110)) for i in pos_examples]
                reward = [1]*len(action_labels)
                grad_rl, loss = self.persistent_sess.run(
                    [self.adnet.weighted_grads_rl, self.adnet.loss_rl],
                    feed_dict={
                        self.adnet.layer_feat: imgs_patches_feat,
                        self.adnet.label_tensor: action_labels,
                        self.adnet.reward: reward,
                        self.adnet.action_history_tensor: action_histories,  ## TODO reshape to np.zeros(shape=(BATCHSIZE, 1, 1, 110))
                        self.learning_rate_placeholder: ADNetConf.get()['rl_episode']['lr'],
                        self.tensor_is_training: True
                    }
                )
                pos_loss = sum(loss)/batch_size
                s +=  'pos_loss:{}'.format(pos_loss)
                #print("Loss Pos: ", sum(loss)/batch_size)

            if train_neg!=[]:
                neg_examples = train_neg[batch_idx*batch_size:(batch_idx+1)*batch_size]
                imgs_patches = []
                for i, neg_ex_index in enumerate(neg_examples):
                    #print(neg_ex_index)
                    img = commons.imread(self.get_image_path(vid_path, imgs_neg[neg_ex_index][0]))
                    imgs_patches.append(commons.extract_region(img, imgs_neg[neg_ex_index][1]))

                #print("type imgs_patches: ", type(imgs_patches))
                imgs_patches_feat = self._get_features(imgs_patches)
                action_labels = [action_labels_neg[i] for i in neg_examples]
                action_histories = [np.reshape(onehots_neg[i], (-1, 1, 110)) for i in neg_examples]
                reward = [-1]*len(action_labels)
                if False:
                    print('type imgs_patches_feat: ', type(imgs_patches_feat))
                    print('type action_labels: ', type(action_labels))
                    print('type reward: ', type(reward))
                    print('type action_histories: ', type(action_histories))

                    print('size imgs_patches_feat: ', len(imgs_patches_feat))
                    print('size action_labels: ', len(action_labels))
                    print('size reward: ', len(reward))
                    print('size action_histories: ', len(action_histories))


                    print('type imgs_patches_feat[0]: ', type(imgs_patches_feat[0]))
                    print('type action_labels[0]: ', type(action_labels[0]))
                    print('type reward[0]: ', type(reward[0]))
                    print('type action_histories[0]: ', type(action_histories[0]))

                    print('type imgs_patches_feat[0]: ', imgs_patches_feat[0].shape)
                    print('action_labels[0]: ', action_labels[0])
                    print('reward[0]: ', reward[0])
                    print('action_histories[0]: ', action_histories[0])
                    print('shape action_histories[0]: ', action_histories[0].shape)


                grad_rl, loss = self.persistent_sess.run(
                    [self.adnet.weighted_grads_rl, self.adnet.loss_rl],
                    feed_dict={
                        self.adnet.layer_feat: imgs_patches_feat,
                        self.adnet.label_tensor: action_labels,
                        self.adnet.reward: reward,
                        self.adnet.action_history_tensor: action_histories,  ## TODO reshape to np.zeros(shape=(BATCHSIZE, 1, 1, 110))
                        self.learning_rate_placeholder: ADNetConf.get()['rl_episode']['lr'],
                        self.tensor_is_training: True
                    }
                )
                neg_loss = sum(loss)/batch_size
                s += ',neg_loss:{}\n'.format(neg_loss)
                #print("Loss Neg: ", neg_loss)
            else:
                s += '\n'
        '''
        self.save_model()


    def get_reward(self, action_idx, gt, prev_bbox, curr_bbox):
        if action_idx == ADNetwork.ACTION_IDX_STOP:
            if gt.iou(prev_bbox) > 0.7:
                return 1
            else:
                return -1
        else:
            #a = gt.iou(prev_bbox) - gt.iou(curr_bbox)
            return ADNetConf.get()['rl_episode']['lambda'] * (gt.iou(curr_bbox) - gt.iou(prev_bbox))

    '''
    def discount_rewards(self, r, normal):
        """ take a list of size max_dec_step * (batch_size, k) and return a list of the same size """
        if len(r) < 2:
            return r
        discounted_r = np.zeros_like(r)
        running_add = 0.0
        for t in reversed(range(0, len(r))):
            running_add = running_add * ADNetConf.get()['rl_episode']['gamma'] + r[t] # rd_t = r_t + gamma * r_{t+1}
            discounted_r[i] = running_add
        
        if normal:
            mean = np.mean(discounted_r)
            std = np.std(discounted_r)
            discounted_r = (discounted_r - mean)/(std)
        return discounted_r.tolist()
    '''
    def tracking4training(self, img, curr_bbox):
        self.iteration += 1
        gt = curr_bbox
        prev_bbox = curr_bbox
        is_tracked = True
        boxes = []
        actions_seq = []
        rewards = []
        num_action = ADNetConf.get()['rl_episode']['num_action']
        num_action_history = ADNetConf.get()['rl_episode']['num_action_history']
        num_action_step_max = ADNetConf.get()['rl_episode']['num_action_step_max']

        onehot_seq = []#np.zeros((num_action * num_action_history, num_action_step_max))
        self.stopwatch.start('tracking4training.do_action')
        track_i = 1
        prev_score = -math.inf
        while track_i <=  num_action_step_max:
            #print('track_i: ', track_i)
            patch = commons.extract_region(img, curr_bbox)
            f  = False
            # forward with image & action history
            print('a: ', patch.shape)
            print('b: ', commons.onehot_flatten(self.action_histories).shape)
            print('c: ', )
            
            actions, classes = self.persistent_sess.run(
                [self.adnet.layer_actions, self.adnet.layer_scores],
                feed_dict={
                    self.adnet.input_tensor: [patch],
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories)],
                    self.tensor_is_training: f
                }
            )

            curr_score = classes[0][1]
            '''
            if curr_score < ADNetConf.g()['rl_episode']['thresh_fail'] and curr_score < prev_score:
                if random.uniform(0, 1) < 0.5:
                    action_idx = random.randint(0,9)
            '''
            #print('confidence: ', curr_score)
            if curr_score < ADNetConf.g()['rl_episode']['thresh_fail']:
                is_tracked = False
                #self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
                #print('1========')
                break
            #print(actions)
            #action_idx = np.argmax(actions[0])
            # Sample from the policy instead of greedily taking the max probability actions
            probs = actions[0]
            print('probs: ', probs)
            action_idx = np.random.choice(np.arange(len(probs)), p=probs) 
            curr_bbox_temp = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)

            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    #print(prev_bbox)
                    #print(curr_bbox)
                    print('box not moved.')
                    is_tracked = False
                    #print('2========')
                    break
            prev_box = curr_bbox_temp
            # move box
            actions_seq.append(action_idx)
            onehot_seq.append(self.action_history2onehot())
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            
            
            # oscillation check
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP
            boxes.append(curr_bbox)

            rewards.append(self.get_reward(action_idx, gt, prev_bbox, curr_bbox))
            if action_idx == ADNetwork.ACTION_IDX_STOP:
                #print('3========')
                break

            track_i += 1
            prev_score = curr_score


        #onehot_seq = onehot_seq[:, :track_i]
        self.stopwatch.stop('tracking4training.do_action')

        # redetection when tracking failed
        new_score = 0.0
        if not is_tracked:
            self.failed_cnt += 1
            # run redetection callback function
            new_box, new_score = self.callback_redetection(curr_bbox, img)
            if new_box is not None:
                curr_bbox = new_box
                patch = commons.extract_region(img, curr_bbox)
                action_idx = BoundingBox.get_action_label(gt, curr_bbox)
                boxes.append(curr_bbox)
                actions_seq.append(action_idx)
                onehot_seq.append(self.action_history2onehot())
                rewards.append(self.get_reward(action_idx, gt, prev_bbox, curr_bbox))
                self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            _logger.debug('redetection success=%s' % (str(new_box is not None)))

        '''
        # save samples
        if is_tracked or new_score > ADNetConf.g()['predict']['thresh_success']:
            self.stopwatch.start('tracking.save_samples.roi')
            imgwh = Coordinate.get_imgwh(img)
            pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']
            pos_boxes, neg_boxes = curr_bbox.get_posneg_samples(
                imgwh, pos_num, neg_num, use_whole=False,
                pos_thresh=ADNetConf.g()['finetune']['pos_thresh'],
                neg_thresh=ADNetConf.g()['finetune']['neg_thresh'],
                uniform_translation_f=2,
                uniform_scale_f=5
            )
            self.stopwatch.stop('tracking.save_samples.roi')
            self.stopwatch.start('tracking.save_samples.feat')
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
            for box, feat in zip(pos_boxes, feats):
                box.feat = feat
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
            for box, feat in zip(neg_boxes, feats):
                box.feat = feat
            pos_lb_action = BoundingBox.get_action_labels(pos_boxes, curr_bbox)
            self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))

            # clear old ones
            self.histories = self.histories[-ADNetConf.g()['finetune']['long_term']:]
            self.stopwatch.stop('tracking.save_samples.feat')

        # online finetune
        if self.iteration % ADNetConf.g()['finetune']['interval'] == 0 or is_tracked is False:
            img_pos, img_neg = [], []
            pos_boxes, neg_boxes, pos_lb_action = [], [], []
            pos_term = 'long_term' if is_tracked else 'short_term'
            for i in range(ADNetConf.g()['finetune'][pos_term]):
                if i >= len(self.histories):
                    break
                pos_boxes.extend(self.histories[-(i+1)][0])
                pos_lb_action.extend(self.histories[-(i+1)][2])
                img_pos.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][0]))
            for i in range(ADNetConf.g()['finetune']['short_term']):
                if i >= len(self.histories):
                    break
                neg_boxes.extend(self.histories[-(i+1)][1])
                img_neg.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][1]))
            self.stopwatch.start('tracking.online_finetune')
            self._finetune_fc(
                (img_pos, img_neg), pos_boxes, neg_boxes, pos_lb_action,
                ADNetConf.get()['finetune']['learning_rate'],
                ADNetConf.get()['finetune']['iter']
            )
            #_logger.debug('finetuned')
            self.stopwatch.stop('tracking.online_finetune')
        '''## Online Finetuning
        if self.verbose:
            cv2.imshow('patch', patch)
        return curr_bbox, boxes, actions_seq, onehot_seq, rewards


    def action_history2onehot(self):
        row = ADNetConf.get()['rl_episode']['num_action']
        col = ADNetConf.get()['rl_episode']['num_action_history']
        arr = np.zeros((row,col))
        for index, action_idx in enumerate(self.action_histories):
            arr[action_idx, index] = 1

        return arr.flatten()


    def tracking(self, img, curr_bbox):
        self.iteration += 1
        is_tracked = True
        boxes = []
        self.latest_score = -1
        self.stopwatch.start('tracking.do_action')
        for track_i in range(ADNetConf.get()['predict']['num_action']):
            patch = commons.extract_region(img, curr_bbox)

            # forward with image & action history
            actions, classes = self.persistent_sess.run(
                [self.adnet.layer_actions, self.adnet.layer_scores],
                feed_dict={
                    self.adnet.input_tensor: [patch],
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories)],
                    self.tensor_is_training: False
                }
            )

            latest_score = classes[0][1]
            if latest_score < ADNetConf.g()['predict']['thresh_fail']:
                is_tracked = False
                self.action_histories_old = np.copy(self.action_histories)
                self.action_histories = np.insert(self.action_histories, 0, 12)[:-1]
                break
            else:
                self.failed_cnt = 0
            self.latest_score = latest_score

            # move box
            action_idx = np.argmax(actions[0])
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            prev_bbox = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)
            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    print(prev_bbox)
                    print(curr_bbox)
                    print('box not moved.')
                    #raise Exception('box not moved.')
                    continue

            # oscillation check
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP

            if action_idx == ADNetwork.ACTION_IDX_STOP:
                break

            boxes.append(curr_bbox)
        self.stopwatch.stop('tracking.do_action')

        # redetection when tracking failed
        new_score = 0.0
        if not is_tracked:
            self.failed_cnt += 1
            # run redetection callback function
            new_box, new_score = self.callback_redetection(curr_bbox, img)
            if new_box is not None:
                curr_bbox = new_box
                patch = commons.extract_region(img, curr_bbox)
            _logger.debug('redetection success=%s' % (str(new_box is not None)))

        # save samples
        if is_tracked or new_score > ADNetConf.g()['predict']['thresh_success']:
            self.stopwatch.start('tracking.save_samples.roi')
            imgwh = Coordinate.get_imgwh(img)
            pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']
            pos_boxes, neg_boxes = curr_bbox.get_posneg_samples(
                imgwh, pos_num, neg_num, use_whole=False,
                pos_thresh=ADNetConf.g()['finetune']['pos_thresh'],
                neg_thresh=ADNetConf.g()['finetune']['neg_thresh'],
                uniform_translation_f=2,
                uniform_scale_f=5
            )
            self.stopwatch.stop('tracking.save_samples.roi')
            self.stopwatch.start('tracking.save_samples.feat')
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
            for box, feat in zip(pos_boxes, feats):
                box.feat = feat
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
            for box, feat in zip(neg_boxes, feats):
                box.feat = feat
            pos_lb_action = BoundingBox.get_action_labels(pos_boxes, curr_bbox)
            self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))

            # clear old ones
            self.histories = self.histories[-ADNetConf.g()['finetune']['long_term']:]
            self.stopwatch.stop('tracking.save_samples.feat')

        # online finetune
        if self.iteration % ADNetConf.g()['finetune']['interval'] == 0 or is_tracked is False:
            img_pos, img_neg = [], []
            pos_boxes, neg_boxes, pos_lb_action = [], [], []
            pos_term = 'long_term' if is_tracked else 'short_term'
            for i in range(ADNetConf.g()['finetune'][pos_term]):
                if i >= len(self.histories):
                    break
                pos_boxes.extend(self.histories[-(i+1)][0])
                pos_lb_action.extend(self.histories[-(i+1)][2])
                img_pos.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][0]))
            for i in range(ADNetConf.g()['finetune']['short_term']):
                if i >= len(self.histories):
                    break
                neg_boxes.extend(self.histories[-(i+1)][1])
                img_neg.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][1]))
            self.stopwatch.start('tracking.online_finetune')
            self._finetune_fc(
                (img_pos, img_neg), pos_boxes, neg_boxes, pos_lb_action,
                ADNetConf.get()['finetune']['learning_rate'],
                ADNetConf.get()['finetune']['iter']
            )
            #_logger.debug('finetuned')
            self.stopwatch.stop('tracking.online_finetune')

        cv2.imshow('patch', patch)
        return curr_bbox

    def save_model(self):
        saver = tf.train.Saver()
        model_path = os.path.join(self.model_path, 'model.ckpt')
        if os.path.exists(self.model_path):
            save_path = saver.save(self.persistent_sess, model_path)
            print("Saving model at: ", save_path)
        else:
            print('model path does not exist: ', model_path)

    def load_model(self):
        saver = tf.train.Saver()
        model_path = os.path.join(self.model_path, 'model.ckpt')
        if os.path.exists(self.model_path):
            saver.restore(self.persistent_sess, model_path)
            print("Loading model at: ", model_path)
        else:
            print('model path does not exist: ', model_path)


    def redetection_by_sampling(self, prev_box, img):
        """
        default redetection method
        """
        imgwh = Coordinate.get_imgwh(img)
        translation_f = min(1.5, 0.6 * 1.15**self.failed_cnt)
        candidates = prev_box.gen_noise_samples(imgwh, 'gaussian', ADNetConf.g()['redetection']['samples'],
                                                gaussian_translation_f=translation_f)

        scores = []
        for c_batch in commons.chunker(candidates, ADNetRunner.MAX_BATCHSIZE):
            samples = [commons.extract_region(img, box) for box in c_batch]
            classes = self.persistent_sess.run(
                self.adnet.layer_scores,
                feed_dict={
                    self.adnet.input_tensor: samples,
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories_old)]*len(c_batch),
                    self.tensor_is_training: False
                }
            )
            scores.extend([x[1] for x in classes])
        top5_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])][:5]
        mean_score = sum([scores[x] for x in top5_idx]) / 5.0
        if mean_score >= self.latest_score:
            mean_box = candidates[0]
            for i in range(1, 5):
                mean_box += candidates[i]
            return mean_box / 5.0, mean_score
        return None, 0.0

    def __del__(self):
        self.persistent_sess.close()
        self.loss_logger.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-vid_path", type=str, default='./data/freeman1')
    parser.add_argument("-model_path", type=str, default='./checkpoint/temp')
    parser.add_argument("-debug", type=str2bool, nargs= '?', default=True)
    parser.add_argument("-rl", type=str2bool, nargs= '?', default=False)
    parser.add_argument("-mode", type=str, default='test')
    parser.add_argument("-skip", type=int, default=0)
    parser.add_argument("-verbose", type=str2bool, nargs='?', default=False)
    args = parser.parse_args()

    current_dir = '.'

    conf_dir = os.path.join(current_dir, 'conf')
    data_dir = os.path.join(current_dir, 'data')
    ADNetConf.get(os.path.join(conf_dir, 'repo.yaml'))

    random.seed(1258)
    np.random.seed(1258)
    tf.set_random_seed(1258)

    #vid_path= os.path.join(data_dir, 'bicycle')
    #vid_path= os.path.join(data_dir, 'bicycle')
    #fire.Fire(ADNetRunner)

    vid_path = args.vid_path
    model_path = args.model_path
    verbose = args.verbose
    mode = args.mode
    debug = args.debug
    train_rl = args.rl
    model = ADNetRunner(model_path=model_path, verbose=verbose)

    if debug:
        if mode == 'train':
            if train_rl:
                print('Training RL with debug mode')
                for i in range(50):
                    print('Training epoch: ', i)
                    model.train_rl_tracking(vid_path=vid_path)
            else:
                print('Training SL with debug mode')
                model.train(vid_path=vid_path)
        else:
            print('Testing with debug mode')
            model.by_dataset(vid_path=vid_path)
        exit(1)

    if mode == 'train':
        folders = ['vot2013', 'vot2014', 'vot2015']
        all_paths = []
        for folder in folders:
            train_dir = os.path.join('train_data', folder)
            for subdir, dirs, files in os.walk(train_dir, topdown=True):
                for name in dirs:
                    if name[-3:] != 'img':
                        #print(os.path.join(subdir, name))
                        all_paths.append(os.path.join(subdir, name))

        num_videos = len(all_paths)
        print("Number folders: ", num_videos)

        t1 = time.time()
        cnt = 0
        num_files_done = args.skip

        num_epochs = 5

        for i, folder in enumerate(all_paths, 1):
            print(i, folder)
        #exit(1)
        for epoch in range(1, num_epochs + 1):
            print("Training Epoch: {}/{}".format(epoch, num_epochs))
            for i, folder in enumerate(all_paths, 1):
                cnt += 1
                #if 'gym' in folder:
                #    continue
                if cnt < num_files_done + 1:
                    continue
                print("epoch: {}, video num: {}/{}, {}".format(epoch, cnt, num_videos, folder))
                if train_rl:
                    model.train_rl_tracking(vid_path=folder)
                else:
                    model.train(vid_path=folder)

                with open('log.txt', 'w+') as f:
                    f.write('writting file epch: {}, folder num: {}, folder: {}'.format(str(epoch), i , folder))
                    f.flush()

        print("Total Time Taken in Training: ", time.time() - t1)
    else:
        print("=====================Running Inference==================")
        model.by_dataset(vid_path=vid_path)
