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
        self.adnet.create_network(self.tensor_input, self.tensor_lb_action, self.tensor_lb_class, self.tensor_action_history, self.tensor_is_training)

        model_path = os.path.join(self.model_path, 'model.ckpt')

        #load_baseline = True
        if load_baseline:
            print('Loading baseline model')
            if 'ADNET_MODEL_PATH' in os.environ.keys():
                self.adnet.read_original_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
            else:
                self.adnet.read_original_weights(self.persistent_sess)
        elif 0:#os.path.exists(model_path):
            saver = tf.train.Saver()
            saver.restore(self.persistent_sess, model_path)
            print('Loading model from: ', model_path)
        else:
            print("Traning Model from scratch")
            init = tf.global_variables_initializer()
            self.persistent_sess.run(init)

        self.action_histories = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.action_histories_old = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
        self.histories = []
        self.iteration = 0
        self.imgwh = None

        self.callback_redetection = self.redetection_by_sampling
        self.failed_cnt = 0
        self.latest_score = 0

        self.stopwatch = StopWatchManager()

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
        pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']


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
            ADNetConf.get()['finetune']['learning_rate'],
            ADNetConf.get()['finetune']['iter']
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

        for i in range(iter):
            batch_idxs = commons.random_idxs(len(pos_boxes), BATCHSIZE)
            batch_feats = [x.feat for x in commons.choices_by_idx(pos_boxes, batch_idxs)]
            batch_lb_action = commons.choices_by_idx(pos_lb_action, batch_idxs)
            self.persistent_sess.run(
                self.adnet.weighted_grads_op1,
                feed_dict={
                    self.adnet.layer_feat: batch_feats,
                    self.adnet.label_tensor: batch_lb_action,
                    self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 1, 1, 110)),
                    self.learning_rate_placeholder: learning_rate,
                    self.tensor_is_training: True
                }
            )

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
                self.persistent_sess.run(
                    self.adnet.weighted_grads_op2,
                    feed_dict={
                        self.adnet.layer_feat: batch_feats + batch_feats_neg,
                        self.adnet.class_tensor: [1]*len(batch_feats) + [0]*len(batch_feats_neg),
                        self.adnet.action_history_tensor: np.zeros(shape=(len(batch_feats)+len(batch_feats_neg), 1, 1, 110)),
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True
                    }
                )

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
        startFrames = range(num_frames) - ADNetConf.get()['rl_episode']['frame_steps']  ## check if number of frames are sufficient
        endFrames = [i + ADNetConf.get()['rl_episode']['frame_steps'] for i in startFrames]
        numOfClips = min(ADNetConf.get()['rl_episode']['num_frames_per_video'], len(startFrames))
        randomIndex = range(len(startFrames))
        random.shuffle(randomIndex)
        startFrames = startFrames[randomIndex]
        endFrames = endFrames[randomIndex]

        onehots_pos = []
        onehots_neg = []
        imgs_pos = []
        imgs_neg = []
        action_labels_pos = []
        action_labels_neg = []
        for i in range(numOfClips):
            img_boxes = []
            action_labels = []
            one_hots = []
            for frame_num in range(startFrames[i], endFrames[i]):  # for each frame in the seq of clips
                img_index = gt_box_tuples[frame_num][0]
                im_path = self.get_image_path(vid_path, img_index)
                img = commons.imread(im_path)
                self.imgwh = Coordinate.get_imgwh(img)
                curr_gt_bbox = gt_box_tuples[frame_num][1]
                ##TODO: if black n white photo append to create 3 channel image
                curr_bbox, boxes, actions_seq, onehot_seq = self.tracking4training(img, curr_gt_bbox)
                img_boxes.append([(img_index, box) for box in boxes])
                one_hots.append(onehot_seq)
                action_labels.append(actions_seq)

            if curr_gt_bbox.iou(curr_bbox)>0.7:
                ##append to pos data
                onehots_pos.append(one_hots)
                imgs_pos.append(img_boxes)
                action_labels_pos.append(action_labels)

            else:
                ## append to neg data
                onehots_neg.append(one_hots)
                imgs_neg.append(img_boxes)
                action_labels_neg.append(action_labels)

        ## Policy Gradient Training
        num_pos = len(action_labels_pos)
        num_neg = len(action_labels_neg)
        train_pos_cnt = 0
        train_pos = []
        train_neg_cnt = 0
        train_neg = []
        batch_size = ADNetConf.get()['rl_episode']['batch_size']
        if num_pos>batch_size/2:
            remain = batch_size*numOfClips
            while(remain>0):
                if train_pos_cnt==0:
                    train_pos_list = range(num_pos)
                    random.shuffle(train_pos_list)

                train_pos.append(train_pos_list[train_pos_cnt+1:min(len(train_pos_list), train_pos_cnt + remain)])
                train_pos_cnt = min(len(train_pos_list), train_pos_cnt + remain)
                train_pos_cnt = train_pos_cnt%len(train_pos_list)
                remain = batch_size*numOfClips - len(train_pos)

        if num_neg>batch_size/2:
            remain = batch_size*numOfClips
            while(remain>0):
                if train_neg_cnt==0:
                    train_neg_list = range(num_neg)
                    random.shuffle(train_neg_list)

                train_neg.append(train_neg_list[train_neg_cnt+1:min(len(train_neg_list), train_neg_cnt + remain)])
                train_neg_cnt = min(len(train_neg_list), train_neg_cnt + remain)
                train_neg_cnt = train_neg_cnt%len(train_neg_list)
                remain = batch_size*numOfClips - len(train_neg)

        ## training
        for batch_idx in range(numOfClips):
            if train_pos!=[]:
                pos_examples = train_pos[batch_idx*batch_size:(batch_idx+1)*batch_size]
                imgs_patches = []
                for i, pos_ex_index in enumerate(pos_examples):
                    img = commons.imread(self.get_image_path(vid_path, imgs_pos[pos_ex_index][0]))
                    imgs_patches.append(commons.extract_region(img, imgs_pos[pos_ex_index][1]))

                imgs_patches_feat = self._get_features(imgs_patches)
                action_labels = action_labels_pos[pos_examples]
                action_histories = onehots_pos[pos_examples]
                self.persistent_sess.run(
                    self.adnet.weighted_grads_rl,
                    feed_dict={
                        self.adnet.layer_feat: imgs_patches_feat,
                        self.adnet.label_tensor: action_labels,
                        self.adnet.reward: [1]*len(action_labels),
                        self.adnet.action_history_tensor: action_histories,  ## TODO reshape to np.zeros(shape=(BATCHSIZE, 1, 1, 110))
                        self.learning_rate_placeholder: ADNetConf.get()['rl_episode']['lr'],
                        self.tensor_is_training: True
                    }
                )

            if train_neg!=[]:
                neg_examples = train_neg[batch_idx*batch_size:(batch_idx+1)*batch_size]
                imgs_patches = []
                for i, neg_ex_index in enumerate(neg_examples):
                    img = commons.imread(self.get_image_path(vid_path, imgs_neg[neg_ex_index][0]))
                    imgs_patches.append(commons.extract_region(img, imgs_neg[neg_ex_index][1]))

                imgs_patches_feat = self._get_features(imgs_patches)
                action_labels = action_labels_neg[neg_examples]
                action_histories = onehots_neg[neg_examples]
                self.persistent_sess.run(
                    self.adnet.weighted_grads_rl,
                    feed_dict={
                        self.adnet.layer_feat: imgs_patches_feat,
                        self.adnet.label_tensor: action_labels,
                        self.adnet.reward: [-1]*len(action_labels),
                        self.adnet.action_history_tensor: action_histories,  ## TODO reshape to np.zeros(shape=(BATCHSIZE, 1, 1, 110))
                        self.learning_rate_placeholder: ADNetConf.get()['rl_episode']['lr'],
                        self.tensor_is_training: True
                    }
                )


    def tracking4training(self, img, curr_bbox):
        self.iteration += 1
        is_tracked = True
        boxes = []
        actions_seq = []
        onehot_seq = np.zeros(len(ADNetConf.get()['rl_episode']['num_action'])*len(ADNetConf.get()['rl_episode']['num_action_history']), len(ADNetConf.get()['rl_episode']['num_action_step_max']))
        self.latest_score = -1
        self.stopwatch.start('tracking4training.do_action')
        track_i = 0
        for track_i in range(ADNetConf.get()['rl_episode']['num_action']):
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
            if latest_score < ADNetConf.g()['rl_episode']['thresh_fail']:
                is_tracked = False
                self.action_histories_old = np.copy(self.action_histories)
                self.action_histories = np.insert(self.action_histories, 0, 12)[:-1]
                break
            else:
                self.failed_cnt = 0
            self.latest_score = latest_score

            # move box
            onehot_seq[:,track_i] = self.action_history2onehot(onehot_seq.shape[0])
            action_idx = np.argmax(actions[0])
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            prev_bbox = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)
            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    print(prev_bbox)
                    print(curr_bbox)
                    raise Exception('box not moved.')

            # oscillation check
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP

            if action_idx == ADNetwork.ACTION_IDX_STOP:
                break

            boxes.append(curr_bbox)
            actions_seq.append(action_idx)

        onehot_seq = onehot_seq[:, :track_i]
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
        cv2.imshow('patch', patch)
        return curr_bbox, boxes, actions_seq, onehot_seq


    def action_history2onehot(self, shape):
        row = len(ADNetConf.get()['rl_episode']['num_action'])
        col = len(ADNetConf.get()['rl_episode']['num_action_history'])
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
    parser.add_argument("-debug", type=str2bool, nargs= '?', default=True)
    parser.add_argument("-mode", type=str, default='test')
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
    model_path = 'checkpoints'
    verbose = args.verbose
    mode = args.mode
    debug = args.debug
    model = ADNetRunner(model_path=model_path, verbose=verbose)

    if debug:
        if mode == 'train':
            print('Training with debug mode')
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
        num_files_done = 84

        num_epochs = 30

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
                print("video num: {}/{}, {}".format(cnt, num_videos, folder))
                model.train(vid_path=folder)

                with open('log.txt', 'w+') as f:
                    f.write('writting file epch: {}, folder num: {}, folder: {}'.format(str(epoch), i , folder))
                    f.flush()

        print("Total Time Taken in Training: ", time.time() - t1)
    else:
        print("=====================Running Inference==================")
        model.by_dataset(vid_path=vid_path)
