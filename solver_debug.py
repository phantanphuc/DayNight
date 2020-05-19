# ---------------------------------------------------------
# Tensorflow CycleGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
# import cv2
import collections
import numpy as np
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
from dataset import Dataset
from cycle_gan import cycleGAN


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, self.flags)
        self.model = cycleGAN(self.sess, self.flags, self.dataset.image_size, self.dataset())
        self.iter_time = 0

        # self._make_folders()

        # self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time))

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train_debug(self):
        # load initialized checkpoint that provided
        # if self.flags.load_model is not None:
        #     if self.load_model():
        #         print(' [*] Load SUCCESS!\n')
        #     else:
        #         print(' [!] Load Failed...\n')

        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            # for iter_time in range(self.flags.iters):
            # while self.iter_time < self.flags.iters:
            #     # samppling images and save them
            #     # self.sample()

            #     # train_step
            #     loss, summary = self.model.train_step()
            #     self.print_info(loss)
            #     self.train_writer.add_summary(summary, self.iter_time)
            #     self.train_writer.flush()

            #     # save model
            #     # self.save_model()

            #     self.iter_time += 1
                
            #     break

            tensor = tf.range(10)
            out = tf.add(tensor, tensor)
            print(out.eval(session = self.sess))

            reader = tf.TFRecordReader()
            filename_queue = tf.train.string_input_producer(['../data/tfrecords/alderley_day.tfrecords'])
            _, serialized_example = reader.read(filename_queue)

            features = tf.parse_single_example(serialized_example, features={
                        'image/file_name': tf.FixedLenFeature([], tf.string),
                        'image/encoded_image': tf.FixedLenFeature([], tf.string)})

            print(features['image/file_name'].eval(session = self.sess))


        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)         


    
    def sample(self):
        if np.mod(self.iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, self.iter_time, self.dataset.image_size, self.sample_out_dir)

    def print_info(self, loss):
        if np.mod(self.iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('G_loss', loss[0]), ('Dy_loss', loss[1]),
                                                  ('F_loss', loss[2]), ('Dx_loss', loss[3]),
                                                  ('dataset', self.dataset.name),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(self.iter_time, ord_output)

