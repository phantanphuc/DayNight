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


class SolverPredict(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, self.flags)

        self.dataset.day_path = './testsite/day'
        self.dataset.night_path = './testsite/night'
        self.dataset.condition_path = './testsite/condition/im.jpg'

        self.model = cycleGAN(self.sess, self.flags, self.dataset.image_size, self.dataset())
        self.iter_time = 0

        self._make_folders()

        self.saver = tf.train.Saver()
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


    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        # read test data
        test_data_files = utils.all_files_under(self.dataset.day_path)

        total_time = 0.

        condition = utils.transform(utils.imagefiles2arrs([self.dataset.condition_path]))

        for idx in range(len(test_data_files)):

            img = utils.imagefiles2arrs([test_data_files[idx]])  # read img
            img = utils.transform(img)  # convert [0, 255] to [-1., 1.]

            # measure inference time
            start_time = time.time()
            print(test_data_files[idx])
            try:
            	imgs = self.model.test_step_v2(img, condition, mode='YtoX')  # inference
            except:
            	pass
            total_time += time.time() - start_time



            self.model.plots(imgs, idx, self.dataset.image_size, '/home/cpu11467/WORK/lv_new/cycleGAN/testsite/out')  # write results

        print('Avg PT: {:3f} msec.'.format(total_time / len(test_data_files) * 1000.))

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

  

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')

            return True
        else:
            return False


import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have muliple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 2, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'day2night', 'dataset name, default day2night')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 1000, 'save frequency for model, default: 1000')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency for saving image, default: 200')
tf.flags.DEFINE_string('load_model', '20200519-1304', 'folder of saved model that you wish to continue training '
                                           '(e.g. 20180412-1610), default: None')


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

solver = SolverPredict(FLAGS)
solver.test()

tf.app.run()

