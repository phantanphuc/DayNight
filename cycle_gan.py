# ---------------------------------------------------------
# Tensorflow CycleGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from vanhuyz
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np

# import cv2
# import scipy.misc

# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
from reader import Reader


# noinspection PyPep8Naming
class cycleGAN(object):
    def __init__(self, sess, flags, image_size, data_path):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.x_path, self.y_path = data_path[0], data_path[1]

        # True: use lsgan (mean squared error)
        # False: use cross entropy loss
        self.use_lsgan = True
        self.use_sigmoid = not self.use_lsgan
        # [instance|batch] use instance norm or batch norm, default: instance
        self.norm = 'instane'
        self.lambda1, self.lambda2 = 10.0, 10.0
        self.ngf, self.ndf = 64, 64
        self.real_label = 0.9
        self.start_decay_step = 100000
        self.decay_steps = 100000
        self.eps = 1e-12
        self.hist_bin_size = 128

        self._G_gen_train_ops, self._F_gen_train_ops = [], []
        self._Dy_dis_train_ops, self._Dx_dis_train_ops = [], []

        self._build_net()
        self._tensorboard()

    def _build_net(self):
        # tfph: TensorFlow PlaceHolder
        self.x_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='x_test_tfph')
        self.y_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='y_test_tfph')

        # self.x_test_hist_tfph = tf.placeholder(tf.float32, shape=[None, *(self.image_size[0], self.image_size[1], int(3 * self.hist_bin_size / 4))], name='x_test_hist_tfph')
        # self.y_test_hist_tfph = tf.placeholder(tf.float32, shape=[None, *(self.image_size[0], self.image_size[1], int(3 * self.hist_bin_size / 4))], name='y_test_hist_tfph')
        
        self.x_test_hist_tfph = tf.placeholder(tf.float32, shape=[None, 128], name='x_test_hist_tfph')
        self.y_test_hist_tfph = tf.placeholder(tf.float32, shape=[None, 128], name='y_test_hist_tfph')

        self.fake_x_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_x_tfph')
        self.fake_y_tfph = tf.placeholder(tf.float32, shape=[None, *self.image_size], name='fake_y_tfph')

        self.G_gen = GeneratorImprove(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._G_gen_train_ops)

        # self.G_gen = Generator(name='G', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
        #                        _ops=self._G_gen_train_ops)

        self.Dy_dis = Discriminator(name='Dy', ndf=self.ndf, norm=self.norm, _ops=self._Dy_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)
        # self.F_gen = Generator(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
        #                        _ops=self._F_gen_train_ops)
        self.F_gen = GeneratorImprove(name='F', ngf=self.ngf, norm=self.norm, image_size=self.image_size,
                               _ops=self._F_gen_train_ops)
        self.Dx_dis = Discriminator(name='Dx', ndf=self.ndf, norm=self.norm, _ops=self._Dx_dis_train_ops,
                                    use_sigmoid=self.use_sigmoid)

        x_reader = Reader(self.x_path, name='X', image_size=self.image_size, batch_size=self.flags.batch_size)
        y_reader = Reader(self.y_path, name='Y', image_size=self.image_size, batch_size=self.flags.batch_size)
        self.x_imgs = x_reader.feed()
        self.y_imgs = y_reader.feed()
        

        self.histogram_x = self.createHistogramV2(self.x_imgs, self.hist_bin_size)
        self.histogram_y = self.createHistogramV2(self.y_imgs, self.hist_bin_size)

        ###########################################################


        self.fake_x_pool_obj = utils.ImagePool(pool_size=50)
        self.fake_y_pool_obj = utils.ImagePool(pool_size=50)

        # cycle consistency loss
        cycle_loss = self.cycle_consistency_loss(self.x_imgs, self.y_imgs, self.histogram_x, self.histogram_y)

        # X -> Y
        self.fake_y_imgs = self.G_gen(self.x_imgs, self.histogram_y)
        self.G_gen_loss = self.generator_loss(self.Dy_dis, self.fake_y_imgs, use_lsgan=self.use_lsgan)
        # self.G_loss = self.G_gen_loss + cycle_loss
        self.Dy_dis_loss = self.discriminator_loss(self.Dy_dis, self.y_imgs, self.fake_y_tfph,
                                                   use_lsgan=self.use_lsgan)

        # Y -> X
        self.fake_x_imgs = self.F_gen(self.y_imgs, self.histogram_x)
        self.F_gen_loss = self.generator_loss(self.Dx_dis, self.fake_x_imgs, use_lsgan=self.use_lsgan)
        # self.F_loss = self.F_gen_loss + cycle_loss
        self.Dx_dis_loss = self.discriminator_loss(self.Dx_dis, self.x_imgs, self.fake_x_tfph,
                                                   use_lsgan=self.use_lsgan)

        # HIST LOSS
        self.histogram_fake_x = self.createHistogramV2(self.fake_x_imgs, self.hist_bin_size)
        self.histogram_fake_y = self.createHistogramV2(self.fake_y_imgs, self.hist_bin_size)

        hist_loss_x = tf.reduce_sum(tf.math.abs(self.histogram_fake_x - self.histogram_x))
        hist_loss_y = tf.reduce_sum(tf.math.abs(self.histogram_fake_y - self.histogram_y))

        self.G_loss = self.G_gen_loss + cycle_loss + hist_loss_y / 384
        self.F_loss = self.F_gen_loss + cycle_loss + hist_loss_x / 384

        # print(self.G_gen_loss)
        # print(self.G_loss)
        # print(self.Dy_dis_loss)
        #

        G_optim = self.optimizer(loss=self.G_loss, variables=self.G_gen.variables, name='Adam_G')
        Dy_optim = self.optimizer(loss=self.Dy_dis_loss, variables=self.Dy_dis.variables, name='Adam_Dy')
        F_optim = self.optimizer(loss=self.F_loss, variables=self.F_gen.variables, name='Adam_F')
        Dx_optim = self.optimizer(loss=self.Dx_dis_loss, variables=self.Dx_dis.variables, name='Adam_Dx')
        self.optims = tf.group([G_optim, Dy_optim, F_optim, Dx_optim])
        # with tf.control_dependencies([G_optim, Dy_optim, F_optim, Dx_optim]):
        #     self.optims = tf.no_op(name='optimizers')

        print(self.x_test_tfph)
        print(self.y_test_hist_tfph)
        print(self.histogram_fake_x)

        # for sampling function
        self.fake_y_sample = self.G_gen(self.x_test_tfph, self.y_test_hist_tfph)
        self.fake_x_sample = self.F_gen(self.y_test_tfph, self.x_test_hist_tfph)

    def createHistogramV2(self, img, bin_count):

        hsv_img = tf.image.rgb_to_hsv((img + 1.0) / 2.0)
        yuv_img = tf.image.rgb_to_yuv((img + 1.0) / 2.0)

        bin_size = 1 / bin_count
        hist_entries_hsv = []
        # hist_entries_yuv = []

        bin_count = int(bin_count / 1)
        for idx, i in enumerate(np.arange(0.0, 1.0, bin_size)):
            gt_hsv = tf.greater(img, i)
            leq_hsv = tf.less_equal(img, i + bin_size)

            node_hsv = tf.reduce_sum(tf.cast(tf.logical_and(gt_hsv, leq_hsv), tf.float32), axis=(1, 2))
            hist_entries_hsv.append(node_hsv)

            #------------
            # gt = tf.greater(img, i)
            # leq = tf.less_equal(img, i + bin_size)

            # node = tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), axis=(1, 2))
            # hist_entries.append(node)

        hist = tf.stack(hist_entries_hsv)
        hist = tf.transpose(hist, perm=[1, 2, 0])
        
        hist = tf.slice(hist, [0, 2, 0], [-1, 1, bin_count])

        hist = tf.reshape(hist, shape=(-1, bin_count))


        # hist = tf.tile(hist, [1, self.image_size[0] * self.image_size[1]])
        # hist = tf.reshape(hist, shape=(-1, self.image_size[0], self.image_size[1], 3 * bin_count))

        return hist / (self.image_size[0] * self.image_size[1])

    def createHistogram(self, img, bin_count):
        bin_size = 2/bin_count
        hist_entries = []

        bin_count = int(bin_count / 4)
        for idx, i in enumerate(np.arange(-0.25, 0.25, bin_size)):
            gt = tf.greater(img, i)
            leq = tf.less_equal(img, i + bin_size)

            node = tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), axis=(1, 2))
            hist_entries.append(node)

        hist = tf.stack(hist_entries)
        hist = tf.transpose(hist, perm=[1, 2, 0])
        hist = tf.reshape(hist, shape=(-1, 3 * bin_count))
        hist = tf.tile(hist, [1, self.image_size[0] * self.image_size[1]])
        hist = tf.reshape(hist, shape=(-1, self.image_size[0], self.image_size[1], 3 * bin_count))
        return hist / (self.image_size[0] * self.image_size[1])


    def optimizer(self, loss, variables, name='Adam'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_decay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate, power=1.0),
                                  starter_learning_rate))
        tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.flags.beta1, name=name).\
            minimize(loss, global_step=global_step, var_list=variables)

        return learn_step

    def cycle_consistency_loss(self, x_imgs, y_imgs, hist_x, hist_y):
        forward_loss = tf.reduce_mean(tf.abs(self.F_gen(self.G_gen(x_imgs, hist_y), hist_x) - x_imgs))
        backward_loss = tf.reduce_mean(tf.abs(self.G_gen(self.F_gen(y_imgs, hist_x), hist_y) - y_imgs))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def generator_loss(self, dis_obj, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            loss = 0.5 * tf.reduce_mean(tf.squared_difference(dis_obj(fake_img), self.real_label))
        else:
            # heuristic, non-saturating loss (I don't understand here!)
            # loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps)) / 2.  (???)
            loss = -tf.reduce_mean(tf.log(dis_obj(fake_img) + self.eps))
        return loss

    def discriminator_loss(self, dis_obj, real_img, fake_img, use_lsgan=True):
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(dis_obj(real_img), self.real_label))
            error_fake = tf.reduce_mean(tf.square(dis_obj(fake_img)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(tf.log(dis_obj(real_img) + self.eps))
            error_fake = -tf.reduce_mean(tf.log(1. - dis_obj(fake_img) + self.eps))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def _tensorboard(self):
        tf.summary.histogram('Dy/real', self.Dy_dis(self.y_imgs))
        # tf.summary.histogram('Dy/fake', self.Dy_dis(self.G_gen(self.x_imgs)))
        tf.summary.histogram('Dx/real', self.Dx_dis(self.x_imgs))
        # tf.summary.histogram('Dx/fake', self.Dx_dis(self.F_gen(self.y_imgs)))

        tf.summary.scalar('loss/G_gen', self.G_gen_loss)
        tf.summary.scalar('loss/Dy_dis', self.Dy_dis_loss)
        tf.summary.scalar('loss/F_gen', self.F_gen_loss)
        tf.summary.scalar('loss/Dx_dis', self.Dx_dis_loss)

        # tf.summary.image('X/generated_Y', tf_utils.batch_convert2int(self.G_gen(self.x_imgs)))
        # tf.summary.image('X/reconstruction', tf_utils.batch_convert2int(self.F_gen(self.G_gen(self.x_imgs))))
        # tf.summary.image('Y/generated_X', tf_utils.batch_convert2int(self.F_gen(self.y_imgs)))
        # tf.summary.image('Y/reconstruction', tf_utils.batch_convert2int(self.G_gen(self.F_gen(self.y_imgs))))

        self.summary_op = tf.summary.merge_all()

    def train_step(self):
        fake_y_val, fake_x_val, x_val, y_val = self.sess.run([self.fake_y_imgs, self.fake_x_imgs,
                                                              self.x_imgs, self.y_imgs])
        _, G_loss, Dy_loss, F_loss, Dx_loss, summary = \
            self.sess.run([self.optims, self.G_loss, self.Dy_dis_loss,
                           self.F_loss, self.Dx_dis_loss, self.summary_op],
                          feed_dict={self.fake_x_tfph: self.fake_x_pool_obj.query(fake_x_val),
                                     self.fake_y_tfph: self.fake_y_pool_obj.query(fake_y_val)})

        return [G_loss, Dy_loss, F_loss, Dx_loss], summary

    def sample_imgs(self):
        x_val, y_val = self.sess.run([self.x_imgs, self.y_imgs])
        x_hist, y_hist = self.sess.run([self.histogram_x, self.histogram_y])

        fake_y, fake_x = self.sess.run([self.fake_y_sample, self.fake_x_sample],
                                       feed_dict={self.x_test_tfph: x_val, self.y_test_tfph: y_val, self.x_test_hist_tfph: x_hist, self.y_test_hist_tfph: y_hist})

        return [x_val, fake_y, y_val, fake_x]

    def test_step(self, img, mode='XtoY'):
        if mode == 'XtoY':
            fake_y = self.sess.run(self.fake_y_sample, feed_dict={self.x_test_tfph: img})
            return [img, fake_y]
        elif mode == 'YtoX':
            fake_x = self.sess.run(self.fake_x_sample, feed_dict={self.y_test_tfph: img})
            return [img, fake_x]
        else:
            raise NotImplementedError

    @staticmethod
    def plots(imgs, iter_time, image_size, save_file):
        # parameters for plot size
        scale, margin = 0.02, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow((imgs[col_index][row_index]).reshape(
                    image_size[0], image_size[1], image_size[2]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def write_tensorboard():
        print('hello write_tensorboard!')


class GeneratorImprove(object):
    def __init__(self, name=None, ngf=64, norm='instance', image_size=(128, 256, 3), _ops=None):
        self.name = name
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x, hist):
        with tf.variable_scope(self.name, reuse=self.reuse):


            # tf_utils.print_activations(x)

            # print(hist)
            # quit()

            # tensor_x_temp = tf.identity(x)

            # # hist (N, H, W, bin_size) 
            # #########################################################################################
            # hist_tensor = tf.concat([hist, tensor_x_temp], axis=3, name='histogram_tensor')
            # # hist_tensor = tf.concat([hist, x], axis=3, name='hist_tensor')

            # hist_tensor = tf_utils.conv2d(hist_tensor, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='SAME',
            #                         name='histogram_conv')

            # hist_tensor = tf_utils.norm(hist_tensor, _type='instance', _ops=self._ops, name='histogram_norm')
            # hist_tensor = tf_utils.relu(hist_tensor, name='histogram_relu', is_print=True)

            # #########################################################################################

            # x = tf.concat([hist_tensor, x], axis=3)

            #############################

            #expect 256 * 512 = 131072

            # 386
            fc_1 = tf_utils.dense(hist, 512, name='fc_1')
            fc_1 = tf_utils.relu(fc_1, name='fc1_relu', is_print=True)

            fc_2 = tf_utils.dense(fc_1, 512, name='fc_2')
            fc_2 = tf_utils.relu(fc_2, name='fc2_relu', is_print=True)

            fc_3 = tf_utils.dense(fc_2, 1024 , name='fc_3')
            fc_3 = tf_utils.relu(fc_3, name='fc3_relu', is_print=True)

            fc_4 = tf_utils.dense(fc_3, 2048, name='fc_4')
            fc_4 = tf_utils.relu(fc_4, name='fc4_relu', is_print=True)

            # fc_5 = tf_utils.dense(fc_4, 32768, name='fc_5')
            # fc_5 = tf_utils.relu(fc_4, name='fc5_relu', is_print=True)

            # fc_6 = tf_utils.dense(fc_5, 131072, name='fc_6')
            # fc_6 = tf_utils.relu(fc_6, name='fc6_relu', is_print=True)

            hist_data = tf.reshape(fc_4, shape=(-1, 32, 64, 1))
            hist_data_deconv = tf_utils.deconv2d(hist_data, 4, name='histogram1_deconv2d')
            hist_data_deconv = tf_utils.norm(hist_data_deconv, _type='instance', _ops=self._ops, name='histogram1_norm')
            hist_data_deconv = tf_utils.relu(hist_data_deconv, name='histogram1_relu', is_print=True)
            

            hist_data_deconv = tf_utils.deconv2d(hist_data_deconv, 8, name='histogram2_deconv2d')
            hist_data_deconv = tf_utils.norm(hist_data_deconv, _type='instance', _ops=self._ops, name='histogram2_norm')
            hist_data_deconv = tf_utils.relu(hist_data_deconv, name='histogram2_relu', is_print=True)


            hist_data_deconv = tf_utils.deconv2d(hist_data_deconv, 13, name='histogram3_deconv2d')
            hist_data_deconv = tf_utils.norm(hist_data_deconv, _type='instance', _ops=self._ops, name='histogram3_norm')
            hist_data_deconv = tf_utils.relu(hist_data_deconv, name='histogram3_relu', is_print=True)


            # hist_data_deconv = tf_utils.deconv2d(hist_data_deconv, self.ngf, name='histogram4_deconv2d')
            # hist_data_deconv = tf_utils.norm(hist_data_deconv, _type='instance', _ops=self._ops, name='histogram4_norm')
            # hist_data_deconv = tf_utils.relu(hist_data_deconv, name='histogram4_relu', is_print=True)


            # hist_tensor = tf_utils.conv2d(hist_data, self.ngf / 2, k_h=7, k_w=7, d_h=1, d_w=1, padding='SAME',
            #                         name='histogram1_conv')
            # hist_tensor = tf_utils.norm(hist_tensor, _type='instance', _ops=self._ops, name='histogram1_norm')
            # hist_tensor = tf_utils.relu(hist_tensor, name='histogram1_relu', is_print=True)


            # hist_tensor = tf_utils.conv2d(hist_tensor, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='SAME',
            #                         name='histogram2_conv')
            # hist_tensor = tf_utils.norm(hist_tensor, _type='instance', _ops=self._ops, name='histogram2_norm')
            # hist_tensor = tf_utils.relu(hist_tensor, name='histogram2_relu', is_print=True)


            x = tf.concat([hist_data_deconv, x], axis=3)


            # (N, H, W, C) -> (N, H, W, 64)
            conv1 = tf_utils.padding2d(x, p_h=3, p_w=3, pad_type='REFLECT', name='conv1_padding')
            conv1 = tf_utils.conv2d(conv1, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID',
                                    name='conv1_conv')

            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.relu(conv1, name='conv1_relu', is_print=True)

            # (N, H, W, 64)  -> (N, H/2, W/2, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm',)
            conv2 = tf_utils.relu(conv2, name='conv2_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H/4, W/4, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm',)
            conv3 = tf_utils.relu(conv3, name='conv3_relu', is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/4, W/4, 256)
            if (self.image_size[0] <= 128) and (self.image_size[1] <= 128):
                # use 6 residual blocks for 128x128 images
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=6, is_print=True)
            else:
                # use 9 blocks for higher resolution
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=9, is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/2, W/2, 128)
            conv4 = tf_utils.deconv2d(res_out, 2*self.ngf, name='conv4_deconv2d')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.relu(conv4, name='conv4_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H, W, 64)
            conv5 = tf_utils.deconv2d(conv4, self.ngf, name='conv5_deconv2d')
            conv5 = tf_utils.norm(conv5, _type='instance', _ops=self._ops, name='conv5_norm')
            conv5 = tf_utils.relu(conv5, name='conv5_relu', is_print=True)

            # (N, H, W, 64) -> (N, H, W, 3)
            conv6 = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            conv6 = tf_utils.conv2d(conv6, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                    padding='VALID', name='output_conv')
            output = tf_utils.tanh(conv6, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Generator(object):
    def __init__(self, name=None, ngf=64, norm='instance', image_size=(128, 256, 3), _ops=None):
        self.name = name
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H, W, 64)
            conv1 = tf_utils.padding2d(x, p_h=3, p_w=3, pad_type='REFLECT', name='conv1_padding')
            conv1 = tf_utils.conv2d(conv1, self.ngf, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID',
                                    name='conv1_conv')
            conv1 = tf_utils.norm(conv1, _type='instance', _ops=self._ops, name='conv1_norm')
            conv1 = tf_utils.relu(conv1, name='conv1_relu', is_print=True)

            # (N, H, W, 64)  -> (N, H/2, W/2, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm',)
            conv2 = tf_utils.relu(conv2, name='conv2_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H/4, W/4, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ngf, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm',)
            conv3 = tf_utils.relu(conv3, name='conv3_relu', is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/4, W/4, 256)
            if (self.image_size[0] <= 128) and (self.image_size[1] <= 128):
                # use 6 residual blocks for 128x128 images
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=6, is_print=True)
            else:
                # use 9 blocks for higher resolution
                res_out = tf_utils.n_res_blocks(conv3, num_blocks=9, is_print=True)

            # (N, H/4, W/4, 256) -> (N, H/2, W/2, 128)
            conv4 = tf_utils.deconv2d(res_out, 2*self.ngf, name='conv4_deconv2d')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.relu(conv4, name='conv4_relu', is_print=True)

            # (N, H/2, W/2, 128) -> (N, H, W, 64)
            conv5 = tf_utils.deconv2d(conv4, self.ngf, name='conv5_deconv2d')
            conv5 = tf_utils.norm(conv5, _type='instance', _ops=self._ops, name='conv5_norm')
            conv5 = tf_utils.relu(conv5, name='conv5_relu', is_print=True)

            # (N, H, W, 64) -> (N, H, W, 3)
            conv6 = tf_utils.padding2d(conv5, p_h=3, p_w=3, pad_type='REFLECT', name='output_padding')
            conv6 = tf_utils.conv2d(conv6, self.image_size[2], k_h=7, k_w=7, d_h=1, d_w=1,
                                    padding='VALID', name='output_conv')
            output = tf_utils.tanh(conv6, name='output_tanh', is_print=True)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name='', ndf=64, norm='instance', _ops=None, use_sigmoid=False):
        self.name = name
        self.ndf = ndf
        self.norm = norm
        self._ops = _ops
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # tf_utils.print_activations(x)

            # (N, H, W, C) -> (N, H/2, W/2, 64)
            conv1 = tf_utils.conv2d(x, self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv1_conv')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu', is_print=True)

            # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
            conv2 = tf_utils.conv2d(conv1, 2*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv2_conv')
            conv2 = tf_utils.norm(conv2, _type='instance', _ops=self._ops, name='conv2_norm')
            conv2 = tf_utils.lrelu(conv2, name='conv2_lrelu', is_print=True)

            # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
            conv3 = tf_utils.conv2d(conv2, 4*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv3_conv')
            conv3 = tf_utils.norm(conv3, _type='instance', _ops=self._ops, name='conv3_norm')
            conv3 = tf_utils.lrelu(conv3, name='conv3_lrelu', is_print=True)

            # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
            conv4 = tf_utils.conv2d(conv3, 8*self.ndf, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME',
                                    name='conv4_conv')
            conv4 = tf_utils.norm(conv4, _type='instance', _ops=self._ops, name='conv4_norm')
            conv4 = tf_utils.lrelu(conv4, name='conv4_lrelu', is_print=True)

            # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
            conv5 = tf_utils.conv2d(conv4, 1, k_h=4, k_w=4, d_h=1, d_w=1, padding='SAME',
                                    name='conv5_conv', is_print=True)

            if self.use_sigmoid:
                output = tf_utils.sigmoid(conv5, name='output_sigmoid', is_print=True)
            else:
                output = tf.identity(conv5, name='output_without_sigmoid')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
