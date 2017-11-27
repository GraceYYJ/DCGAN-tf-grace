from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from model import *

class Train(object):
    def __init__(self,sess):
        self.sess=sess

    def train(self, model,config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(model.d_loss, var_list=model.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(model.g_loss, var_list=model.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([model.z_sum, model.d__sum,
                                    model.G_sum, model.d_loss_fake_sum, model.g_loss_sum])
        self.d_sum = merge_summary(
            [model.z_sum, model.d_sum, model.d_loss_real_sum, model.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(model.sample_num, model.z_dim))

        if config.dataset == 'mnist':
            sample_inputs = model.data_X[0:model.sample_num]
            sample_labels = model.data_y[0:model.sample_num]
        else:
            sample_files = model.data[0:model.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=model.input_height,
                          input_width=model.input_width,
                          resize_height=model.output_height,
                          resize_width=model.output_width,
                          crop=model.crop,
                          grayscale=model.grayscale) for sample_file in sample_files]
            if (model.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = model.load(self,model.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(model.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob(os.path.join(
                    "./data", config.dataset, model.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = model.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = model.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                else:
                    batch_files = model.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch = [
                        get_image(batch_file,
                                  input_height=model.input_height,
                                  input_width=model.input_width,
                                  resize_height=model.output_height,
                                  resize_width=model.output_width,
                                  crop=model.crop,
                                  grayscale=model.grayscale) for batch_file in batch_files]
                    if model.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, model.z_dim]) \
                    .astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       model.inputs: batch_images,
                                                       model.z: batch_z,
                                                       model.y: batch_labels,
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       model.z: batch_z,
                                                       model.y: batch_labels,
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={model.z: batch_z, model.y: batch_labels})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = model.d_loss_fake.eval({
                        model.z: batch_z,
                        model.y: batch_labels
                    })
                    errD_real = model.d_loss_real.eval({
                        model.inputs: batch_images,
                        model.y: batch_labels
                    })
                    errG = model.g_loss.eval({
                        model.z: batch_z,
                        model.y: batch_labels
                    })
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={model.inputs: batch_images, model.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={model.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={model.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({model.z: batch_z})
                    errD_real = self.d_loss_real.eval({model.inputs: batch_images})
                    errG = self.g_loss.eval({model.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [model.sampler, model.d_loss, model.g_loss],
                            feed_dict={
                                model.z: sample_z,
                                model.inputs: sample_inputs,
                                model.y: sample_labels,
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [model.sampler, model.d_loss, model.g_loss],
                                feed_dict={
                                    model.z: sample_z,
                                    model.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        except:
                            print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    model.save(self,config.checkpoint_dir, counter)
