from image_manipulation import image_resizing
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def model_inputs(image_width, image_height, image_channels, z_dim):

    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_channels), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs_real, inputs_z, learning_rate


def discriminator(images, reuse=False):
    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):

        conv1 = tf.layers.conv2d(images, 64, 5, 2, 'SAME')
        lrelu1 = tf.maximum(alpha * conv1, conv1)

        conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME')
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        conv3 = tf.layers.conv2d(lrelu2, 256, 5, 1, 'SAME')
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

        flat = tf.reshape(lrelu3, (-1, 4 * 4 * 256))

        logits = tf.layers.dense(flat, 1)

        out = tf.sigmoid(logits)

        return out, logits


def generator(z, out_channel_dim, is_train=True):
    alpha = 0.2

    with tf.variable_scope('generator', reuse=False if is_train == True else True):
        x_1 = tf.layers.dense(z, 2 * 2 * 512)

        deconv_2 = tf.reshape(x_1, (-1, 2, 2, 512))
        batch_norm2 = tf.layers.batch_normalization(deconv_2, training=is_train)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='VALID')
        batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding='SAME')
        batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

        logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 5, 2, padding='SAME')

        out = tf.tanh(logits)

        return out


def model_loss(input_real, input_z, out_channel_dim):
    label_smoothing = 0.9

    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake) * label_smoothing))

    return d_loss, g_loss, d_logits_real, d_logits_fake


def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def show_generator_output(sess, n_images, input_z, out_channel_dim, step, epoch):
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})
    for sample in samples:
        plt.imshow((sample * 255).astype(np.uint8))
    fig = plt.savefig(f'../faces/face-{epoch}_{step}.png')
    plt.close(fig)


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape):
    d_logits_real = np.array([])
    d_logits_fake = np.array([])

    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss, d_logits_r, d_logits_f = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    d_logits_real = np.append(d_logits_real, d_logits_r)
    d_logits_fake = np.append(d_logits_fake, d_logits_f)

    discriminator_loss = np.array([])
    generator_loss = np.array([])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('Up and running!')
        for epoch_i in range(epoch_count):
            steps = 0
            for batch_images in image_resizing.resizer.get_batches(batch_size):

                batch_images = batch_images * 2
                steps += 1

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})

                if steps % 400 == 0:
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    logit_r = d_logits_r.eval({input_z: batch_z, input_real: batch_images})

                    d_logits_real = np.append(d_logits_real, logit_r)

                    discriminator_loss = np.append(discriminator_loss, train_loss_d)
                    generator_loss = np.append(generator_loss, train_loss_g)

                    print("Epoch {}/{}. {:.3f}".format(epoch_i + 1, epochs, steps/(200000.0/16.0)),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                    show_generator_output(sess, 1, input_z, data_shape[3], steps, epoch_i+1)

            os.mkdir(f'../models/m_{epoch_i+1}/')
            save_path = saver.save(sess, f'../models/m_{epoch_i+1}/m_{epoch_i+1}.ckpt')
            print(f'Saving model in epoch {epoch_i+1} on path {save_path}')

            if epoch_i == 1 or epoch_i == 3 or epoch_i == 7:
                plt.plot(discriminator_loss, label='descriminator loss')
                plt.plot(generator_loss, label='generator loss')
                plt.legend()
                plt.savefig(f'../losses/loss-{epoch_i+1}.png')
                plt.show()

        os.mkdir(f'../models/m_final/')
        save_path = saver.save(sess, f'../models/m_final/m_final.ckpt')
        print(f'Saving final model on path {save_path}')

    return discriminator_loss, generator_loss, d_logits_real, d_logits_fake

batch_size = 16
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5
epochs = 8

with tf.Graph().as_default():
    desc_loss, gen_loss, d_logits_real, d_logits_fake = \
        train(epochs, batch_size, z_dim, learning_rate, beta1, image_resizing.resizer.get_batches,
          image_resizing.resizer.shape)


print(desc_loss)
print(gen_loss)
plt.plot(desc_loss, label='descriminator loss')
plt.plot(gen_loss, label='generator loss')
plt.legend()
plt.savefig("../losses/loss-final.png")
plt.show()

# print(np.array(d_logits_real.data))
# print(d_logits_fake)
# plt.plot(d_logits_real, label='descriminator logits real')
# plt.savefig('d-logits-real.png')
# plt.show()
# plt.plot(d_logits_fake, label='descriminator logits real')
# plt.savefig('d-logits-fake.png')
# plt.show()