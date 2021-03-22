"""
Conditional GAN for simple 1D simulations, version 2
    Modified from conditional DCGAN,
    so 'images' in the code do not actually mean images, rather mean 'actions' or 'mission plans'
    /home/tanaka/tf2/GAN/conditional_dcgan

"""
from simple_1D_simulator_for_gan import Fighter, Jammer, SAM
from simple_1D_simulator_for_gan import DataBuffer, summarize_simulation_results
from utils.data_maker import data_maker, make_positive_dataset, balance_training_data
from utils.mission_results_analyzer import simulation_results_to_tensorboard

import numpy as np
import tensorflow as tf
import os
import joblib
import wandb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

"""
Change the neme of model, otherwise model is overwritten.
"""
FINAL_ACTIVATION = 'sigmoid'  # maybe 'sigmoid' or 'relu'

PROJECT = 'Rand report 1D GAN'
MODEL_NAME = 'model-' + FINAL_ACTIVATION + '-selected-samples'

FIXED_Z_FOR_EVAL = False  # If True, use fixed z for evaluation, otherwise z~N(0,I)

BALANCE_SAMPLES = True  # If True, balance w1, w2, w3 in training dataset
NUM_BALANCE_SAMPLES = 15000  # Number of training samples for balance selected each from w1, w2, w3


def conditional_gan_test(y_test, generator, z=None):
    blue_win = DataBuffer()
    blue_not_win = DataBuffer()

    success_count = 0
    not_success_count = 0

    # Do simulation for each scenario
    for y in y_test:
        ### Transform data type
        y_input = tf.expand_dims(tf.cast(y, tf.float32), axis=0)

        ### Sample from N(0,1)
        if z is None:
            z = np.random.randn(1, CODING_SIZE)
        gen_images = generator([z, y_input])

        ### Recover the original metric [km]
        fighter.firing_range = y[0] * fighter.max_firing_range
        sam.offset = y[1] * sam.max_offset
        sam.firing_range = y[2] * sam.max_firing_range
        sam.jammed_firing_range = y[3] * sam.max_firing_range * jammer.jam_effectiveness

        fighter.ingress = gen_images[0, 0].numpy() * sam.max_offset
        jammer.ingress = gen_images[0, 1].numpy() * sam.max_offset

        """ Perform simulation """
        # Blue team win without using jammer
        blue_win_condition_1 = (fighter.ingress < sam.offset - sam.firing_range) and \
                               (jammer.ingress < sam.offset - sam.firing_range) and \
                               (fighter.firing_range > sam.offset - fighter.ingress)

        # Blue team wins with using jammer
        blue_win_condition_2 = (jammer.jam_range > sam.firing_range) and \
                               (fighter.ingress < sam.offset - sam.jammed_firing_range) and \
                               (jammer.ingress + jammer.jam_range > sam.offset) and \
                               (jammer.ingress < sam.offset - sam.jammed_firing_range) and \
                               (fighter.firing_range > sam.offset - fighter.ingress)

        """ Classify simulation results """
        if blue_win_condition_1 or blue_win_condition_2:
            success_count += 1
            blue_win.result_type_classifier(fighter, jammer, sam)
        else:
            not_success_count += 1
            blue_not_win.result_type_classifier(fighter, jammer, sam)

    """ Summarize the results and data """
    blue_win_array, blue_not_win_array, whole_array = \
        summarize_simulation_results(blue_win, blue_not_win, success_count, not_success_count)

    return blue_win_array, blue_not_win_array, whole_array


class GAN(object):
    """
    GANを定義
    """

    def __init__(self, coding_size):
        self.coding_size = coding_size
        self.action_size = 2
        self.condition_size = 4

    def create_generator(self):
        """
        Define the Generator
        """
        g1 = Input(shape=(self.coding_size,))  # (None,100)
        c1 = Input(shape=(self.condition_size,))  # (None,5)
        cond_in = Concatenate(axis=-1)([g1, c1])  # (None,105)

        g2 = Dense(units=HIDDEN_UNITS, activation='relu')(cond_in)  # (None,128)
        g3 = BatchNormalization()(g2)  # (None,128)
        g4 = Dense(units=HIDDEN_UNITS, activation='relu')(g3)  # (None,128)
        # g5 = Dense(units=self.action_size, activation='linear')(g4)  # (None,2)
        g5 = Dense(units=self.action_size, activation=FINAL_ACTIVATION)(g4)  # (None,2)

        generator = Model([g1, c1], g5, name='generator')
        return generator

    def create_discriminator(self):
        """
        Define the Discriminator
        """
        d1 = Input(shape=(self.action_size,))  # (None,2)
        c1 = Input(shape=(self.condition_size,))  # (None,5)
        cond_in = Concatenate(axis=-1)([d1, c1])
        d2 = Dense(units=HIDDEN_UNITS, activation='relu')(cond_in)  # (None,128)
        d3 = Dropout(rate=0.5)(d2)  # (None,128)
        d4 = Dense(units=HIDDEN_UNITS, activation='relu')(d3)  # (None,128)
        d5 = Dense(units=1, activation='sigmoid')(d4)  # (None,1)

        discriminator = Model([d1, c1], d5, name='discriminator')
        return discriminator

    def create_gan(self):
        # Create the Generator
        generator = self.create_generator()

        # Create the Discriminator
        discriminator = self.create_discriminator()
        optimizer = Adam(learning_rate=DISCRIMINATOR_LR)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Create the GAN
        noise_input = Input(shape=(self.coding_size,))  # (None,100)
        cond_input = Input(shape=(self.condition_size,))  # (None,5)
        f_image = generator([noise_input, cond_input])  # (None,2)
        gan_output = discriminator([f_image, cond_input])  # (None,1)

        gan = Model([noise_input, cond_input], gan_output, name='gan')
        optimizer = Adam(learning_rate=GAN_LR)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return gan, generator, discriminator


def make_summary_of_gan(generator, discriminator, gan, model_name):
    print('\n')
    generator.summary()
    to_file = 'architecture_plot/' + model_name + '_generator.png'
    plot_model(generator, show_shapes=True, to_file=to_file)

    print('\n')
    discriminator.summary()
    to_file = 'architecture_plot/' + model_name + '_discriminator.png'
    plot_model(discriminator, show_shapes=True, to_file=to_file)

    print('\n')
    gan.summary()
    to_file = 'architecture_plot/' + model_name + '_gan.png'
    plot_model(gan, show_shapes=True, to_file=to_file)
    print('\n')


def train_gan():
    """ Some preparation """
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    evaluation_results = []
    test_results = []

    epoch = 0

    """ For wandb """
    model_name = MODEL_NAME
    config = {"coding_size": CODING_SIZE, "discriminator lr": DISCRIMINATOR_LR,
              "gan (generator) rl": GAN_LR, "hidden units": HIDDEN_UNITS, "random seed": RANDOM_SEED,
              "final activation of generator": FINAL_ACTIVATION,
              "balance w1, w2, w3 for training": BALANCE_SAMPLES,
              "number of balanced samples, each": NUM_BALANCE_SAMPLES,
              "use fixed z for evaluation": FIXED_Z_FOR_EVAL}
    wandb.init(name=model_name, project=PROJECT, sync_tensorboard=True)
    wandb.config.update(config)

    """ For Tensorboard """
    train_log_dir = 'logs/' + model_name
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    evaluation = 'Evaluation'
    test = 'Test'

    # Set hyper_parameters
    max_epochs = MAX_EPOCHS
    batch_size = BATCH_SIZE

    """
    ミニバッチを生成するトレーニング用データセットと評価用、テスト用データセットの作成
    """
    # Training data: Blue tem won data in 1D simulation
    X_train = training_data[1]  # actions(plans)
    y_train = training_data[0]  # conditions

    my_data = (X_train, y_train)
    dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    # Evaluation data: Not related to 1D simulation, and z is fixed
    y_eval = evaluation_data[0]  # whole conditions
    # y_positive_eval = positive_evaluation_data[0]  # conditions for potential win of blue team

    # Fixed z for evaluation
    if FIXED_Z_FOR_EVAL:
        z_for_eval = np.random.randn(1, CODING_SIZE)
    else:
        z_for_eval = None  # z will be selected as z ~ N(0, I)

    # Test data: Not related to 1D simulation
    y_test = test_data[0]  # whole conditions
    # y_positive_test = positive_test_data[0]  # conditions for potential win of blue team

    """
    Create the GAN
    """
    coding_size = CODING_SIZE  # Dimension of z
    gan, generator, discriminator = GAN(coding_size).create_gan()

    make_summary_of_gan(generator, discriminator, gan, model_name)

    """
    Initial evaluation
    """
    print(f'\n******************** Now evaluation is started at epoch {epoch} ********************')
    blue_win, blue_not_win, whole = conditional_gan_test(y_eval, generator, z_for_eval)
    evaluation_results.append([epoch, blue_win, blue_not_win, whole])
    simulation_results_to_tensorboard(blue_win, blue_not_win, whole, summary_writer, epoch, evaluation)

    """
    Initial test
    """
    print(f'\n******************** Now test is started at epoch {epoch} ********************')
    blue_win, blue_not_win, whole = conditional_gan_test(y_test, generator)
    test_results.append([epoch, blue_win, blue_not_win, whole])
    simulation_results_to_tensorboard(blue_win, blue_not_win, whole, summary_writer, epoch, test)

    """
    Train the GAN
    """
    while True:
        epoch += 1
        print(f'Current epoch = {epoch}')
        d_loss = 0
        g_loss = 0
        d_accuracy = 0
        g_accuracy = 0

        for (X_batch, y_batch) in dataset:
            # データの型を変換
            X_batch = tf.cast(X_batch, tf.float32)
            y_batch = tf.cast(y_batch, tf.float32)

            """
            Train the discriminator
            """
            discriminator.trainable = True
            generator.trainable = False

            # Sample from N(0,1)
            z = np.random.randn(batch_size, coding_size)  # (256,100)
            gen_images = generator([z, y_batch])  # (256,2), float32

            # Make the dataset for the training
            X_fake_vs_real = tf.concat([gen_images, X_batch], axis=0)  # (512,2)
            y_fake_vs_real = \
                tf.concat([tf.zeros(batch_size), tf.ones(batch_size)], axis=0)  # (512,)
            y_fake_vs_real = tf.expand_dims(y_fake_vs_real, axis=1)  # (512,1)
            cond_label = tf.concat([y_batch, y_batch], axis=0)  # (512,5)

            # Train the discriminator
            d = discriminator.fit(x=[X_fake_vs_real, cond_label], y=y_fake_vs_real, epochs=1, verbose=0)
            d_loss += d.history['loss'][0]
            d_accuracy += d.history['accuracy'][0]

            """
            Train the generator
            """
            discriminator.trainable = False
            generator.trainable = True

            # Sample from N(0,1)
            noise = np.random.randn(batch_size, coding_size)  # (256,100)
            lab = y_batch
            y_gen = tf.ones(batch_size)  # (256,)
            y_gen = tf.expand_dims(y_gen, axis=1)  # (256,1)

            # Train the generator
            g = gan.fit(x=[noise, lab], y=y_gen, epochs=1, verbose=0)
            g_loss += g.history['loss'][0]
            g_accuracy += g.history['accuracy'][0]

        # Check performance of training GAN
        if epoch % EVAL_INTERVAL == 0:
            print(f'******************** Now evaluation is started at epoch {epoch} ********************')
            blue_win, blue_not_win, whole = conditional_gan_test(y_eval, generator, z_for_eval)
            evaluation_results.append([epoch + 1, blue_win, blue_not_win, whole])
            simulation_results_to_tensorboard(
                blue_win, blue_not_win, whole, summary_writer, epoch, evaluation)

            # Save results
            file_name = 'evaluation/' + model_name + '_results.txt'
            joblib.dump(evaluation_results, file_name)

        if epoch % TEST_INTERVAL == 0:
            print(f'******************** Now test is started at epoch {epoch} ********************')
            blue_win, blue_not_win, whole = conditional_gan_test(y_test, generator)
            test_results.append([epoch + 1, blue_win, blue_not_win, whole])
            simulation_results_to_tensorboard(
                blue_win, blue_not_win, whole, summary_writer, epoch, test)

            # Save results
            file_name = 'test/' + model_name + '_results.txt'
            joblib.dump(test_results, file_name)

        if epoch % SAVE_INTERVAL == 0:
            gan_name = model_name + '/my_gan_model_' + str(epoch)
            gan.save(gan_name)

            generator_name = model_name + '/my_generator_model_' + str(epoch)
            generator.save(generator_name)

            discriminator_name = model_name + '/my_discriminator_model_' + str(epoch)
            discriminator.save(discriminator_name)

        print(f'd_loss = {d_loss}, d_accuracy = {d_accuracy}, g_loss = {g_loss}, g_accuracy = {g_accuracy}')

        ### For tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('discriminator_loss', d_loss, step=epoch)
            tf.summary.scalar('discriminator_accuracy', d_accuracy, step=epoch)
            tf.summary.scalar('gan_loss', g_loss, step=epoch)
            tf.summary.scalar('gan_accuracy', g_accuracy, step=epoch)
        summary_writer.flush()

        # wandb.log({'d_loss': d_loss, 'd_accuracy': d_accuracy, 'g_loss': g_loss, 'g_accuracy': g_accuracy})

        if epoch > max_epochs:
            break

    # Save the learnt models
    gan.save(model_name + '/my_gan_model')
    generator.save(model_name + '/my_generator_model')
    discriminator.save(model_name + '/my_discriminator_model')

    # Save results
    file_name = 'evaluation/' + model_name + '_results.txt'
    joblib.dump(evaluation_results, file_name)
    temp = joblib.load(file_name)
    print(f'Evaluation results is successfully saved in {file_name}')

    file_name = 'test/' + model_name + '_results.txt'
    joblib.dump(test_results, file_name)
    temp = joblib.load(file_name)
    print(f'Test results is successfully saved in {file_name}')

    print('Conglaturations!  Finally, max epoch is achieved !')


if __name__ == "__main__":
    """ Define hyper parameters """
    RANDOM_SEED = 6000

    MAX_EPOCHS = 2000  # 2000
    BATCH_SIZE = 256
    EVAL_INTERVAL = 10  # epochs
    TEST_INTERVAL = 10  # epochs
    SAVE_INTERVAL = 10  # epochs

    CODING_SIZE = 20  # Dim of latent space
    DISCRIMINATOR_LR = 1e-5
    GAN_LR = 5e-6
    HIDDEN_UNITS = 32  # All dense layer have same hidden units, just for simplicity

    """ Make necessary directories, if not exist """
    file_list = [MODEL_NAME, "logs", "architecture_plot", "evaluation", "test"]
    for file in file_list:
        filedir = os.path.join(os.getcwd(), file)
        if not os.path.exists(filedir):
            os.mkdir(filedir)

    # Instanciate the agents
    fighter = Fighter()
    jammer = Jammer()
    sam = SAM()

    # Load and make dataset for training, evaluation, and test
    training_data, evaluation_data, test_data = data_maker(fighter, jammer, sam)

    # make positive dataset for evaluation and test
    # print('\n\n   * positive evaluation data (Potential blue team win (w1, w2, w3))')
    # positive_evaluation_data = make_positive_dataset(evaluation_data)

    # print('\n   * positive test data (Potential blue team win (w1, w2, w3))')
    # positive_test_data = make_positive_dataset(test_data)

    """ Select w1, w2, w3 from training dataset, if specified by BALANCE_SAMPLES"""
    if BALANCE_SAMPLES:
        training_data = balance_training_data(training_data, NUM_BALANCE_SAMPLES)

    """ Train, evaluate, and test GAN """
    print('\n\n==================== Now start training of GAN ====================')
    train_gan()
