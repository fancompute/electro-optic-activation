import numpy as np
import scipy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.python.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append("../neurophox")

from neurophox.tensorflow import RM
from neurophox.ml.nonlinearities import cnormsq

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns

from collections import namedtuple

def plot_confusion_matrix(cm, ax=None, figsize=(4,4), title=None, norm_axis=1, normalize=True):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=norm_axis)[:, np.newaxis]
        print("Acc = %.4f" % np.mean(np.diag(cm)))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    mask1 = np.eye(10) == 0
    mask2 = np.eye(10) == 1
    pal1 = sns.blend_palette(["#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"], as_cmap=True)
    pal2 = sns.blend_palette(["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"], as_cmap=True)
    sns.heatmap(100*cm,
                fmt=".1f",
                annot=True,
                cmap=pal1,
                linewidths=0,
                cbar=False,
                mask=mask1,
                ax=ax,
                square=True,
                linecolor="#ffffff", 
                annot_kws={"size": "small"})
    sns.heatmap(100*cm,
                fmt=".1f",
                annot=True,
                cmap=pal2,
                linewidths=0,
                cbar=False,
                mask=mask2,
                ax=ax,
                square=True,
                linecolor="#ffffff", 
                annot_kws={"size": "small"})
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if title is not None:
        ax.set_title(title)
    
    return fig


def plot_model_comparison(histories, 
                          labels,
                          figsize=(3.5,3.5),
                          axs=None,
                          ylim_acc=[40,100],
                          height_ratios=[1,0.33]):
    N = len(histories)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    if axs is None:
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = GridSpec(2, 1, figure=fig, height_ratios=height_ratios)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        axs = [ax0, ax1]
        
    for i in range(0, N):
        c1 = colors[2*i]
        c2 = colors[2*i+1]
        epoch_cnt = range(1, len(histories[i].history['loss']) + 1)
        acc     = [i*100 for i in histories[i].history['accuracy']]
        val_acc = [i*100 for i in histories[i].history['val_accuracy']]
        axs[0].plot(epoch_cnt, acc, "--", color=c2)
        axs[0].plot(epoch_cnt, val_acc, "-", color=c1, label=labels[i])
        axs[1].plot(epoch_cnt, histories[i].history['loss'], "-", color=c1, label=labels[i])

    axs[1].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    axs[0].set_ylim(ylim_acc)
    axs[0].legend(fontsize='small')
    axs[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f\%%'))
    
    fig.align_labels()
    
    return fig, axs


class EOIntensityModulation(tf.keras.layers.Layer):
    def __init__(self,
                 N,
                 alpha=0.1,
                 g=np.pi,
                 phi_b=np.pi,
                 train_alpha=False,
                 train_g=False,
                 train_phi_b=False,
                 single_param_per_layer=True):
        '''
        Implements the electro-optic intensity modulation activation function for tensorflow/Keras. This 
        activation function is described in more detail in the following paper:

            I. A. D. Williamson, T. W. Hughes, M. Minkov, B. Bartlett, S. Pai, and S. Fan, "Reprogrammable
            Electro-Optic Nonlinear Activation Functions for Optical Neural Networks," arXiv:1903.04579,
            Mar. 2019. <https://arxiv.org/abs/1903.04579>
        
        train_alpha, train_g, and train_phi_b specify whether the parameters should be trained. If false,
        they are fixed at the supplied values. If true, they are initialized to the supplied values and
        trained.
        
        single_param_per_layer specifies whether a single value for g, alpha, and phi_b should be used per
        layer. If false, then the parameters are element-wise.        
        '''
        
        
        super(EOIntensityModulation, self).__init__()
        
        if single_param_per_layer:
            var_shape = [1]
        else:
            var_shape = [N]
        
        self.g     = self.add_variable(shape=var_shape,
                                       name="g",
                                       initializer=tf.constant_initializer(g),
                                       trainable=train_g,
                                       constraint=lambda x: tf.clip_by_value(x, 1e-3, 1.5*np.pi))
        self.phi_b = self.add_variable(shape=var_shape,
                                       name="phi_b",
                                       initializer=tf.constant_initializer(phi_b),
                                       trainable=train_phi_b,
                                       constraint=lambda x: tf.clip_by_value(x, -np.pi, +np.pi))
        self.alpha = self.add_variable(shape=var_shape,
                                       name="alpha",
                                       initializer=tf.constant_initializer(alpha),
                                       trainable=train_alpha,
                                       constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99))
    
    def call(self, inputs):
        alpha, g, phi_b = tf.complex(self.alpha, 0.0), tf.complex(self.g, 0.0), tf.complex(self.phi_b, 0.0)
        Z = inputs
        return 1j * tf.sqrt(1-alpha) * tf.exp(-1j*0.5*g*tf.math.conj(Z)*Z - 1j*0.5*phi_b) * tf.cos(0.5*g*tf.math.conj(Z)*Z + 0.5*phi_b) * Z


def construct_onn_linear_tf(N, N_classes=10, L=1, theta_init_name='haar_rect', phi_init_name='random_phi'):
    '''
    Constructs an L layer linear ONN model with the specified alpha, g, and phi_b
    '''
    layers=[]
    
    for i in range(0,L):
        layers.append(RM(N, theta_init_name=theta_init_name, phi_init_name=phi_init_name))
    
    layers.append(Activation(cnormsq))
    layers.append(Lambda(lambda x: tf.math.real(x[:, :N_classes])))
    
    return tf.keras.models.Sequential(layers)

    
def construct_onn_EO_tf(N,
                        N_classes=10,
                        L=2,
                        train_alpha=False,
                        train_g=False,
                        train_phi_b=False,
                        single_param_per_layer=True,
                        theta_init_name='haar_rect',
                        phi_init_name='random_phi',
                        alpha=0.1,
                        g=0.05*np.pi,
                        phi_b=1*np.pi):
    '''
    Constructs an L layer EO ONN model with the specified alpha, g, and phi_b
    
    The parameters alpha, g, and phi_b can be scalars or vectors. If scalar-valued, the value is used
    for all L layers. If vector-valued, the length must be equal to L and each vector element is used 
    for the corresponding layer.
    '''
    alpha = np.asarray(alpha)
    g     = np.asarray(g)
    phi_b = np.asarray(phi_b)
    
    if alpha.size == 1:
        alpha = np.tile(alpha, L)
    else:
        assert alpha.size == L, 'alpha has a size which is inconsistent with L'
    
    if g.size == 1:
        g = np.tile(g, L)
    else:
        assert g.size == L, 'g has a size which is inconsistent with L'
    
    if phi_b.size == 1:
        phi_b = np.tile(phi_b, L)
    else:
        assert phi_b.size == L, 'phi_b has a size which is inconsistent with L'
    
    layers=[]
    for i in range(L):
        layers.append(RM(N, theta_init_name=theta_init_name, phi_init_name=phi_init_name))
        layers.append(EOIntensityModulation(N,
                                            alpha[i],
                                            g[i],
                                            phi_b[i],
                                            train_alpha=train_alpha,
                                            train_g=train_g,
                                            train_phi_b=train_phi_b,
                                            single_param_per_layer=single_param_per_layer))
    
#     layers.append(Activation(cnormsq))
#     keep = N // N_classes
    
#     layers.append(Lambda(lambda x: tf.math.real(tf.reduce_sum(
#         tf.reshape(x[:, :keep * N_classes], shape=(-1, N_classes, keep)), -1))))

    layers.append(Activation(cnormsq))
    layers.append(Lambda(lambda x: tf.math.real(x[:, :N_classes])))
    
    return tf.keras.models.Sequential(layers)


def calc_confusion_matrix_tf(model, x_test_norm, y_test_onehot):
    y_truth = y_test_onehot.argmax(axis=-1)
    y_pred  = model.predict(x_test_norm).argmax(axis=-1)
    cf_matrix = tf.math.confusion_matrix(y_truth, y_pred).numpy()
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=0)[:, np.newaxis]
    return cf_matrix

def value_to_one_hot(val, N):
    one_hot = np.zeros((N,))
    one_hot[int(val)] = 1
    return one_hot

def argmax_to_class(outputs):
    N_examples = outputs.shape[1]
    out_class = np.zeros((N_examples,))
    for i in range(N_examples):
        out_class[i] = np.argmax(outputs[:, i]) 
    return out_class


def norm_inputs(inputs, feature_axis=1):
    if feature_axis == 1:
        n_features, n_examples = inputs.shape
    elif feature_axis == 0:
        n_examples, n_features = inputs.shape
    for i in range(n_features):
        l1_norm = np.mean(np.abs(inputs[i, :]))
        inputs[i, :] /= l1_norm
    return inputs

ONNData = namedtuple('ONNData', ['x_train', 'y_train', 'x_test', 'y_test', 'units', 'num_classes'])

class MNISTDataProcessor:
    def __init__(self):
        (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = mnist.load_data()
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.x_test_raw.shape[0]
        self.x_train_ft = np.fft.fftshift(np.fft.fft2(self.x_train_raw), axes=(1, 2))
        self.x_test_ft = np.fft.fftshift(np.fft.fft2(self.x_test_raw), axes=(1, 2))
        
    def fourier(self, freq_radius):
        min_r, max_r = 14 - freq_radius, 14 + freq_radius
        x_train_ft = self.x_train_ft[:, min_r:max_r, min_r:max_r]
        x_test_ft = self.x_test_ft[:, min_r:max_r, min_r:max_r]
        return ONNData(
            x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1))).astype(np.complex64),
            y_train=np.eye(10)[self.y_train],
            x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1))).astype(np.complex64),
            y_test=np.eye(10)[self.y_test],
            units=freq_radius**2,
            num_classes=10
        )
