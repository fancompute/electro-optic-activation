import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns

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
        acc     = [i*100 for i in histories[i].history['acc']]
        val_acc = [i*100 for i in histories[i].history['val_acc']]
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