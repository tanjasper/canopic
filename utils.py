import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_plot(accs, plot_vals, savename, scale=1, stride=1):

    f = plt.figure()

    # print each plot_val
    for i in range(len(plot_vals)):
        xp = range(len(accs[plot_vals[i]][0::stride]))
        plt.plot(xp, [z * scale for z in accs[plot_vals[i]][0::stride]])
    plt.legend(plot_vals)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    f.savefig(savename, bbox_inches='tight')


def save_accuracy_plot(accs, savename, legend, scale=1, stride=1, ylim=None):

    f = plt.figure()

    # print each plot_val
    for i in range(len(accs)):
        xp = range(len(accs[i][0::stride]))
        plt.plot(xp, [z * scale for z in accs[i][0::stride]])
    plt.legend(legend)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    f.savefig(savename, bbox_inches='tight')
    plt.close(f)


def adjust_learning_rate(lr, epoch, decrease_lr_freq, optimizer):
    curr_lr = lr * 0.1 ** (epoch // decrease_lr_freq)  # decrease learning rate by 0.1 every decrease_lr_freq epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = curr_lr