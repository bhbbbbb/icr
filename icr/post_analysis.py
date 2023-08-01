import os
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss
# from icr.tools import balanced_log_loss

FIG_OUT_DIR = os.path.join('log', 'figures')
os.makedirs(FIG_OUT_DIR, exist_ok=True)

def post_analysis(y_post: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, name: str):
    pred_analysis(y_pred, y_true, f'{name}_before')
    pred_analysis(y_post, y_true, f'{name}_after')
    return

def pred_analysis(y_pred: np.ndarray, y_true: np.ndarray, title: str):
    fig = _pred_analysis(y_pred, y_true)
    fig.savefig(os.path.join(FIG_OUT_DIR, f'{title}.png'))
    fig.clear()
    fig.clf()
    plt.close(fig)
    return

def _pred_analysis(y_pred: np.ndarray, y_true: np.ndarray):
    ones = y_pred[y_true == 1]
    zeros = y_pred[y_true == 0]
    # class_weights = 1 / np.array([(y_true == 0).sum(), (y_true == 1).sum()])
    # ones_w = np.array([0, 1 / (y_true == 1).sum()])
    # zeros_w = np.array([1 / (y_true == 0).sum(), 0])
    ones_loss = .5 * log_loss(y_true[y_true == 1], ones, eps=1e-15, labels=[0, 1])
    zeros_loss = .5 * log_loss(y_true[y_true == 0], zeros, eps=1e-15, labels=[0, 1])
    all_loss = ones_loss + zeros_loss

    fig, axs = plt.subplots(2)
    fig.set_size_inches(18., 9.)
    fig.suptitle(
        f'loss: {all_loss: .4f}, '
        f'zeros: {zeros_loss / all_loss * 100: .1f}%, '
        f'ones: {100 * ones_loss / all_loss: .1f}%'
    )
    _draw(axs[0], 0, zeros, all_loss, len(y_true))
    _draw(axs[1], 1, ones, all_loss, len(y_true))
    return fig

def _draw(ax: plt.Axes, class_label, class_probs, all_loss, n_samples: int):

    scatter = np.linspace(0, 1, len(class_probs))
    # class_probs = 1 - class_probs if class_label == 1 else class_probs
    # scatter = [random.random() for _ in class_probs]

    # ax.set_xscale('log')
    # ax.set_xlim(1e-15, 1e-0)
    ax.set_xlim(0, 1)
    # ax.set_xlim(xmax=1e-0)
    ax.scatter(class_probs, scatter, s=15, color=('blue' if class_label == 0 else 'red'))
    w = .5 / len(class_probs)
    small_losses = 0.
    small_count = 0
    for i, p in enumerate(class_probs):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        p_eff = p if class_label == 0 else 1 - p
        # loss = w * (- math.log(p if class_label == 1 else 1 - p))
        loss = w * (- math.log(1 - p_eff))
        loss_unnorm = loss * n_samples
        if p_eff > .1:
            ax.annotate(
                f'{loss / all_loss * 100:.1f}%\nl={loss_unnorm:.3f}\np={p:.3f}',
                (class_probs[i], scatter[i])
            )
        else:
            small_losses += loss
            small_count += 1
    
    small_unnorm_loss = small_losses * n_samples / small_count
    title = (
        ('zeros' if class_label == 0 else 'ones')
        #+ f' loss(p<.1): {100 * small_losses / all_loss:.1f}%, l={small_unnorm_loss:.3f}'
    )
    ax.set_title(title)
    ax.annotate(
        f'loss(p<.1): {100 * small_losses / all_loss:.1f}%,\nl={small_unnorm_loss:.3f}',
        (.0 if class_label == 0 else .9, 1.05),
    )
    return

# y_pred = np.random.random(10)
# y_true = (y_pred > .5).astype(int)
# post_analysis(y_pred, y_true)
