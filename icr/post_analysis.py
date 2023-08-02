import os
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss
from icr.tools import alpha_to_class

FIG_OUT_DIR = os.path.join('log', 'figures')
os.makedirs(FIG_OUT_DIR, exist_ok=True)

# def post_analysis(y_post: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, name: str):
#     pred_analysis(y_pred, y_true, f'{name}_before')
#     pred_analysis(y_post, y_true, f'{name}_after')
#     return

def pred_analysis(
        y_pred: np.ndarray,
        alpha_true: np.ndarray,
        title: str,
        out_dir: str = FIG_OUT_DIR
    ):
    fig = _pred_analysis(y_pred, alpha_true)
    fig.savefig(os.path.join(out_dir, f'{title}.png'))
    fig.clear()
    fig.clf()
    plt.close(fig)
    return

def _pred_analysis(y_pred: np.ndarray, alpha_true: np.ndarray):
    y_true = alpha_to_class(alpha_true)
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
    _draw(axs[1], 1, ones, all_loss, len(y_true), alpha_true[y_true == 1])
    return fig

def _draw(
        ax: plt.Axes,
        class_label,
        class_probs,
        all_loss,
        n_samples: int,
        alpha_labels: np.ndarray = None,
    ):

    scatter = np.linspace(0, 1, len(class_probs))

    # ax.set_xscale('log')
    # ax.set_xlim(1e-15, 1e-0)
    ax.set_xlim(0, 1)
    if class_label == 0:
        ax.scatter(class_probs, scatter, s=15, color='blue')
    else:
        assert alpha_labels is not None
        alphas = ['B', 'D', 'G']
        colors = ['red', 'green', 'cyan']
        for i, color, alpha in zip(range(1, 4), colors, alphas):
            indices = alpha_labels == i
            ax.scatter(class_probs[indices], scatter[indices], s=15, color=color, label=alpha)
        ax.legend()

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
    
    title = (
        ('zeros' if class_label == 0 else 'ones')
        #+ f' loss(p<.1): {100 * small_losses / all_loss:.1f}%, l={small_unnorm_loss:.3f}'
    )
    ax.set_title(title)
    if small_count > 0:
        small_unnorm_loss = small_losses * n_samples / small_count
        ax.annotate(
            f'loss(p<.1): {100 * small_losses / all_loss:.1f}%,\nl={small_unnorm_loss:.3f}',
            (.0 if class_label == 0 else .9, 1.05),
        )
    return

# y_pred = np.random.random(10)
# y_true = (y_pred > .5).astype(int)
# post_analysis(y_pred, y_true)
