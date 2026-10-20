from typing import Optional
import os

import numpy as np
from matplotlib import pyplot as plt

from .model import Model
from .dataset import DataSample

_SAVE_FIG_DPI = 200


def _get_savefig_path(filename: str, folder: str = 'img', ext: str = 'png') -> str:
    if not folder:
        folder = '.'
    if not ext:
        ext = 'png'
    if ext.startswith('.'):
        ext = ext.lstrip('.')
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not filename.endswith(f'.{ext}'):
        filename = f'{filename}.{ext}'

    return os.path.join(folder, filename)


def plot_margins(model: Model, sample: DataSample, filename: Optional[str] = None):
    margins = model.margins if model.margins is not None else model.calc_margins_sample(sample)
    sorted_margins: list[np.float64] = sorted(list(margins))
    plot_x = np.array([i for i in range(len(sorted_margins))])

    for x, m in zip(plot_x, sorted_margins):
        if m <= -0.3:
            color = 'r'
        elif m >= 0.5:
            color = 'g'
        else:
            color = 'y'
        plt.plot([x, x], [m, 0], color, linewidth=0.5)
    plt.plot([plot_x[0], plot_x[-1]], [0, 0], 'gray')

    plt.plot(plot_x, sorted_margins)

    x_delta = plot_x[-1] - plot_x[0]
    m_delta = sorted_margins[-1] - sorted_margins[0]
    plt.axis([ # type: ignore
        plot_x[0] - round(x_delta * 0.125),
        plot_x[-1] + round(x_delta * 0.025),
        sorted_margins[0] - round(m_delta * 0.05),
        sorted_margins[-1] + round(m_delta * 0.05)
    ])

    plt.title('Sorted margins')
    plt.xlabel('i')
    plt.ylabel('margin')

    if filename:
        plt.savefig(_get_savefig_path(filename), dpi=_SAVE_FIG_DPI)
    else:
        plt.show()
    plt.close()


def plot_loss_history(model: Model, save_image: bool = True):
    if model.loss_history is None:
        raise Exception('Model must have loss history to plot it')
    train_history, test_history = model.loss_history.train, model.loss_history.test

    if len(train_history) * len(test_history) == 0:
        raise Exception('Both train_history and test_history must be not empty')
    
    plt.plot(range(len(train_history)), train_history, label='train')
    plt.plot(range(len(test_history)), test_history, label='test')
    
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title(f'{(model.name + ' ') if model.name else ''}Loss')
    
    plt.legend()
    if save_image:
        plt.savefig(_get_savefig_path(f'{model.name}_loss'), dpi=_SAVE_FIG_DPI)
    else:
        plt.show()
    plt.close()
