"""Images logging."""

import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from alpacka.data import nested_unzip


def concatenate_images(images):
    """Concatenates stacked images into single one."""
    images = np.concatenate(images, axis=-3)
    images = np.concatenate(images, axis=-2)
    return images


def images2grid(images):
    """Converts sparse grid of images to single image."""
    max_y = max(y for y, _ in images)
    max_x = max(x for _, x in images)
    image_shape = next(iter(images.values())).shape
    stacked_images = np.ones((max_x, max_y) + image_shape)
    for pos, image in images.items():
        y, x = pos[0] - 1, pos[1] - 1
        stacked_images[x, y] = image / 255

    images_grid = concatenate_images(stacked_images)
    return (255 * images_grid).astype(np.uint8)


def fig2rgb(figure, dpi=None):
    """Converts the matplotlib plot specified by 'figure' to a PNG image.

    The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, format='png')
    plt.close(figure)
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=3).numpy()


def obs_rgb2fig(images, info_bottom=None, info_top=None):
    """Return a Nx1 grid of images as a matplotlib figure."""
    # Create a figure to contain the plot.
    fig, axs = plt.subplots(
        1, len(images), figsize=(2 * len(images), 6)
    )

    # First board is the current board.
    captions_bottom = ['current_state']
    if info_bottom is None:
        captions_bottom = ['' for _ in images]
    else:
        for image_info in nested_unzip(info_bottom):
            caption = ''
            for name, value in image_info.items():
                if isinstance(value, str):
                    caption += f'{name:>14}= {value}\n'
                else:
                    caption += f'{name:>14}={value: .5f}\n'
            # Remove newline at the end of the string.
            caption = caption.rstrip('\n')
            captions_bottom.append(caption)

    captions_top = []
    if info_top is None:
        captions_top = ['' for _ in images]
    else:
        for image_info in nested_unzip(info_top):
            caption = ''
            for name, value in image_info.items():
                if isinstance(value, str):
                    caption += f'{name:>14}= {value}\n'
                else:
                    caption += f'{name:>14}={value: .5f}\n'
            # Remove newline at the end of the string.
            caption = caption.rstrip('\n')
            captions_top.append(caption)

    for i, (image, caption_bottom, caption_top) in enumerate(
            zip(images, captions_bottom, captions_top)
    ):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].grid(False)
        axs[i].set_xlabel(
            caption_bottom, position=(0., 0), horizontalalignment='left',
            family='monospace', fontsize=7
        )
        axs[i].set_title(caption_top, family='monospace', fontsize=7)
        axs[i].imshow(image)

    return fig


def visualize_model_predictions(obs_rgb, captions_bottom, captions_top=None):
    fig = obs_rgb2fig(obs_rgb, captions_bottom, captions_top)
    return fig2rgb(fig)
