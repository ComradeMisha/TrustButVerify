"""Gif images from logged episodes."""
import argparse
import os
from collections import defaultdict
from collections import namedtuple

import imageio
import tensorflow as tf
from tensorflow.core.util import event_pb2  # pylint: disable=import-error, no-name-in-module

TensorBoardImage = namedtuple('TensorBoardImage', ['name', 'image', 'step'])


def extract_images_from_event(event_filename, image_tag_prefix):
    """Extracts images from tensorboard event."""
    topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())  # pylint: disable=no-member
        for v in event.summary.value:
            if v.tag.startswith(image_tag_prefix):
                if v.HasField('tensor'):  # event for images using tensor field
                    s = v.tensor.string_val[2]  # first elements are W and H

                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()

                    topic_counter[v.tag] += 1

                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(name=v.tag, image=np_img, step=cnt)

                    yield tbi


def generate_gifs(tb_event_path, output_dir,
                  image_tag_prefix='episode_model/rollout_epoch'):
    """Generates episodes gifs from tensorboard event images."""
    images = extract_images_from_event(tb_event_path, image_tag_prefix)
    previous_image_name = ''
    writer = None
    for tbi in images:
        if tbi.name != previous_image_name:
            if writer is not None:
                writer.close()
            gif_filename = tbi.name.replace('/', '_')
            gif_filepath = os.path.join(output_dir, f'{gif_filename}.gif')
            writer = imageio.get_writer(gif_filepath, mode='i', duration=0.5)
            previous_image_name = tbi.name
        writer.append_data(tbi.image)

    writer.close()


def main():
    """Script for generating gif images from agent's episodes."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--events_dir', type=str, default='out',
        help='Path to the root directory of Tensorboard events.'
    )
    parser.add_argument(
        '--event_name', type=str, default=None,
        help='Name of the Tensorboard event. Script will load Tensorboard '
             'event from directory {events_dir/event_name} and assumes that '
             'there is only one Tensorboard event in that directory. If it is '
             'not provided, then most recent Tensorboard event is loaded.')
    parser.add_argument(
        '--output_dir', type=str, default='out/gifs',
        help='Path to directory where gif images will be saved.'
    )
    args = parser.parse_args()

    if args.event_name is None:
        events_names = sorted(os.listdir(args.events_dir))
        event_name = events_names[-1]
    else:
        event_name = args.event_name

    tb_files = os.listdir(os.path.join(args.events_dir, event_name))
    assert len(tb_files) == 1
    tb_event_path = os.path.join(args.events_dir, event_name, tb_files[0])

    output_dir = os.path.join(args.output_dir, event_name)
    os.makedirs(output_dir, exist_ok=True)

    generate_gifs(tb_event_path, output_dir)


if __name__ == '__main__':
    main()
