import numpy as np
import cv2
import tensorflow as tf


def put_text(imgs, texts, offset_h=0, offset_w=30, color='blue'):
    if isinstance(color, bytes):
        color = color.decode()
    color_map = {
        'blue': (0, 0, 1),
        'red': (1, 0, 0),
        'green': (0, 1, 0)
    }
    result = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        text = texts[i]
        if isinstance(text, bytes):
            text = text.decode()
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
        result[i, :, :, :] = cv2.putText(imgs[i, :, :, :], str(text)[:15], (offset_h, offset_w), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_map[color], 1)
    return result


def put_static_text(imgs, static_text, offset_h=0, offset_w=30, color='blue'):
    if isinstance(color, bytes):
        color = color.decode()
    color_map = {
        'blue': (0, 0, 1),
        'red': (1, 0, 0),
        'green': (0, 1, 0)
    }
    result = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        text = static_text
        if isinstance(text, bytes):
            text = text.decode()
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
        result[i, :, :, :] = cv2.putText(imgs[i, :, :, :], str(text), (offset_h, offset_w), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_map[color], 1)
    return result


def tf_put_text(imgs, texts, offset_h, offset_w, color, static=False):
    if static:
        return tf.py_func(put_static_text, [imgs, texts, offset_h, offset_w, color], Tout=imgs.dtype)
    else:
        return tf.py_func(put_text, [imgs, texts, offset_h, offset_w, color], Tout=imgs.dtype)
