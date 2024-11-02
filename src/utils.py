# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def get_args():
    parse = argparse.ArgumentParser(description='Football Jerseys')
    parse.add_argument('-p', '--data_path', type=str, default='../../data/VNCurrency')
    parse.add_argument('-b', '--batch_size', type=int, default=32)
    parse.add_argument('-e', '--epochs', type=int, default=22)
    parse.add_argument('-l', '--lr', type=float, default=1e-3) #for adam
    parse.add_argument('-w', '--num_workers', type=int, default=os.cpu_count())
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None) #None = train tu dau
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--VNCur_trained_models', type=str, default="VNCur_trained_models")
    args = parse.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="plasma")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
