# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parse = argparse.ArgumentParser(description='Football Jerseys')
    parse.add_argument('-p', '--data_path', type=str, default='./data/animals/')
    parse.add_argument('-p2', '--data_path2', type=str, default='./data/animals/')
    parse.add_argument('-b', '--batch_size', type=int, default=32)
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument('-l', '--lr', type=float, default=1e-2)
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None) #None = train tu dau
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--trained_models', type=str, default="trained_models")
    args = parse.parse_args()
    return args
