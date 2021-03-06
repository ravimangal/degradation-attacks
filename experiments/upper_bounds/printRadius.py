import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from cachable.loaders import Loader
# from dbify import dbify
from scriptify import scriptify
import tensorflow_datasets as tfds
from time import time

from gloro import GloroNet

from experiments.training.utils import get_data
import csv
import pandas as pd
        

def print_radius(
        model,
        data):
    
    label = []
    predict = []
    radius = []
    correct = []
    print("start evaluation")
    for x,y in tfds.as_numpy(data):
        preds, eps, y_b = model.predict_with_certified_radius(x)
        
        yy = tf.argmax(preds, axis=1, output_type='int32')[:,None]
        yy = tf.reshape(yy, tf.shape(y))
        label.append(y)
        predict.append(yy)
        radius.append(eps)
        correct.append(yy.numpy() == y)
        # print("- label: ",y)
        # print("- predict: ",yy)
        # print("- radius: ", eps)
        # print("- correct: ", yy.numpy() == y)
    label = np.concatenate(label, axis=0)
    predict = np.concatenate(predict, axis=0)
    radius = np.concatenate(radius, axis=0)
    correct = np.concatenate(correct, axis=0)
    print("evaluation done")
    return label,predict,radius,correct

if __name__ == '__main__':
    @scriptify
    def script(
                dataset="cifar10",
                augmentation=None,
                batch_size=128,
                model="cifar10_0.14_N",
                using_gpu=False,
                gpu=0):
    
        if using_gpu:
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            device = gpus[gpu]
    
            for device in tf.config.experimental.get_visible_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
                
        train, test, metadata = get_data(dataset, batch_size, augmentation)
        print("loaded data")
        gloro_model = GloroNet.load_model(f'./models/gloro/{dataset}/{model}.gloronet')
        print("loaded model")
        
        label,predict,radius,correct = print_radius(gloro_model,test)
                
        df = pd.DataFrame({
               'label': label,
               'predict': predict,
               'radius': radius,
               'correct': correct})
        print("generated table")
        data_file = f'./data/gloro/{dataset}/{model}'
        df.to_csv(data_file, index=True)
