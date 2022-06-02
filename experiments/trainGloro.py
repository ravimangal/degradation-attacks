import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from cachable.loaders import Loader
# from dbify import dbify
from scriptify import scriptify
import tensorflow_datasets as tfds
from time import time

from gloro import GloroNet
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import TradesScheduler
from gloro.training.callbacks import LrScheduler
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra
from gloro.training.metrics import rejection_rate
from gloro.training import losses

from training.utils import get_data, get_optimizer
import training.architectures as architectures

class GloroLoader(Loader):

    def load(self, filename):
        return GloroNet.load_model(filename + '.gloronet')

    def save(self, filename, model):
        model.save(filename + '.gloronet')


def get_architecture(architecture, input_shape, num_classes):
    try:
        _orig_architecture = architecture
        params = '{}'

        if '.' in architecture:
            architecture, params = architecture.split('.', 1)

        architecture = getattr(architectures, architecture)(
            input_shape, num_classes, **json.loads(params))

    except:
        raise ValueError(f'unknown architecture: {_orig_architecture}')

    return architecture

def train_gloro(
        dataset="mnist",
        architecture="cnn_4C3F",
        epsilon=0.3,
        epsilon_train=0.3,
        power_iterations=5,
        epsilon_schedule='fixed',
        loss='sparse_crossentropy',
        augmentation=None,
        epochs=200,
        batch_size=128,
        optimizer='adam',
        lr=1e-3,
        lr_schedule='decay_to_0.000001_after_half',
        trades_schedule=None):
        
    # Load data and set up data pipeline.
    print('loading data...')

    train, test, metadata = get_data(dataset, batch_size, augmentation)

    # Create the model.
    print('creating model...')
    
    input_shape = metadata.features['image'].shape
    num_classes = metadata.features['label'].num_classes
    
    architecture = get_architecture(architecture,input_shape, num_classes)
    
    g = GloroNet(
        *architecture, epsilon=epsilon_train, num_iterations=power_iterations)
    metrics = [vra,clean_acc,rejection_rate]

    g.summary()

    # Compile and train the model.
    print('compiling model...')

    g.compile(
        loss=losses.get(loss),
        optimizer=get_optimizer(optimizer, lr),
        metrics=metrics)

    print(train)
    
    g.fit(
        train,
        epochs=epochs, 
        validation_data=test,
        callbacks=[EpsilonScheduler(epsilon_schedule),LrScheduler(lr_schedule),] +
                  ([TradesScheduler(trades_schedule)] if trades_schedule else []))

    return g

def fpr_experiment(superrobust='N',
            dataset="mnist",
            architecture="minmax_cnn_4C3F",
            epsilon=0.3,
            epsilon_train=0.3,
            epsilon_schedule='fixed',
            loss='sparse_crossentropy',
            augmentation=None,
            epochs=200,
            batch_size=128,
            optimizer='adam',
            lr=1e-3,
            lr_schedule='decay_to_0.000001_after_half',
            trades_schedule=None,
            using_gpu=True,
            gpu=0,
            filename='0921.log',
            train_model=True,
            model_path=None):

    print("Experiment parameters:\n")
    print("dataset=", dataset)
    print("architecture=", architecture)
    print("superrobust=", superrobust)
    print("epsilon=", epsilon)
    print("epsilon_train=", epsilon_train)
    print("epsilon_schedule=", epsilon_schedule)
    print("loss=", loss)
    print("augmentation=", augmentation)
    print("epochs=", epochs)
    print("batch_size=", batch_size)
    print("optimizer=", optimizer)
    print("lr=", lr)
    print("lr_schedule=", lr_schedule)
    print("trades_schedule=", trades_schedule)
    print("log filename=", filename)
    print("train_model=", train_model)
    print("model_path=", model_path)

    f = open(filename, "a")
    f.write("superrobust="+str (superrobust)
            +", dataset="+str (dataset)
            +", architecture="+str (architecture)
            +", epsilon="+str (epsilon)
            +", epsilon_train="+str (epsilon_train)
            +", epochs="+str (epochs)+":\n")

    if using_gpu:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        tfds_dir = os.environ['TFDS_DIR'] if 'TFDS_DIR' in os.environ else None

    print('loading data...')
    train, test, metadata = get_data(dataset, batch_size, augmentation)

    if train_model:
        gloro_model = train_gloro(
            dataset=dataset,
            architecture=architecture,
            epsilon=epsilon,
            epsilon_train=2*epsilon if superrobust == 'Y' else epsilon_train,
            loss=loss,
            epsilon_schedule=epsilon_schedule,
            augmentation=augmentation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            lr=lr,
            lr_schedule=lr_schedule,
            trades_schedule=trades_schedule)
        gloro_model.save(f'model_{dataset}_{epsilon}_2{superrobust}.gloronet')
    else:
        gloro_model =  GloroNet.load_model(model_path)

    gloro_model.compile(
        loss=losses.get(loss),
        optimizer=get_optimizer(optimizer, lr),
        metrics=[clean_acc, vra, rejection_rate])

    gloro_model.epsilon = epsilon
    test_eval = gloro_model.evaluate(test)
    results = {}
    results.update({
        'test_' + metric.name.split('pred_')[-1]: round(value, 4)
        for metric, value in zip(gloro_model.metrics, test_eval)
    })

    r = f'Results for {epsilon}:',results
    f.write(str(r)+"\n")
    print(r)


    gloro_model.epsilon = 2*epsilon
    test_eval = gloro_model.evaluate(test)
    results = {}
    results.update({
        'test_' + metric.name.split('pred_')[-1]: round(value, 4)
        for metric, value in zip(gloro_model.metrics, test_eval)
    })

    r = f'Results for {2*epsilon}:',results
    f.write(str(r)+"\n")
    print(r)

    fullDistribution = 0
    robustPoints = 0
    superRobustPoints = 0
    all_eps = []
    
    for x, y in tfds.as_numpy(test):
        preds, eps, y_b = gloro_model.predict_with_certified_radius(x)
    
        all_eps.append(eps.numpy())
        fullDistribution += 1
    
    eps = np.concatenate(all_eps, axis=0)
    robustPoints = (eps >= epsilon).sum()
    superRobustPoints = (eps >= 2 * epsilon).sum()
    
    print(robustPoints, superRobustPoints, fullDistribution)
    
    return (robustPoints - superRobustPoints) / robustPoints


if __name__ == '__main__':

    @scriptify
    def script(superrobust='N',
            dataset='mnist',
            architecture='minmax_cnn_4C3F',
            epsilon=1.58,
            epsilon_train=1.58,
            epsilon_schedule='fixed',
            loss='trades.1.5',
            augmentation=None,
            epochs=20,
            batch_size=128,
            optimizer='adam',
            lr=1e-3,
            lr_schedule='decay_to_0.000001_after_half',
            trades_schedule=None,
            using_gpu=True,
            gpu=0,
            filename='moreMnist.log',
            train_model=False,
            model_path=None):
            
        fpr_experiment(superrobust=superrobust,
            dataset=dataset,
            architecture=architecture,
            epsilon=epsilon,
            epsilon_train=epsilon_train,
            epsilon_schedule=epsilon_schedule,
            loss=loss,
            augmentation=augmentation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            lr=lr,
            lr_schedule=lr_schedule,
            trades_schedule=trades_schedule,
            using_gpu=using_gpu,
            gpu=gpu,
            filename=filename)
