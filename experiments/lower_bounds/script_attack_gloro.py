import foolbox as fb
import numpy as np
import tensorflow as tf
import os

from dbify import dbify
from scriptify import scriptify

from tensorflow.keras.utils import Progbar
from gloro import GloroNet

from ..training.utils import get_data

class DosAttack(object):

    def __init__(
        self, epsilon, base_attack='pgd', bounds=None, **attack_kwargs
    ):
        self._epsilon = epsilon

        if bounds is None:
            bounds = (0. - 2*epsilon, 1. + 2*epsilon)

        self._bounds = bounds

        if base_attack == 'pgd':
            self._attack = fb.attacks.L2PGD(**attack_kwargs)

        else:
            raise ValueError(f'unknown base attack: {base_attack}')

    def _attack_base_model(self, model, data, gpu=0):
        fb_model = fb.TensorFlowModel(
            model, bounds=self._bounds, device=f'GPU:{gpu}')

        adv_x_batches = []

        pb = Progbar(target=len(data))
        pb.add(0, [])

        for batch_x, batch_y in data:
            # TODO: do we only want to attack points that are already correct
            #   and robust?
            batch_y_pred = tf.argmax(model(batch_x), axis=1)

            # Attack the model at a radius of 2\epsilon.
            _, [adv_x], successes = self._attack(
                fb_model,
                batch_x,
                batch_y_pred,
                epsilons=[2 * self._epsilon])

            # Project adversarial examples back to the original epsilon ball.
            perturbations = adv_x - batch_x
            perturbations = self._epsilon * perturbations / tf.sqrt(
                tf.reduce_sum(
                    perturbations**2,
                    axis=tf.range(1, perturbations.ndim),
                    keepdims=True))

            adv_x_batches.append(batch_x + perturbations)

            pb.add(1, [('success_rate', successes.numpy().mean())])

        return adv_x_batches

    def eval(self, model, data, gpu=0):
        raise NotImplementedError


class GloroDosAttack(DosAttack):

    def eval(self, model, data, gpu=0):
        print('attacking at a radius of 2 * epsilon...')
        adv_x_batches = self._attack_base_model(model.f, data, gpu=gpu)

        pb = Progbar(target=2 * len(adv_x_batches))
        pb.add(0, [])

        # Get the points for which we are both correct and robust. This will
        # give us our denominator: these are the only points that we care about
        # our DOS attack succeding on.
        correct = []
        for batch_x, batch_y in data:
            correct.append(
                tf.cast(
                    tf.argmax(model(batch_x), axis=1) == batch_y, 'float32'))
            pb.add(1, [('vra', tf.reduce_mean(correct[-1]))])

        dos_success = []
        for adv_x in adv_x_batches:
            adv_y = model(adv_x)
            num_classes = adv_y.shape[1] - 1
            dos_success.append(
                tf.cast(tf.argmax(adv_y, axis=1) == num_classes, 'float32'))
            pb.add(1, [('dos_success', tf.reduce_mean(dos_success[-1]))])

        correct = tf.concat(correct, axis=0)
        dos_success = tf.concat(dos_success, axis=0)

        # This is the fraction of points which are correctly classified, robust,
        # and do not have any points rejected in their epsilon neighborhood.
        vra_under_dos = tf.reduce_mean(correct * (1. - dos_success))

        # This is the fraction of correctly classified and robust (at radius
        # epsilon) test points that have a point in their epsilon neighborhood
        # that is rejected.
        fraction_susceptible = (
            tf.reduce_sum(correct * dos_success) / tf.reduce_sum(correct))

        return vra_under_dos, fraction_susceptible


class GloroDosAttackNaive(DosAttack):

    def eval(self, model, data, gpu=0):
        print('attacking at a radius of epsilon...')

        fb_model = fb.TensorFlowModel(
            model.f, bounds=self._bounds, device=f'GPU:{gpu}')

        adv_x_batches = []

        pb = Progbar(target=len(data))
        pb.add(0, [])

        for batch_x, batch_y in data:
            # TODO: do we only want to attack points that are already correct
            #   and robust?
            batch_y_pred = tf.argmax(model.f(batch_x), axis=1)

            # Attack the model at a radius of 2\epsilon.
            _, [adv_x], _ = self._attack(
                fb_model,
                batch_x,
                batch_y_pred,
                epsilons=[self._epsilon])

            successes = tf.equal(
                tf.argmax(model(adv_x), axis=1), model.output_shape[1] - 1)

            adv_x_batches.append(adv_x)

            pb.add(1, [('success_rate', successes.numpy().mean())])

        pb = Progbar(target=2 * len(adv_x_batches))
        pb.add(0, [])

        # Get the points for which we are both correct and robust. This will
        # give us our denominator: these are the only points that we care about
        # our DOS attack succeding on.
        correct = []
        for batch_x, batch_y in data:
            correct.append(
                tf.cast(
                    tf.argmax(model(batch_x), axis=1) == batch_y, 'float32'))
            pb.add(1, [('vra', tf.reduce_mean(correct[-1]))])

        dos_success = []
        for adv_x in adv_x_batches:
            adv_y = model(adv_x)
            num_classes = adv_y.shape[1] - 1
            dos_success.append(
                tf.cast(tf.argmax(adv_y, axis=1) == num_classes, 'float32'))
            pb.add(1, [('dos_success', tf.reduce_mean(dos_success[-1]))])

        correct = tf.concat(correct, axis=0)
        dos_success = tf.concat(dos_success, axis=0)

        # This is the fraction of points which are correctly classified, robust,
        # and do not have any points rejected in their epsilon neighborhood.
        vra_under_dos = tf.reduce_mean(correct * (1. - dos_success))

        # This is the fraction of correctly classified and robust (at radius
        # epsilon) test points that have a point in their epsilon neighborhood
        # that is rejected.
        fraction_susceptible = (
            tf.reduce_sum(correct * dos_success) / tf.reduce_sum(correct))

        return vra_under_dos, fraction_susceptible


## Script ######################################################################


if __name__ == '__main__':
    @scriptify
    @dbify('gloro', 'dos_kw')
    def script(
        dataset,
        architecture,
        epsilon,
        batch_size=256,
    ):
        tf.keras.backend.set_image_data_format('channels_last')

        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpu = 2

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)


        path = os.environ['GLORO_MODEL_DIR']
        file = f'{dataset}/{architecture}'

        attack = GloroDosAttack(epsilon)
        train, test, meta = get_data(dataset, batch_size=batch_size)
        g = GloroNet.load_model(f'{path}/{file}')

        vra_under_dos, fraction_susceptible = attack.eval(g, test)

        print(
            f'vra_under_dos: {vra_under_dos.numpy():.3f}\n'
            f'fraction_susceptible: {fraction_susceptible.numpy():.3f}')

        return {
            vra_under_dos: vra_under_dos.numpy(),
            fraction_susceptible: fraction_susceptible.numpy()
        }

