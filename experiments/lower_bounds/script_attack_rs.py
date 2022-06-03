import os
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

#from dbify import dbify
from scriptify import scriptify
from tensorflow.keras.utils import Progbar

from cohen_et_al import get_architecture
from cohen_et_al import get_dataset
from cohen_et_al import get_num_classes
from cohen_et_al import Smooth


## Pytorch implementation of DOS attack ########################################


class RsL2Pgd(object):

    def __init__(
        self,
        epsilon,
        sigma=0.125,
        samples=100,
        steps=50,
        step_size=None,
        loss=None,
        aggregate_before_grad=False,
    ):
        self.epsilon = torch.tensor(epsilon)
        self.sigma = sigma
        self.samples = samples
        self.steps = steps

        if step_size is None:
            self.step_size = epsilon / steps * 2
        else:
            self.step_size = step_size

        if loss is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = loss

        self.aggregate_before_grad = aggregate_before_grad

    def __call__(self, model, x, y):
        n = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]

        x_0 = x

        for _ in range(self.steps):
            x = torch.autograd.Variable(x, requires_grad=True)

            y_noised = torch.repeat_interleave(y, self.samples, axis=0)

            noise = torch.randn(
                (self.samples, 1, h, w, c), requires_grad=True) * self.sigma

            x_noised = torch.reshape(
                x[None] + noise.to(x.device), (-1, h, w, c))
            y_pred = model(x_noised)

            if self.aggregate_before_grad:
                loss = self.loss(
                    torch.mean(
                        torch.reshape(y_pred, (self.samples, n, -1)),
                        axis=0),
                    y)

                loss.backward()
                grad = x.grad

            else:
                loss = self.loss(y_pred, y_noised)
                loss.backward()

                grad = torch.mean(
                    torch.reshape(
                        x_noised.grad,
                        (self.samples, n, h, w, c)),
                    axis=0)

            x = x + self.step_size * grad

            norm = torch.sqrt(torch.sum(
                (x - x_0)**2, axis=(1,2,3), keepdims=True))

            x = x_0 + (x - x_0) / norm * torch.minimum(
                self.epsilon.to(norm.device), norm)

        # Check if the attack succeeded.
        noise = torch.randn((self.samples, 1, h, w, c)) * self.sigma

        x_noised = torch.reshape(x[None] + noise.to(x.device), (-1, h, w, c))
        y_noised = model(x_noised)

        # Computes the mode prediction on the noised inputs.
        y_pred = torch.argmax(
            # Computes the number of times each class was predicted.
            torch.sum(
                torch.reshape(
                    torch.eq(
                        y_noised,
                        torch.max(y_noised, axis=1, keepdims=True).values).type(
                          torch.FloatTensor),
                    (self.samples, n, -1)),
                axis=0),
            axis=1).to(y.device)

        successes = torch.logical_not(torch.eq(y, y_pred))

        return x, successes


class RsDosAttack(object):
    '''
    Does a DOS attack on a smoothed model by proxy, by searching for adversarial
    examples on the underlying model.
    '''
    def __init__(
        self,
        epsilon,
        base_attack='pgd',
        bounds=None,
        project_back=True,
        skip=100,
        **attack_kwargs,
    ):
        self._epsilon = epsilon

        if bounds is None:
            bounds = (0. - 2*epsilon, 1. + 2*epsilon)

        self._bounds = bounds

        if base_attack == 'pgd':
            self._attack = RsL2Pgd(
                2 * epsilon if project_back else epsilon, **attack_kwargs)

        else:
            raise ValueError(f'unknown base attack: {base_attack}')

        self._project_back = project_back
        self._skip = skip

    def _attack_base_model(self, model, data, gpu=0):
        adv_x_batches = []

        pb = Progbar(target=len(data) // self._skip)
        pb.add(0, [])
        i = 0

        for batch_x, batch_y in data:
            if i % self._skip != 0:
                i += 1
                continue

            # Attack the model at a radius of 2\epsilon.
            adv_x, successes = self._attack(
                model,
                batch_x[None].to(gpu),
                torch.tensor([batch_y]).to(gpu))

            if self._project_back:
                # Project adversarial examples back to the original epsilon
                # ball.
                perturbations = adv_x - batch_x.to(adv_x.device)
                perturbations = self._epsilon * perturbations / torch.sqrt(
                    torch.sum(
                        perturbations**2,
                        axis=(1, 2, 3),
                        keepdim=True))

                adv_x = batch_x + perturbations.to(batch_x.device)

            adv_x_batches.append(adv_x)

            pb.add(1, [('success_rate', successes.cpu().numpy().mean())])

            i += 1

        return adv_x_batches

    def eval(
        self,
        model,
        data,
        data_name,
        gpu='cuda:0',
        sigma=0.12,
        n0=100,
        n=int(1e5),
        alpha=0.001,
        bs=1000,
    ):
        pb = Progbar(target=len(data) // self._skip)
        pb.add(0, [])

        smoothed = Smooth(model, get_num_classes(data_name), sigma)

        # Get the points for which we are both correct and robust. This will
        # give us our denominator: these are the only points that we care about
        # our DOS attack succeding on.
        correct = []
        i = 0
        for batch_x, batch_y in data:
            if i % self._skip != 0:
                i += 1
                continue

            pred, r = smoothed.certify(batch_x[None].to(gpu), n0, n, alpha, bs)

            correct.append(
                torch.logical_and(
                    torch.eq(torch.tensor(pred), torch.tensor(batch_y)),
                    torch.tensor(r >= self._epsilon)).type(
                        torch.FloatTensor)[None])
            pb.add(1, [('vra', torch.mean(correct[-1]).cpu().numpy())])

            print(f'{i}\t{batch_y}\t{pred}\t{r}\t{correct[-1]}')

            i += 1

        print('attacking underlying model at a radius of 2 * epsilon...')
        adv_x_batches = self._attack_base_model(model, data, gpu=gpu)

        pb = Progbar(target=len(adv_x_batches))
        pb.add(0, [])

        dos_success = []
        for adv_x in adv_x_batches:

            adv_y, r = smoothed.certify(adv_x.to(gpu), n0, n, alpha, bs)

            dos_success.append(
                torch.logical_or(
                    torch.eq(torch.tensor(adv_y), torch.tensor(-1)),
                    torch.tensor(r < self._epsilon)).type(
                        torch.FloatTensor)[None])

            pb.add(
                1, [('dos_success', torch.mean(dos_success[-1]).cpu().numpy())])

            print(f'{i}\t{adv_y}\t{r}\t{dos_success[-1]}')

        correct = torch.cat(correct, axis=0)
        dos_success = torch.cat(dos_success, axis=0)

        vra = torch.mean(correct)

        # This is the fraction of points which are correctly classified, robust,
        # and do not have any points rejected in their epsilon neighborhood.
        vra_under_dos = torch.mean(correct * (1. - dos_success))

        # This is the fraction of correctly classified and robust (at radius
        # epsilon) test points that have a point in their epsilon neighborhood
        # that is rejected.
        false_positive_rate = (
            torch.sum(correct * dos_success) / torch.sum(correct))

        return vra, vra_under_dos, false_positive_rate


## Script ######################################################################


if __name__ == '__main__':
    @scriptify
    # @dbify('gloro', 'dos_rs')
    def script(
        dataset,
        architecture,
        sigma,
        epsilon,
        n0=100,
        n=100000,
        alpha=0.001,
        batch_size=1000,
        skip=100,
    ):
        path = os.environ['COHEN_ET_AL_MODEL_DIR']
        file = f'{dataset}/{architecture}/noise_{sigma:.2f}/checkpoint.pth.tar'

        checkpoint = torch.load(f'{path}/{file}')
        base_model = get_architecture(checkpoint['arch'], dataset)
        base_model.load_state_dict(checkpoint['state_dict'])

        if dataset == 'imagenet':
            imagenet_dir = os.environ['IMAGENET_DIR']

            (test,), metadata = tfds.load(
                'imagenet2012',
                split=['validation'],
                with_info=True,
                as_supervised=True,
                shuffle_files=False,
                data_dir=imagenet_dir)

            def preprocess(x, y):
                x = tf.image.resize(x, [256,256], preserve_aspect_ratio=True)
                x = tf.image.resize_with_crop_or_pad(x, 224, 224)
                x = tf.transpose(x, (2,0,1))
                return (tf.cast(x, 'float32') / 255., y)

            test = (test
                .map(
                    preprocess,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
                .batch(skip)
                .prefetch(tf.data.experimental.AUTOTUNE))

            print('loading data...')
            pb = Progbar(target=len(test))
            pb.add(0, [])

            instances = []
            for x, y in test:
                instances.append(
                    (torch.tensor(x[0].numpy()), torch.tensor(y[0].numpy())))
                pb.add(1, [])

            test = instances

            skip = 1

        else:
            test = get_dataset(dataset, 'test')

        dos_attack = RsDosAttack(epsilon, aggregate_before_grad=True, skip=skip)

        vra, vra_under_dos, fpr = dos_attack.eval(
            base_model, test, dataset,
            sigma=sigma,
            n0=n0,
            n=n,
            alpha=alpha,
            bs=batch_size)

        print(f'base_vra: {float(vra.cpu().numpy())}\n',
            f'vra_under_dos: {float(vra_under_dos.cpu().numpy())}\n',
            f'false_positive_rate: {float(fpr.cpu().numpy())}\n',
            f'vra_reduction: {float((vra - vra_under_dos).cpu().numpy())}')

        return {
            'base_vra': float(vra.cpu().numpy()),
            'vra_under_dos': float(vra_under_dos.cpu().numpy()),
            'false_positive_rate': float(fpr.cpu().numpy()),
            'vra_reduction': float((vra - vra_under_dos).cpu().numpy()),
        }
