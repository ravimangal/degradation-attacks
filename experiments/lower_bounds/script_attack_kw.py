import os
import torch

#from dbify import dbify
from scriptify import scriptify

import kw
from kw import evaluate_robust


## Pytorch implementation of DOS attack ########################################


class LinfDosAttack(object):
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
            self._attack = LinfPgd(
                2 * epsilon if project_back else epsilon, **attack_kwargs)

        else:
            raise ValueError(f'unknown base attack: {base_attack}')

        self._project_back = project_back
        self._skip = skip

    def _attack_base_model(self, model, data, gpu):
        adv_x_batches = []

        i = 0
        for batch_x, batch_y in data:
            if i % self._skip != 0:
                i += 1
                continue

            # Attack the model at a radius of 2\epsilon.
            adv_x, successes = self._attack(
                model,
                batch_x.to(gpu),
                batch_y.to(gpu))

            if self._project_back:
                # Project adversarial examples back to the original epsilon
                # ball.
                eps = torch.tensor(self._epsilon)
                perturbation = adv_x - batch_x.to(adv_x.device)
                perturbation = torch.max(
                    torch.min(
                        perturbation, eps.to(perturbation.device)),
                    -eps.to(perturbation.device))

                adv_x = batch_x + perturbation.to(batch_x.device)

            adv_x_batches.append(adv_x)

            print(f'{i}: success_rate: {successes.cpu().numpy().mean()}')

            i += 1

        return adv_x_batches

    def eval(
        self,
        model,
        data,
        data_name,
        gpu='cuda:1',
        bs=1000,
    ):
        model = model.to(gpu)

        vra = evaluate_robust(data, model, self._epsilon, True, skip=self._skip)

        upper_bound_vra = evaluate_robust(
            data, model, 2 * self._epsilon, True, skip=self._skip)

        print('attacking underlying model at a radius of 2 * epsilon...')
        adv_x_batches = self._attack_base_model(model, data, gpu=gpu)

        labels = [y for _, y in data]
        adv_data = list(zip(adv_x_batches, labels))

        vra_under_dos = evaluate_robust(
            adv_data, model, self._epsilon, True, skip=1)

        return vra, vra_under_dos, upper_bound_vra



class LinfPgd(object):

    def __init__(
        self,
        epsilon,
        steps=50,
        step_size=None,
        loss=None,
        aggregate_before_grad=False,
    ):
        self.epsilon = torch.tensor(epsilon)
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
        x_0 = x

        for _ in range(self.steps):
            x = torch.autograd.Variable(x, requires_grad=True)

            y_pred = model(x)

            loss = self.loss(y_pred, y)
            loss.backward()
            grad = x.grad

            x = x + self.step_size * grad.sign()

            perturbation = x - x_0

            perturbation = torch.max(
                torch.min(perturbation, self.epsilon.to(perturbation.device)),
                -self.epsilon.to(perturbation.device))

            x = x_0 + perturbation

        # Check if the attack succeeded.
        y_pred = torch.argmax(model(x), axis=1).to(y.device)

        successes = torch.logical_not(torch.eq(y, y_pred))

        return x, successes


## Script ######################################################################


if __name__ == '__main__':
    @scriptify
    # @dbify('gloro', 'dos_kw')
    def script(
        dataset,
        architecture,
        epsilon,
        batch_size=1000,
        skip=100,
        use_2eps=False,
    ):
        path = os.environ['KWL_MODEL_DIR']
        file = f'{dataset}/{architecture}/checkpoint.pth'

        state = torch.load(f'{path}/{file}')['state_dict'][0]
        base_model = getattr(kw, architecture)()
        base_model.load_state_dict(state)

        _, test = getattr(kw, dataset)(batch_size)

        dos_attack = LinfDosAttack(epsilon, skip=skip, project_back=use_2eps)

        vra, vra_under_dos, upper_bound_vra = dos_attack.eval(
            base_model, test, dataset)

        print(f'base_vra: {float(vra)}\n',
            f'vra_under_dos: {float(vra_under_dos)}\n',
            f'upper_bound_vra: {upper_bound_vra}\n',
            f'vra_reduction: {float((vra - vra_under_dos))}')

        return {
            'base_vra': float(vra),
            'vra_under_dos': float(vra_under_dos),
            'upper_bound_vra': upper_bound_vra,
            'vra_reduction': float((vra - vra_under_dos)),
        }
