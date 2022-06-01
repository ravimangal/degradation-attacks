import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DEBUG = True

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def large():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

def mnist(batch_size, shuffle_test=False):
    mnist_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        pin_memory=True)

    return train_loader, test_loader


def cifar(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def large_color():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model



from torch.autograd import Variable

import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_robust(loader, model, epsilon, verbose, skip=1,
                    real_time=False, parallel=False, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()

    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        if i % skip != 0:
            continue

        X,y = X.cuda(1), y.cuda(1).long()
        if y.dim() == 2:
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(i, robust_ce.item(), robust_err, ce.item(), err.item())
        if verbose:
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time,
                      loss=losses, error=errors, rloss = robust_losses,
                      rerrors = robust_errors), end=endline)

        del X, y, robust_ce, out, ce
        # if DEBUG and i ==10:
        #     break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))

    return 1. - robust_errors.avg






from abc import ABCMeta, abstractmethod

class DualObject(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """ Initialize a dual layer by initializing the variables needed to
        compute this layer's contribution to the upper and lower bounds.
        In the paper, if this object is at layer i, this is initializing `h'
        with the required cached values when nu[i]=I and nu[i]=-I.
        """
        super(DualObject, self).__init__()

    @abstractmethod
    def apply(self, dual_layer):
        """ Advance cached variables initialized in this class by the given
        dual layer.  """
        raise NotImplementedError

    @abstractmethod
    def bounds(self):
        """ Return this layers contribution to the upper and lower bounds. In
        the paper, this is the `h' upper bound where nu is implicitly given by
        c=I and c=-I. """
        raise NotImplementedError

    @abstractmethod
    def objective(self, *nus):
        """ Return this layers contribution to the objective, given some
        backwards pass. In the paper, this is the `h' upper bound evaluated on a
        the given nu variables.
        If this is layer i, then we get as input nu[k] through nu[i].
        So non-residual layers will only need nu[-1] and nu[-2]. """
        raise NotImplementedError

class DualLayer(DualObject):
    @abstractmethod
    def forward(self, *xs):
        """ Given previous inputs, apply the affine layer (forward pass) """
        raise NotImplementedError

    @abstractmethod
    def T(self, *xs):
        """ Given previous inputs, apply the transposed affine layer
        (backward pass) """
        raise NotImplementedError





def full_bias(l, n=None):
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified.
    if isinstance(l, nn.Linear):
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d):
        if n is None:
            raise ValueError("Need to pass n=<output dimension>")
        b = l.bias.unsqueeze(1).unsqueeze(2)
        if isinstance(n, int):
            k = int((n/(b.numel()))**0.5)
            return b.expand(1,b.numel(),k,k).contiguous().view(1,-1)
        else:
            return b.expand(1,*n)
    elif isinstance(l, Dense):
        return sum(full_bias(layer, n=n) for layer in l.Ws if layer is not None)
    elif isinstance(l, nn.Sequential) and len(l) == 0:
        return 0
    else:
        raise ValueError("Full bias can't be formed for given layer.")



class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]

class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x,W in zip(xs, self.Ws) if W is not None)
        return out


class DualNetwork(nn.Module):
    def __init__(self, net, X, epsilon,
                 proj=None, norm_type='l1', bounded_input=False,
                 input_l=0, input_u=1,
                 data_parallel=True):
        """
        This class creates the dual network.
        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)):
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad():
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net):
                zs = [X]
            else:
                zs = [X[:1]]
            nf = [zs[0].size()]
            for l in net:
                if isinstance(l, Dense):
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())


        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input,l=input_l,u=input_u)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)):
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                                      in_f, out_f, zs[i])

            # skip last layer
            if i < len(net)-1:
                for l in dual_net:
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else:
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]

        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for
           i,l in enumerate(dual_net))

class DualNetBounds(DualNetwork):
    def __init__(self, *args, **kwargs):
        warnings.warn("DualNetBounds is deprecated. Use the proper "
                      "PyTorch module DualNetwork instead. ")
        super(DualNetBounds, self).__init__(*args, **kwargs)

    def g(self, c):
        return self(c)

class RobustBounds(nn.Module):
    def __init__(self, net, epsilon, parallel=False, **kwargs):
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs
        self.DualNetworkClass = ParallelDualNetwork if parallel else DualNetwork

    def forward(self, X,y):
        num_classes = self.net[-1].out_features
        dual = self.DualNetworkClass(self.net, X, self.epsilon, **self.kwargs)
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda(1)
        f = -dual(c)
        return f

def robust_loss(net, epsilon, X, y,
                size_average=True, device_ids=None, parallel=False, **kwargs):
    reduction = 'mean' if size_average else 'none'
    if parallel:
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else:
        f = RobustBounds(net, epsilon, **kwargs)(X,y)
    err = (f.max(1)[1] != y)
    if size_average:
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduction=reduction)(f, y)
    return ce_loss, err

class InputSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        self.i = 0
        super(InputSequential, self).__init__(*args, **kwargs)

    def set_start(self, i):
        self.i = i

    def forward(self, input):
        """ Helper class to apply a sequential model starting at the ith layer """
        xs = [input]
        for j,module in enumerate(self._modules.values()):
            if j >= self.i:
                if 'Dense' in type(module).__name__:
                    xs.append(module(*xs))
                else:
                    xs.append(module(xs[-1]))
        return xs[-1]

class ParallelDualNetwork(DualNetwork):
    def __init__(self, net, X, epsilon,
                 proj=None, norm_type='l1', bounded_input=False,
                 input_l=0, input_u=1):
        super(DualNetwork, self).__init__()

        if any('BatchNorm2d' in str(l.__class__.__name__) for l in net):
            raise NotImplementedError
        if X.size(0) != 1:
            raise ValueError('Only use this function for a single example. This is '
                'intended for the use case when a single example does not fit in '
                'memory.')
        zs = [X[:1]]
        nf = [zs[0].size()]

        for l in net:
            if 'Dense' in type(l).__name__:
                zs.append(l(*zs))
            else:
                zs.append(l(zs[-1]))
            nf.append(zs[-1].size())

        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input,l=input_l,u=input_u)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)):
            if isinstance(layer, nn.ReLU):
                # compute bounds
                D = (InputSequential(*dual_net[1:]))
                Dp = nn.DataParallel(D)
                zl,zu = 0,0
                for j,dual_layer in enumerate(dual_net):
                    D.set_start(j)
                    out = dual_layer.bounds(network=Dp)
                    zl += out[0]
                    zu += out[1]

                dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                    in_f, out_f, zs[i], zl=zl, zu=zu)
            else:
                dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                    in_f, out_f, zs[i])

            dual_net.append(dual_layer)
        self.dual_net = dual_net[:-1]
        self.last_layer = dual_net[-1]


# Data parallel versions of the loss calculation
def robust_loss_parallel(net, epsilon, X, y, proj=None,
                 norm_type='l1', bounded_input=False, size_average=True):
    if any('BatchNorm2d' in str(l.__class__.__name__) for l in net):
        raise NotImplementedError
    if bounded_input:
        raise NotImplementedError('parallel loss for bounded input spaces not implemented')
    if X.size(0) != 1:
        raise ValueError('Only use this function for a single example. This is '
            'intended for the use case when a single example does not fit in '
            'memory.')
    zs = [X[:1]]
    nf = [zs[0].size()]
    for l in net:
        if isinstance(l, Dense):
            zs.append(l(*zs))
        else:
            zs.append(l(zs[-1]))
        nf.append(zs[-1].size())

    dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

    for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)):
        if isinstance(layer, nn.ReLU):
            # compute bounds
            D = (InputSequential(*dual_net[1:]))
            Dp = nn.DataParallel(D)
            zl,zu = 0,0
            for j,dual_layer in enumerate(dual_net):
                D.set_start(j)
                out = dual_layer.bounds(network=Dp)
                zl += out[0]
                zu += out[1]

            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i], zl=zl, zu=zu)
        else:
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i])

        dual_net.append(dual_layer)

    num_classes = net[-1].out_features
    c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda(1)

    # same as f = -dual.g(c)
    nu = [-c]
    for l in reversed(dual_net[1:]):
        nu.append(l.T(*nu))

    f = -sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))])
             for i,l in enumerate(dual_net))

    err = (f.max(1)[1] != y)

    if size_average:
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err




def select_input(X, epsilon, proj, norm, bounded_input, l=0,u=1):
    if proj is not None and norm=='l1_median' and X[0].numel() > proj:
        if bounded_input:
            return InfBallProjBounded(X,epsilon,proj, l=l, u=u)
        else:
            return InfBallProj(X,epsilon,proj)
    elif norm == 'l1':
        if bounded_input:
            return InfBallBounded(X, epsilon, l=l, u=u)
        else:
            return InfBall(X, epsilon)
    elif proj is not None and norm=='l2_normal' and X[0].numel() > proj:
        return L2BallProj(X,epsilon,proj)
    elif norm == 'l2':
        return L2Ball(X,epsilon)
    else:
        raise ValueError("Unknown estimation type: {}".format(norm))

class InfBall(DualObject):
    def __init__(self, X, epsilon):
        super(InfBall, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l1 = nu_1.abs().sum(1)
        if isinstance(epsilon, torch.Tensor):
            while epsilon.dim() < nu_x.dim():
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l1,
                nu_x + epsilon*l1)

    def objective(self, *nus):
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor):
            while epsilon.dim() < nu.dim()-1:
                epsilon = epsilon.unsqueeze(1)
        l1 = (epsilon*nu.abs()).sum(2)
        return -nu_x - l1

class InfBallBounded(DualObject):
    def __init__(self, X, epsilon, l=0, u=1):
        super(InfBallBounded, self).__init__()
        self.epsilon = epsilon
        if torch.is_tensor(l):
            l_ = torch.max(X-epsilon,l)
        else:
            l_ = (X-epsilon).clamp(min=l)
        if torch.is_tensor(u):
            u_ = torch.min(X+epsilon,u)
        else:
            u_ = (X+epsilon).clamp(max=u)

        self.l = l_.view(X.size(0), 1, -1)
        self.u = u_.view(X.size(0), 1, -1)

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu_1[-1]
        else:
            nu = network(self.nu_1[0])
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)

        zu = (self.u.matmul(nu_pos) + self.l.matmul(nu_neg)).squeeze(1)
        zl = (self.u.matmul(nu_neg) + self.l.matmul(nu_pos)).squeeze(1)
        return (zl.view(zl.size(0), *nu.size()[2:]),
                zu.view(zu.size(0), *nu.size()[2:]))

    def objective(self, *nus):
        nu = nus[-1]
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)
        u, l = self.u.unsqueeze(3).squeeze(1), self.l.unsqueeze(3).squeeze(1)
        return (-nu_neg.matmul(l) - nu_pos.matmul(u)).squeeze(2)

class InfBallProj(InfBall):
    def __init__(self, X, epsilon, k):
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu = [X.new(1,k,*X.size()[1:]).cauchy_()]

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else:
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        l1 = torch.median(self.nu[-1].abs(), 1)[0]
        return (nu_x - self.epsilon*l1,
                nu_x + self.epsilon*l1)

class InfBallProjBounded(InfBallProj):
    def __init__(self, X, epsilon, k, l=0, u=1):
        self.epsilon = epsilon

        self.nu_one_l = [(X-epsilon).clamp(min=l)]
        self.nu_one_u = [(X+epsilon).clamp(max=u)]
        self.nu_x = [X]

        self.l = self.nu_one_l[-1].view(X.size(0), 1, -1)
        self.u = self.nu_one_u[-1].view(X.size(0), 1, -1)

        n = X[0].numel()
        R = X.new(1,k,*X.size()[1:]).cauchy_()
        self.nu_l = [R * self.nu_one_l[-1].unsqueeze(1)]
        self.nu_u = [R * self.nu_one_u[-1].unsqueeze(1)]

    def apply(self, dual_layer):
        self.nu_l.append(dual_layer(*self.nu_l))
        self.nu_one_l.append(dual_layer(*self.nu_one_l))
        self.nu_u.append(dual_layer(*self.nu_u))
        self.nu_one_u.append(dual_layer(*self.nu_one_u))

    def bounds(self, network=None):
        if network is None:
            nu_u = self.nu_u[-1]
            nu_one_u = self.nu_one_u[-1]
            nu_l = self.nu_l[-1]
            nu_one_l = self.nu_one_l[-1]
        else:
            nu_u = network(self.nu_u[0])
            nu_one_u = network(self.nu_one_u[0])
            nu_l = network(self.nu_l[0])
            nu_one_l = network(self.nu_one_l[0])

        nu_l1_u = torch.median(nu_u.abs(),1)[0]
        nu_pos_u = (nu_l1_u + nu_one_u)/2
        nu_neg_u = (-nu_l1_u + nu_one_u)/2

        nu_l1_l = torch.median(nu_l.abs(),1)[0]
        nu_pos_l = (nu_l1_l + nu_one_l)/2
        nu_neg_l = (-nu_l1_l + nu_one_l)/2

        zu = nu_pos_u + nu_neg_l
        zl = nu_neg_u + nu_pos_l
        return zl,zu

# L2 balls
class L2Ball(DualObject):
    def __init__(self, X, epsilon):
        super(L2Ball, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l2 = nu_1.norm(2, 1)

        if isinstance(epsilon, torch.Tensor):
            while epsilon.dim() < nu_x.dim():
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l2,
                nu_x + epsilon*l2)

    def objective(self, *nus):
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor):
            while epsilon.dim() < nu.dim()-1:
                epsilon = epsilon.unsqueeze(1)

        l2 = nu.norm(2,2)
        return -nu_x - epsilon*l2


class L2BallProj(L2Ball):
    def __init__(self, X, epsilon, k):
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu = [X.new(1,k,*X.size()[1:]).normal_()]

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else:
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        k = nu.size(1)
        l2 = nu.norm(2, 1)/(k**0.5)

        return (nu_x - self.epsilon*l2,
                nu_x + self.epsilon*l2)



def select_input(X, epsilon, proj, norm, bounded_input, l=0,u=1):
    if proj is not None and norm=='l1_median' and X[0].numel() > proj:
        if bounded_input:
            return InfBallProjBounded(X,epsilon,proj, l=l, u=u)
        else:
            return InfBallProj(X,epsilon,proj)
    elif norm == 'l1':
        if bounded_input:
            return InfBallBounded(X, epsilon, l=l, u=u)
        else:
            return InfBall(X, epsilon)
    elif proj is not None and norm=='l2_normal' and X[0].numel() > proj:
        return L2BallProj(X,epsilon,proj)
    elif norm == 'l2':
        return L2Ball(X,epsilon)
    else:
        raise ValueError("Unknown estimation type: {}".format(norm))

class InfBall(DualObject):
    def __init__(self, X, epsilon):
        super(InfBall, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l1 = nu_1.abs().sum(1)
        if isinstance(epsilon, torch.Tensor):
            while epsilon.dim() < nu_x.dim():
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l1,
                nu_x + epsilon*l1)

    def objective(self, *nus):
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor):
            while epsilon.dim() < nu.dim()-1:
                epsilon = epsilon.unsqueeze(1)
        l1 = (epsilon*nu.abs()).sum(2)
        return -nu_x - l1

class InfBallBounded(DualObject):
    def __init__(self, X, epsilon, l=0, u=1):
        super(InfBallBounded, self).__init__()
        self.epsilon = epsilon
        if torch.is_tensor(l):
            l_ = torch.max(X-epsilon,l)
        else:
            l_ = (X-epsilon).clamp(min=l)
        if torch.is_tensor(u):
            u_ = torch.min(X+epsilon,u)
        else:
            u_ = (X+epsilon).clamp(max=u)

        self.l = l_.view(X.size(0), 1, -1)
        self.u = u_.view(X.size(0), 1, -1)

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu_1[-1]
        else:
            nu = network(self.nu_1[0])
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)

        zu = (self.u.matmul(nu_pos) + self.l.matmul(nu_neg)).squeeze(1)
        zl = (self.u.matmul(nu_neg) + self.l.matmul(nu_pos)).squeeze(1)
        return (zl.view(zl.size(0), *nu.size()[2:]),
                zu.view(zu.size(0), *nu.size()[2:]))

    def objective(self, *nus):
        nu = nus[-1]
        nu_pos = nu.clamp(min=0).view(nu.size(0), nu.size(1), -1)
        nu_neg = nu.clamp(max=0).view(nu.size(0), nu.size(1), -1)
        u, l = self.u.unsqueeze(3).squeeze(1), self.l.unsqueeze(3).squeeze(1)
        return (-nu_neg.matmul(l) - nu_pos.matmul(u)).squeeze(2)

class InfBallProj(InfBall):
    def __init__(self, X, epsilon, k):
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu = [X.new(1,k,*X.size()[1:]).cauchy_()]

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else:
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        l1 = torch.median(self.nu[-1].abs(), 1)[0]
        return (nu_x - self.epsilon*l1,
                nu_x + self.epsilon*l1)

class InfBallProjBounded(InfBallProj):
    def __init__(self, X, epsilon, k, l=0, u=1):
        self.epsilon = epsilon

        self.nu_one_l = [(X-epsilon).clamp(min=l)]
        self.nu_one_u = [(X+epsilon).clamp(max=u)]
        self.nu_x = [X]

        self.l = self.nu_one_l[-1].view(X.size(0), 1, -1)
        self.u = self.nu_one_u[-1].view(X.size(0), 1, -1)

        n = X[0].numel()
        R = X.new(1,k,*X.size()[1:]).cauchy_()
        self.nu_l = [R * self.nu_one_l[-1].unsqueeze(1)]
        self.nu_u = [R * self.nu_one_u[-1].unsqueeze(1)]

    def apply(self, dual_layer):
        self.nu_l.append(dual_layer(*self.nu_l))
        self.nu_one_l.append(dual_layer(*self.nu_one_l))
        self.nu_u.append(dual_layer(*self.nu_u))
        self.nu_one_u.append(dual_layer(*self.nu_one_u))

    def bounds(self, network=None):
        if network is None:
            nu_u = self.nu_u[-1]
            nu_one_u = self.nu_one_u[-1]
            nu_l = self.nu_l[-1]
            nu_one_l = self.nu_one_l[-1]
        else:
            nu_u = network(self.nu_u[0])
            nu_one_u = network(self.nu_one_u[0])
            nu_l = network(self.nu_l[0])
            nu_one_l = network(self.nu_one_l[0])

        nu_l1_u = torch.median(nu_u.abs(),1)[0]
        nu_pos_u = (nu_l1_u + nu_one_u)/2
        nu_neg_u = (-nu_l1_u + nu_one_u)/2

        nu_l1_l = torch.median(nu_l.abs(),1)[0]
        nu_pos_l = (nu_l1_l + nu_one_l)/2
        nu_neg_l = (-nu_l1_l + nu_one_l)/2

        zu = nu_pos_u + nu_neg_l
        zl = nu_neg_u + nu_pos_l
        return zl,zu

# L2 balls
class L2Ball(DualObject):
    def __init__(self, X, epsilon):
        super(L2Ball, self).__init__()
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu_1 = [X.new(n,n)]
        torch.eye(n, out=self.nu_1[0])
        self.nu_1[0] = self.nu_1[0].view(-1,*X.size()[1:]).unsqueeze(0)

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu_1.append(dual_layer(*self.nu_1))

    def bounds(self, network=None):
        if network is None:
            nu_1 = self.nu_1[-1]
            nu_x = self.nu_x[-1]
        else:
            nu_1 = network(self.nu_1[0])
            nu_x = network(self.nu_x[0])

        epsilon = self.epsilon
        l2 = nu_1.norm(2, 1)

        if isinstance(epsilon, torch.Tensor):
            while epsilon.dim() < nu_x.dim():
                epsilon = epsilon.unsqueeze(1)

        return (nu_x - epsilon*l2,
                nu_x + epsilon*l2)

    def objective(self, *nus):
        epsilon = self.epsilon
        nu = nus[-1]
        nu = nu.view(nu.size(0), nu.size(1), -1)
        nu_x = nu.matmul(self.nu_x[0].view(self.nu_x[0].size(0),-1).unsqueeze(2)).squeeze(2)
        if isinstance(self.epsilon, torch.Tensor):
            while epsilon.dim() < nu.dim()-1:
                epsilon = epsilon.unsqueeze(1)

        l2 = nu.norm(2,2)
        return -nu_x - epsilon*l2


class L2BallProj(L2Ball):
    def __init__(self, X, epsilon, k):
        DualObject.__init__(self)
        self.epsilon = epsilon

        n = X[0].numel()
        self.nu_x = [X]
        self.nu = [X.new(1,k,*X.size()[1:]).normal_()]

    def apply(self, dual_layer):
        self.nu_x.append(dual_layer(*self.nu_x))
        self.nu.append(dual_layer(*self.nu))

    def bounds(self, network=None):
        if network is None:
            nu = self.nu[-1]
            nu_x = self.nu_x[-1]
        else:
            nu = network(self.nu[0])
            nu_x = network(self.nu_x[0])

        k = nu.size(1)
        l2 = nu.norm(2, 1)/(k**0.5)

        return (nu_x - self.epsilon*l2,
                nu_x + self.epsilon*l2)










def select_layer(layer, dual_net, X, proj, norm_type, in_f, out_f, zsi,
                 zl=None, zu=None):
    if isinstance(layer, nn.Linear):
        return DualLinear(layer, out_f)
    elif isinstance(layer, nn.Conv2d):
        return DualConv2d(layer, out_f)
    elif isinstance(layer, nn.ReLU):
        if zl is None and zu is None:
            zl, zu = zip(*[l.bounds() for l in dual_net])
            zl, zu = sum(zl), sum(zu)
        if zl is None or zu is None:
            raise ValueError("Must either provide both l,u bounds or neither.")
        I = ((zu > 0).detach() * (zl < 0).detach())
        if proj is not None and (norm_type=='l1_median' or norm_type=='l2_normal') and I.sum().item() > proj:
            return DualReLUProj(zl, zu, proj)
        else:
            return DualReLU(zl, zu)

    elif 'Flatten' in (str(layer.__class__.__name__)):
        return DualReshape(in_f, out_f)
    elif isinstance(layer, Dense):
        return DualDense(layer, dual_net, out_f)
    elif isinstance(layer, nn.BatchNorm2d):
        return DualBatchNorm2d(layer, zsi, out_f)
    else:
        print(layer)
        raise ValueError("No module for layer {}".format(str(layer.__class__.__name__)))

def batch(A, n):
    return A.view(n, -1, *A.size()[1:])
def unbatch(A):
    return A.view(-1, *A.size()[2:])

class DualLinear(DualLayer):
    def __init__(self, layer, out_features):
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.layer = layer
        if layer.bias is None:
            self.bias = None
        else:
            self.bias = [full_bias(layer, out_features[1:])]

    def apply(self, dual_layer):
        if self.bias is not None:
            self.bias.append(dual_layer(*self.bias))

    def bounds(self, network=None):
        if self.bias is None:
            return 0,0
        else:
            if network is None:
                b = self.bias[-1]
            else:
                b = network(self.bias[0])
            if b is None:
                return 0,0
            return b,b

    def objective(self, *nus):
        if self.bias is None:
            return 0
        else:
            nu = nus[-2]
            nu = nu.view(nu.size(0), nu.size(1), -1)
            return -nu.matmul(self.bias[0].view(-1))

    def forward(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        return F.linear(x, self.layer.weight)

    def T(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        return F.linear(x, self.layer.weight.t())

# Convolutional helper functions to minibatch large inputs for CuDNN
def conv2d(x, *args, **kwargs):
    """ Minibatched inputs to conv2d """
    i = 0
    out = []
    batch_size = 10000
    while i < x.size(0):
        out.append(F.conv2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

def conv_transpose2d(x, *args, **kwargs):
    i = 0
    out = []
    batch_size = 10000
    while i < x.size(0):
        out.append(F.conv_transpose2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

class DualConv2d(DualLinear):
    def __init__(self, layer, out_features):
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("Expected nn.Conv2d input.")
        self.layer = layer
        if layer.bias is None:
            self.bias = None
        else:
            self.bias = [full_bias(layer, out_features[1:]).contiguous()]

    def forward(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        if xs[-1].dim() == 5:
            n = x.size(0)
            x = unbatch(x)
        out = conv2d(x, self.layer.weight,
                       stride=self.layer.stride,
                       padding=self.layer.padding)
        if xs[-1].dim() == 5:
            out = batch(out, n)
        return out

    def T(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        if xs[-1].dim() == 5:
            n = x.size(0)
            x = unbatch(x)
        out = conv_transpose2d(x, self.layer.weight,
                                 stride=self.layer.stride,
                                 padding=self.layer.padding)
        if xs[-1].dim() == 5:
            out = batch(out, n)
        return out

class DualReshape(DualLayer):
    def __init__(self, in_f, out_f):
        super(DualReshape, self).__init__()
        self.in_f = in_f[1:]
        self.out_f = out_f[1:]

    def forward(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        shape = x.size()[:-len(self.in_f)] + self.out_f
        return x.view(shape)

    def T(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        shape = x.size()[:-len(self.out_f)] + self.in_f
        return x.view(shape)

    def apply(self, dual_layer):
        pass

    def bounds(self, network=None):
        return 0,0

    def objective(self, *nus):
        return 0

class DualReLU(DualLayer):
    def __init__(self, zl, zu):
        super(DualReLU, self).__init__()


        d = (zl >= 0).detach().type_as(zl)
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        n = d[0].numel()
        if I.sum().item() > 0:
            self.I_empty = False
            self.I_ind = I.view(-1,n).nonzero()


            self.nus = [zl.new(I.sum().item(), n).zero_()]
            self.nus[-1].scatter_(1, self.I_ind[:,1,None], d[I][:,None])
            self.nus[-1] = self.nus[-1].view(-1, *(d.size()[1:]))
            self.I_collapse = zl.new(self.I_ind.size(0),zl.size(0)).zero_()
            self.I_collapse.scatter_(1, self.I_ind[:,0][:,None], 1)
        else:
            self.I_empty = True

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu

    def apply(self, dual_layer):
        if self.I_empty:
            return
        if isinstance(dual_layer, DualReLU):
            self.nus.append(dual_layer(*self.nus, I_ind=self.I_ind))
        else:
            self.nus.append(dual_layer(*self.nus))

    def bounds(self, network=None):
        if self.I_empty:
            return 0,0
        if network is None:
            nu = self.nus[-1]
        else:
            nu = network(self.nus[0])
        if nu is None:
            return 0,0
        size = nu.size()
        nu = nu.view(nu.size(0), -1)
        zlI = self.zl[self.I]
        zl = (zlI * (-nu.t()).clamp(min=0)).mm(self.I_collapse).t().contiguous()
        zu = -(zlI * nu.t().clamp(min=0)).mm(self.I_collapse).t().contiguous()

        zl = zl.view(-1, *(size[1:]))
        zu = zu.view(-1, *(size[1:]))
        return zl,zu

    def objective(self, *nus):
        nu_prev = nus[-1]
        if self.I_empty:
            return 0
        n = nu_prev.size(0)
        nu = nu_prev.view(n, nu_prev.size(1), -1)
        zl = self.zl.view(n, -1)
        I = self.I.view(n, -1)
        return (nu.clamp(min=0)*zl.unsqueeze(1)).matmul(I.type_as(nu).unsqueeze(2)).squeeze(2)


    def forward(self, *xs, I_ind=None):
        x = xs[-1]
        if x is None:
            return None

        if self.d.is_cuda:
            d = self.d.cuda(device=x.get_device())
        else:
            d = self.d
        if x.dim() > d.dim():
            d = d.unsqueeze(1)

        if I_ind is not None:
            I_ind = I_ind.to(dtype=torch.long, device=x.device)
            return d[I_ind[:,0]]*x
        else:
            return d*x

    def T(self, *xs):
        return self(*xs)


class DualReLUProj(DualReLU):
    def __init__(self, zl, zu, k):
        DualLayer.__init__(self)
        d = (zl >= 0).detach().type_as(zl)
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        n = I.size(0)

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu

        if I.sum().item() == 0:
            warnings.warn('ReLU projection has no origin crossing activations')
            self.I_empty = True
            return
        else:
            self.I_empty = False

        nu = zl.new(n, k, *(d.size()[1:])).zero_()
        nu_one = zl.new(n, *(d.size()[1:])).zero_()
        if  I.sum() > 0:
            nu[I.unsqueeze(1).expand_as(nu)] = nu.new(I.sum().item()*k).cauchy_()
            nu_one[I] = 1
        nu = zl.unsqueeze(1)*nu
        nu_one = zl*nu_one

        self.nus = [d.unsqueeze(1)*nu]
        self.nu_ones = [d*nu_one]

    def apply(self, dual_layer):
        if self.I_empty:
            return
        self.nus.append(dual_layer(*self.nus))
        self.nu_ones.append(dual_layer(*self.nu_ones))

    def bounds(self, network=None):
        if self.I_empty:
            return 0,0

        if network is None:
            nu = self.nus[-1]
            no = self.nu_ones[-1]
        else:
            nu = network(self.nus[0])
            no = network(self.nu_ones[0])

        n = torch.median(nu.abs(), 1)[0]

        # From notes:
        # \sum_i l_i[nu_i]_+ \approx (-n + no)/2
        # which is the negative of the term for the upper bound
        # for the lower bound, use -nu and negate the output, so
        # (n - no)/2 since the no term flips twice and the l1 term
        # flips only once.
        zl = (-n - no)/2
        zu = (n - no)/2

        return zl,zu

class DualDense(DualLayer):
    def __init__(self, dense, net, out_features):
        super(DualDense, self).__init__()
        self.duals = nn.ModuleList([])
        for i,W in enumerate(dense.Ws):
            if isinstance(W, nn.Conv2d):
                dual_layer = DualConv2d(W, out_features)
            elif isinstance(W, nn.Linear):
                dual_layer = DualLinear(W, out_features)
            elif isinstance(W, nn.Sequential) and len(W) == 0:
                dual_layer = Identity()
            elif W is None:
                dual_layer = None
            else:
                print(W)
                raise ValueError("Don't know how to parse dense structure")
            self.duals.append(dual_layer)

            if i < len(dense.Ws)-1 and W is not None:
                idx = i-len(dense.Ws)+1
                # dual_ts needs to be len(dense.Ws)-i long
                net[idx].dual_ts = nn.ModuleList([dual_layer] + [None]*(len(dense.Ws)-i-len(net[idx].dual_ts)-1) + list(net[idx].dual_ts))

        self.dual_ts = nn.ModuleList([self.duals[-1]])


    def forward(self, *xs):
        duals = list(self.duals)[-min(len(xs),len(self.duals)):]
        if all(W is None for W in duals):
            return None
        # recursively apply the dense sub-layers
        out = [W(*xs[:i+1])
            for i,W in zip(range(-len(duals) + len(xs), len(xs)),
                duals) if W is not None]

        # remove the non applicable outputs
        out = [o for o in out if o is not None]

        # if no applicable outputs, return None
        if len(out) == 0:
            return None

        # otherwise, return the sum of the outputs
        return sum(o for o in out if o is not None)

    def T(self, *xs):
        dual_ts = list(self.dual_ts)[-min(len(xs),len(self.dual_ts)):]
        if all(W is None for W in dual_ts):
            return None

        # recursively apply the dense sub-layers
        out = [W.T(*xs[:i+1])
            for i,W in zip(range(-len(dual_ts) + len(xs), len(xs)),
                dual_ts) if W is not None]
        # remove the non applicable outputs
        out = [o for o in out if o is not None]

        # if no applicable outputs, return None
        if len(out) == 0:
            return None

        # otherwise, return the sum of the outputs
        return sum(o for o in out if o is not None)


    def apply(self, dual_layer):
        for W in self.duals:
            if W is not None:
                W.apply(dual_layer)

    def bounds(self, network=None):
        fvals = list(W.bounds(network=network) for W in self.duals
                        if W is not None)
        l,u = zip(*fvals)
        return sum(l), sum(u)

    def objective(self, *nus):
        fvals = list(W.objective(*nus) for W in self.duals if W is not None)
        return sum(fvals)

class DualBatchNorm2d(DualLayer):
    def __init__(self, layer, minibatch, out_features):
        if layer.training:
            minibatch = minibatch.data.transpose(0,1).contiguous()
            minibatch = minibatch.view(minibatch.size(0), -1)
            mu = minibatch.mean(1)
            var = minibatch.var(1)
        else:
            mu = layer.running_mean
            var = layer.running_var

        eps = layer.eps

        weight = layer.weight
        bias = layer.bias
        denom = torch.sqrt(var + eps)

        self.D = (weight/denom).unsqueeze(1).unsqueeze(2)
        self.ds = [((bias - weight*mu/denom).unsqueeze(1).unsqueeze
            (2)).expand(out_features[1:]).contiguous()]


    def forward(self, *xs):
        x = xs[-1]
        if x is None:
            return None
        return self.D*x

    def T(self, *xs):
        if x is None:
            return None
        return self(*xs)

    def apply(self, dual_layer):
        self.ds.append(dual_layer(*self.ds))

    def bounds(self, network=None):
        if network is None:
            d = self.ds[-1]
        else:
            d = network(self.ds[0])
        return d, d

    def objective(self, *nus):
        nu = nus[-2]
        d = self.ds[0].view(-1)
        nu = nu.view(nu.size(0), nu.size(1), -1)
        return -nu.matmul(d)

class Identity(DualLayer):
    def forward(self, *xs):
        return xs[-1]

    def T(self, *xs):
        return xs[-1]

    def apply(self, dual_layer):
        pass

    def bounds(self, network=None):
        return 0,0

    def objective(self, *nus):
        return 0


