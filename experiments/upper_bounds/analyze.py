import numpy as np
# import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
from scriptify import scriptify

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str, rs_data=True):
        self.data_file_path = data_file_path
        self.rs_data=rs_data

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        if self.rs_data:
            df = pd.read_csv(self.data_file_path, delimiter="\t")
        else:
            df = pd.read_csv(self.data_file_path, header=0,delimiter=",")

        return np.array([self.at_radius(df, radius) for radius in radii])


    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class RobustnessReductionRatio(Accuracy):
    def __init__(self, data_file_path: str, rs_data=True):
        self.data_file_path = data_file_path
        self.rs_data=rs_data

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        if self.rs_data:
            df = pd.read_csv(self.data_file_path, delimiter="\t")
        else:
            df = pd.read_csv(self.data_file_path, header=0,delimiter=",")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean_rad = (df["radius"] >= radius).mean()
        mean_2rad = (df["radius"] >= 2*radius).mean()
        reduction_ratio = (1 - mean_2rad/mean_rad)

        return reduction_ratio
    
class RobustnessEpsilon(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean_rad = (df["radius"] >= radius).mean()
        return mean_rad

class Robustness2Epsilon(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean_2rad = (df["radius"] >= 2*radius).mean()
        return mean_2rad

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01, ylabel='certified accuracy',legend='lower right',) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend([method.legend for method in lines], loc=legend, fontsize=16)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()



if __name__ == "__main__":

    @scriptify
    def script():
        plot_certified_accuracy(
            outfile="./plots/gloro/mnist_vra", 
            title="MNIST, vary $\epsilon$", 
            max_radius=3.5, 
            ylabel="VRA",
            legend='upper right',
            lines=[
                Line(ApproximateAccuracy("./data/gloro/mnist/mnist_0.3_N",rs_data=False), "$\epsilon = 0.3$"),
                Line(ApproximateAccuracy("./data/gloro/mnist/mnist_0.3_Y",rs_data=False), "$\epsilon = 0.6$"),
                Line(ApproximateAccuracy("./data/gloro/mnist/mnist_1.58_N",rs_data=False), "$\epsilon = 1.58$"),
                Line(ApproximateAccuracy("./data/gloro/mnist/mnist_1.58_Y",rs_data=False), "$\epsilon = 3.16$")
            ])
        plot_certified_accuracy(
            outfile="./plots/gloro/mnist_fpr", 
            title="MNIST, vary $\epsilon$", 
            max_radius=3.5, 
            ylabel='false positive rate',
            lines=[
                Line(RobustnessReductionRatio("./data/gloro/mnist/mnist_0.3_N",rs_data=False), "$\epsilon = 0.3$"),
                Line(RobustnessReductionRatio("./data/gloro/mnist/mnist_0.3_Y",rs_data=False), "$\epsilon = 0.6$"),
                Line(RobustnessReductionRatio("./data/gloro/mnist/mnist_1.58_N",rs_data=False), "$\epsilon = 1.58$"),
                Line(RobustnessReductionRatio("./data/gloro/mnist/mnist_1.58_Y",rs_data=False), "$\epsilon = 3.16$")
            ])    

        plot_certified_accuracy(
            outfile="./plots/gloro/cifar10_vra", 
            title="CIFAR-10, vary $\epsilon$", 
            max_radius=1.5, 
            ylabel="VRA",
            legend='upper right',
            lines=[
                Line(ApproximateAccuracy("./data/gloro/cifar10/cifar10_0.14_N",rs_data=False), "$\epsilon = 0.14$"),
                Line(ApproximateAccuracy("./data/gloro/cifar10/cifar10_0.14_Y",rs_data=False), "$\epsilon = 0.28$")
            ])
        plot_certified_accuracy(
            outfile="./plots/gloro/cifar10_fpr", 
            title="CIFAR-10, vary $\epsilon$", 
            max_radius=1.5, 
            ylabel='false positive rate',
            lines=[
                Line(RobustnessReductionRatio("./data/gloro/cifar10/cifar10_0.14_N",rs_data=False), "$\epsilon = 0.14$"),
                Line(RobustnessReductionRatio("./data/gloro/cifar10/cifar10_0.14_Y",rs_data=False), "$\epsilon = 0.28$")
            ])

        plot_certified_accuracy(
            outfile = "./plots/rs/vary_noise_cifar10", 
            title = "CIFAR-10, vary $\sigma$", 
            max_radius = 2, 
            ylabel='VRA', 
            legend='upper right',
            lines = [
                Line(ApproximateAccuracy("./data/rs/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
                Line(ApproximateAccuracy("./data/rs/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
                Line(ApproximateAccuracy("./data/rs/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
                Line(ApproximateAccuracy("./data/rs/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
            ])

        plot_certified_accuracy(
            outfile = "./plots/rs/vary_noise_imagenet", 
            title = "ImageNet, vary $\sigma$", 
            max_radius = 2, 
            ylabel='VRA', 
            legend='upper right',
            lines = [
                Line(ApproximateAccuracy("./data/rs/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
                Line(ApproximateAccuracy("./data/rs/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
                Line(ApproximateAccuracy("./data/rs/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
            ])

        plot_certified_accuracy(
            outfile = "./plots/rs/RobustnessReductionRatio_cifar10", 
            title = "CIFAR-10, vary $\sigma$", 
            max_radius = 2, 
            ylabel='false positive rate', 
            lines = [
                Line(RobustnessReductionRatio("./data/rs/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
                Line(RobustnessReductionRatio("./data/rs/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
                Line(RobustnessReductionRatio("./data/rs/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
                Line(RobustnessReductionRatio("./data/rs/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
            ])
        

        plot_certified_accuracy(
            outfile = "./plots/rs/RobustnessReductionRatio_imagenet", 
            title = "ImageNet, vary $\sigma$", 
            max_radius = 2, 
            ylabel='false positive rate', 
            lines = [
                Line(RobustnessReductionRatio("./data/rs/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
                Line(RobustnessReductionRatio("./data/rs/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
                Line(RobustnessReductionRatio("./data/rs/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
            ])
        
    
