import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle

import pandas as pd

from constrained.cube.entropic import UnitCubeEntropic
from constrained.sampling.svgd import svgd_update
from constrained.sampling.svmd import svmd_update_v2
from constrained.sampling.kernel import imq
from constrained.target import Target
from utils import energy_dist

np.random.seed(123)
tf.random.set_seed(123)

# define target
class UniformTarget(Target):
    def __init__(self):
        map = UnitCubeEntropic()
        super(UniformTarget, self).__init__(map)

    @tf.function
    def grad_logp(self, theta):
        return tf.zeros_like(theta)

    @tf.function
    def nabla_psi_inv_grad_logp(self, theta):
        return tf.zeros_like(theta)

# run experiments
if __name__ == "__main__":

    K = 100
    D = 2
    target = UniformTarget()

    rng = np.random.default_rng(12345)
    ground_truth_set = rng.uniform(-1., 1., size=(1000, D)).astype(np.float64)
    ground_truth_set = tf.constant(ground_truth_set, dtype=tf.float64)

    # plot ground truth samples
    plot_samples = False
    if plot_samples:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ground_truth_set_np = ground_truth_set.numpy()
        ax.scatter(ground_truth_set_np[:, 0], ground_truth_set_np[:, 1], alpha=.6, c="g", s=20)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        plt.close("all")

    # all methods
    def run(target, ground_truth_set, method="smvd", lr=0.005, eds_freq=50, n_iters=250, seed=42):
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(-0.5, 0.5, (K, D)).astype(np.float64)
        theta = theta0
        eta0 = target.mirror_map.nabla_psi(theta)
        # eta: [K, D]
        eta = tf.Variable(eta0)
        kernel = imq
        eds = []
        ed = energy_dist(ground_truth_set, theta)
        eds.append(ed.numpy())
        L = 0
        eta_grad_sum = 0
        reward = 0
        abs_eta_grad_sum = 0
        trange = tqdm(range(n_iters))
        optimizer = tf.keras.optimizers.RMSprop(lr)
        for t in trange:
            if method == "svmd" or method == "coin-svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.98, kernel_width2=0.01)
            elif method == "msvgd" or method == "coin-msvgd":
                eta_grad = svgd_update(target, eta, theta, kernel, kernel_width2=0.01)
            else:
                raise NotImplementedError()

            if "coin" in method:
                abs_eta_grad = abs(eta_grad)
                L = tf.maximum(abs_eta_grad, L)
                eta_grad_sum += eta_grad
                abs_eta_grad_sum += abs_eta_grad
                reward = tf.maximum(reward + tf.multiply(eta - eta0, eta_grad), 0)
                eta = eta0 + eta_grad_sum / (L * (abs_eta_grad_sum + L)) * (L + reward)
                theta = target.mirror_map.nabla_psi_star(eta)
            if "coin" not in method:
                optimizer.apply_gradients([(-eta_grad, eta)])
                theta = target.mirror_map.nabla_psi_star(eta)
            if t % eds_freq == 0:
                ed = energy_dist(ground_truth_set, theta)
                eds.append(ed.numpy())
        return theta, eds

    # methods to run
    methods = ["coin-msvgd", "coin-svmd", "msvgd", "svmd"]
    color_map = {
        "coin-msvgd": "C0",
        "proj-coin-svgd": "C3",
        "msvgd": "C1",
        "svmd": "C2",
        "proj_svgd": "C4",
        "coin-svmd": "C5"
    }
    name_map = {
        "msvgd": "MSVGD",
        "svmd": "SVMD",
        "coin-msvgd": "Coin MSVGD",
        "coin-svmd": "Coin SVMD"
    }

    # compare methods as a function of the learning rate (plot energy distance vs learning rate, after 250 iterations)
    step_size_plot = False
    methods = ["coin-msvgd", "msvgd"] # only compare Coin MSVGD and MSVGD here
    if step_size_plot:

        seeds = [0, 1, 2, 3, 4]
        search_lr = np.logspace(-5, 0, 20)

        all_method_samples_dict = defaultdict(list)
        all_method_eds_dict = defaultdict(list)

        reps = 5
        for j, method in enumerate(methods):
            print("method: " + str(j + 1) + "/" + str(len(methods)))
            samples_dict = defaultdict(list)
            eds_dict = defaultdict(list)
            for kk, rep in enumerate(range(reps)):
                print("Repeat: " + str(rep + 1) + "/" + str(reps))
                rng = np.random.default_rng(kk)
                ground_truth_set = rng.uniform(-1., 1., size=(1000, D)).astype(np.float64)
                ground_truth_set = tf.constant(ground_truth_set, dtype=tf.float64)

                for lr in search_lr:
                    theta, eds = run(target, ground_truth_set, method=method, lr=lr, eds_freq=50)
                    eds_dict[lr].append(eds[-1])
                    samples_dict[lr].append(theta)

            all_method_samples_dict[method].append(samples_dict)
            all_method_eds_dict[method].append(eds_dict)

        colors = ["C" + str(i) for i in range(len(methods))]
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        for ii, method in enumerate(methods):
            if "coin" in method:
                eds_mean = np.mean(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                eds_upper = np.max(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                eds_lower = np.min(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                ax.axhline(eds_mean, label=name_map[method], color=color_map[method])
                ax.fill_between(search_lr, eds_lower, eds_upper, color=color_map[method], alpha=0.2)
            if "coin" not in method and "svmd" not in method:
                eds_mean = np.mean(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                eds_upper = np.max(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                eds_lower = np.min(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                ax.plot(search_lr, eds_mean, ".-", label=name_map[method], color=color_map[method],
                        linestyle="dashed")
                ax.fill_between(search_lr, eds_lower, eds_upper, alpha=0.2, color=color_map[method])

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.tick_params(axis='both', labelsize=18)
        ax.set_xlabel("Learning Rate", fontsize=18)
        ax.set_ylabel("Energy Distance", fontsize=18)
        ax.legend(prop={'size': 16}, loc='upper right')
        ax.set_ylim(5e-4, 1e0)
        plt.margins(x=0)
        plt.savefig("results/uniform/eds_vs_lr_msvgd.pdf", bbox_inches="tight", dpi=300)
        plt.show()

    # compare methods: plot energy distance vs iterations, using a good choice of LR from the previous experiment
    compare_methods_plot = False
    if compare_methods_plot:

        samples_dict = {}
        eds_dict = {}
        eds_freq = 2

        lr_map = {
            "msvgd": 0.1,
            "svmd": 0.1,
            "coin-msvgd": 0.1,  # placeholder
            "coin-svmd": 0.1  # placeholder
        }

        eds_freq = 1
        for method in methods:
            theta, eds = run(target, ground_truth_set, method=method, lr=lr_map[method], eds_freq=eds_freq,
                             n_iters=250)
            eds_dict[method] = eds
            samples_dict[method] = theta

        f, ax = plt.subplots(1, 1, figsize=(5, 4))

        for i, method in enumerate(methods):
            if "coin" in method:
                linestyle = "solid"
            if "coin" not in method:
                linestyle = "dashed"
            ax.plot(np.arange(len(eds_dict[method])) * eds_freq, eds_dict[method],
                    label="{}".format(name_map[method]),
                    color=color_map[method], linestyle=linestyle)

        # ax.set_ylim(bottom=0)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Energy Distance")
        ax.set_yscale("log")
        ax.legend()
        # plt.savefig("results/uniform/eds_vs_iter.pdf", bbox_inches="tight", dpi=300)
        plt.show()

        # plot output samples
        if D == 2:
            plot_samples = False
            if plot_samples:
                f, ax = plt.subplots(1, 4, figsize=(4 * 4, 4))
                for i, method in enumerate(methods):
                    samples_np = samples_dict[method].numpy()
                    ax[i].scatter(samples_np[:, 0], samples_np[:, 1], alpha=.6, c="g", s=20)
                    ax[i].set_xlim([-1.2, 1.2])
                    ax[i].set_ylim([-1.2, 1.2])
                    ax[i].set_title(name_map[method])
                plt.savefig("results/uniform/samples_.pdf".format(method), bbox_inches="tight",
                            dpi=300)
                plt.show()


