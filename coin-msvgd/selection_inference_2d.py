import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib import rc

import math
from functools import reduce
from scipy import stats
from operator import mul
from collections import defaultdict
import seaborn as sns

from constrained.nonnegative.entropic import NonnegativeEntropicMap
from constrained.sampling.svgd import svgd_update, proj_svgd_update
from constrained.sampling.svmd import svmd_update, svmd_update_v2
from constrained.sampling.kernel import imq, rbf
from constrained.target import Target
from utils import energy_dist

# set random seed
np.random.seed(123)
tf.random.set_seed(123)

# define target
class SelectiveTarget(Target):
    def __init__(self, A, b, noise_scale):
        self.A = tf.constant(A, dtype=tf.float64)
        self.b = tf.constant(b, dtype=tf.float64)
        self.noise_scale = tf.constant(noise_scale, dtype=tf.float64)
        super(SelectiveTarget, self).__init__(NonnegativeEntropicMap())

    def logp(self, theta):
        # theta: [..., K, D]
        # ret: [..., K]
        recon = tf.einsum("...ki,ji->...kj", theta, self.A) + self.b
        return -0.5 * tf.reduce_sum(tf.square(recon), axis=-1) / tf.square(self.noise_scale)

    def grad_logp(self, theta):
        # ret: [..., K, D]
        # recon = theta @ tf.transpose(self.A) + self.b
        recon = tf.einsum("...ki,ji->...kj", theta, self.A) + self.b
        # ret = -recon @ self.A / tf.square(self.noise_scale)
        ret = -tf.einsum("...kj,ji->...ki", recon, self.A) / tf.square(self.noise_scale)
#         tf.print(ret)
#         tf.print("check:", super(QuadraticTarget, self).nabla_psi_inv_grad_logp(theta))
        return ret


# run experiment
if __name__ == "__main__":

    # load in data
    data = np.load("psi/sel.npz")
    A = data["linear"]
    b = np.squeeze(data["offset"], -1)
    noise_scale = data["noise_scale"]
    theta_init = data["init"]
    print("A:", A.shape)
    print("b:", b.shape)
    print("A:", A)
    print("b:", b)
    print("theta_init:", theta_init)
    print("noise_scale:", noise_scale)
    # noise_scale = 2.810999
    -0.5 / noise_scale**2


    K = 50 # number of particles
    D = b.shape[0] # dimension
    print("K:", K)
    print("D:", D)

    # define target
    target = SelectiveTarget(A, b, noise_scale)

    # initialise the HMC transition kernel.
    num_results = 100
    num_burnin_steps = 1000
    num_steps_between_results = 5
    adaptive_hmc = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=lambda x: target.dual_logp(x),
    #         num_leapfrog_steps=10,
            step_size=0.01)#,
        #num_adaptation_steps=int(2000))


    # run the chain (with burn-in).
    @tf.function(experimental_compile=True)
    def run_chain():
        samples, pkr = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=num_steps_between_results,
            current_state=np.zeros([10, D], dtype=np.float64),
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr)
        return samples, pkr

    eta_samples, pkr = run_chain()

    # is_accepted = pkr.inner_results.is_accepted
    r_hat = tfp.mcmc.potential_scale_reduction(eta_samples)
    # print("chain:", eta_samples.shape)
    # print("acceptance rate:", is_accepted.numpy().mean())
    # print("R-hat diagnostic (per dim):", r_hat.numpy())

    theta_hmc = target.mirror_map.nabla_psi_star(tf.reshape(eta_samples, [-1, D]))

    # plot samples
    plot_hmc_samples = False
    if plot_hmc_samples:
        f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
        theta_hmc_np = theta_hmc.numpy()[:, :2]
        ax.scatter(theta_hmc_np[:, 0], theta_hmc_np[:, 1], alpha=.6, c="g", s=20)
        ax.set_xlim([0, 0.8])
        ax.set_ylim([0.8, 2.2])

    # all methods
    def run(target, ground_truth_set, method="smvd", lr=0.005, n_chain=10, eds_freq=5, n_iters=1000):
        # g = tf.random.Generator.from_seed(1)
        eta0 = target.mirror_map.nabla_psi(theta_init[None, :]) + tf.random.normal([n_chain, K, D], dtype=tf.float64)
        # eta: [n_chain, K, D - 1]
        eta = tf.Variable(eta0)
        theta0 = target.mirror_map.nabla_psi_star(eta0)
        if method == "proj_svgd":
            theta = tf.Variable(theta0)
        else:
            theta = theta0
            # eta: [n_chain, K, D - 1]
            eta = tf.Variable(eta0)
        kernel = imq
        if method == "coin-svmd" or method == "coin-svgd":
            L = 0
            eta_grad_sum = 0
            reward = 0
            abs_eta_grad_sum = 0
        if method == "proj-coin-svgd":
            L = 0
            theta_grad_sum = 0
            reward = 0
            abs_theta_grad_sum = 0
        eds = []
        trange = tqdm(range(n_iters))
        optimizer = tf.keras.optimizers.RMSprop(lr)

        for t in trange:
            if method == "svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.99)
            elif method == "coin-svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.99)
            elif method == "svgd":
                eta_grad = svgd_update(target, eta, theta, kernel)
            elif method == "coin-svgd":
                eta_grad = svgd_update(target, eta, theta, kernel)
            elif method == "proj_svgd":
                theta_grad = proj_svgd_update(target, theta, kernel)
            elif method == "proj-coin-svgd":
                theta_grad = proj_svgd_update(target, theta, kernel)
            else:
                raise NotImplementedError()

            if method == "proj_svgd":
                optimizer.apply_gradients([(-theta_grad, theta)])
                theta.assign(tf.maximum(theta, 0.))
            elif method == "proj-coin-svgd":
                abs_theta_grad = abs(theta_grad)
                L = tf.maximum(abs_theta_grad, L)
                theta_grad_sum += theta_grad
                abs_theta_grad_sum += abs_theta_grad
                reward = tf.maximum(reward + tf.multiply(theta - theta0, theta_grad), 0)
                theta = theta0 + theta_grad_sum / (L * (abs_theta_grad_sum + L)) * (L + reward)
                theta = tf.Variable(theta)
                theta.assign(tf.maximum(theta, 0.))
            elif method == "coin-svmd" or method == "coin-svgd":
                abs_eta_grad = abs(eta_grad)
                L = tf.maximum(abs_eta_grad, L)
                eta_grad_sum += eta_grad
                abs_eta_grad_sum += abs_eta_grad
                reward = tf.maximum(reward + tf.multiply(eta - eta0, eta_grad), 0)
                eta = eta0 + eta_grad_sum / (L * (abs_eta_grad_sum + L)) * (L + reward)
                theta = target.mirror_map.nabla_psi_star(eta)
            else:
                optimizer.apply_gradients([(-eta_grad, eta)])
                theta = target.mirror_map.nabla_psi_star(eta)
            if t % eds_freq == 0:
                ed = energy_dist(ground_truth_set, theta[0])
                eds.append(ed.numpy())
        return tf.reshape(theta, [-1, theta.shape[-1]]), eds


    # compare methods for different learning rates (plot energy distance vs iterations)
    run_method_comparison = False
    if run_method_comparison:
        methods = ["coin-svgd", "proj-coin-svgd", "svgd", "proj_svgd"] # "svmd", "coin-svmd"

        ground_truth_set = theta_hmc
        theta_hmc_np = theta_hmc.numpy()

        search_lr = [0.1, 0.05, 0.01, 0.005]
        all_lr_samples_dict = defaultdict(list)
        all_lr_eds_dict = defaultdict(list)
        eds_freq = 10
        n_iters = 1000

        reps = 5
        for i, lr in enumerate(search_lr):
            print("LR: " + str(i + 1) + "/" + str(len(search_lr)))
            samples_dict = defaultdict(list)
            eds_dict = defaultdict(list)
            for rep in range(reps):
                print("Repeat: " + str(rep + 1) + "/" + str(reps))
                for method in methods:
                    theta, eds = run(target, ground_truth_set, method=method, lr=lr, eds_freq=eds_freq, n_iters=n_iters)
                    eds_dict[method].append(eds)
                    samples_dict[method].append(theta)
            all_lr_samples_dict[lr].append(samples_dict)
            all_lr_eds_dict[lr].append(eds_dict)

        name_map = {
            "svgd": "MSVGD",
            "proj_svgd": "Proj. SVGD",
            "coin-svgd": "Coin MSVGD",
            "proj-coin-svgd": "Proj. Coin SVGD",
            "truth": "Truth"
        }

        linestyle_map = {
            0.1: 'dotted',
            0.05: "dashed",
            0.01: "dashdot",
            0.005: (0, (3, 5, 1, 5))
        }

        color_map = {
            0.1: "C1",
            0.05: "C2",
            0.01: "C3",
            0.005: "C4",
        }

        coin_methods = methods[0:2]
        non_coin_methods = methods[2:4]

        for i, (method_coin, method) in enumerate(zip(coin_methods, non_coin_methods)):
            f, axes = plt.subplots(1, 1, figsize=(6, 4))
            iters = np.arange(len(all_lr_eds_dict[search_lr[0]][0][method_coin][0])) * eds_freq
            eds_mean = np.mean(np.array(all_lr_eds_dict[search_lr[0]][0][method_coin]), axis=0)
            eds_upper = np.max(np.array(all_lr_eds_dict[search_lr[0]][0][method_coin]), axis=0)
            eds_lower = np.min(np.array(all_lr_eds_dict[search_lr[0]][0][method_coin]), axis=0)
            axes.plot(iters, eds_mean, label=name_map[method_coin], zorder=10)
            axes.fill_between(iters, eds_upper, eds_lower, alpha=0.2)
            for j, lr in enumerate(search_lr):
                iters = np.arange(len(all_lr_eds_dict[search_lr[0]][0][method][0])) * eds_freq
                eds_mean = np.mean(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                eds_upper = np.max(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                eds_lower = np.min(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                axes.plot(iters, eds_mean, linestyle=linestyle_map[lr], label=name_map[method] + " (LR={})".format(lr),
                          color=color_map[lr])
                axes.fill_between(iters, eds_upper, eds_lower, alpha=0.2, color=color_map[lr])
                # axes.set_title(name_map[method])
                axes.set_ylim(top=0.8, bottom=0)
                axes.set_xlabel("Iterations")
                axes.set_ylabel("Energy Distance")
            axes.legend(ncol=2)
            plt.savefig("results/selection_inference_2d/iter_vs_eds_wrt_lr_{}.pdf".format(method),
                        bbox_inches="tight", dpi=300)
        #plt.show()