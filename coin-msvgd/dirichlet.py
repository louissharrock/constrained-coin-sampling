import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from functools import reduce
from scipy import stats
from operator import mul
from collections import defaultdict
from itertools import cycle

from constrained.simplex.entropic import SimplexEntropicMap, safe_reciprocal
from constrained.simplex.proj import euclidean_proj_simplex
from constrained.sampling.svgd import svgd_update, proj_svgd_update
from constrained.sampling.svmd import svmd_update_v2
from constrained.sampling.kernel import imq, rbf
from constrained.target import Target
from utils import energy_dist

# Set random seeds
np.random.seed(100)
tf.random.set_seed(100)

# Define Dirichlet target
class DirichletTarget(Target):
    def __init__(self, alpha):
        self.alpha = tf.constant(alpha, dtype=tf.float64)
        super(DirichletTarget, self).__init__(SimplexEntropicMap())

    @tf.function
    def grad_logp(self, theta):
        return (self.alpha[:-1] - 1.) * safe_reciprocal(theta) - (self.alpha[-1] - 1.) * safe_reciprocal(
            1. - tf.reduce_sum(theta, axis=-1, keepdims=True))

    @tf.function
    def nabla_psi_inv_grad_logp(self, theta):
        # ret: [K, D - 1]
        return self.alpha[:-1] - 1. - tf.reduce_sum(self.alpha - 1.) * theta

    @tf.function
    def dual_grad_logp(self, eta, theta=None):
        return self.alpha[:-1] - tf.reduce_sum(self.alpha) * theta

# Define plotting function
class PlotSimplex:
    def __init__(self, corners):
        self._corners = corners
        self._triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        # Midpoints of triangle sides opposite of each corner
        self._midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 for i in range(3)]

    def xy2bc(self, xy, tol=1.e-16):
        s = [(self._corners[i] - self._midpoints[i]).dot(xy - self._midpoints[i]) / 0.75 for i in range(3)]
        return s

    def draw_pdf_contours(self, ax, target, label=None, nlevels=100, subdiv=5, **kwargs):
        """Draw pdf contours for a Dirichlet distribution"""
        # Subdivide the triangle into a triangular mesh
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        triangles = trimesh.triangles
        xys = [(trimesh.x[tri].mean(), trimesh.y[tri].mean()) for tri in triangles]

        # convert to barycentric coordinates and compute probabilities of the given distribution
        pvals = [tf.exp(target.logp(np.array(self.xy2bc(xy))[None, :2])).numpy().squeeze(0) for xy in xys]

        # YlGnBu
        tcf = ax.tripcolor(trimesh, pvals, cmap="YlGnBu", antialiased=False, edgecolors="face")
        ax.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75 ** 0.5)
        if label is not None:
            ax.set_title(
                np.array2string(np.array(label), separator=", ", suppress_small=True, precision=2)[1:-1],
                fontsize=18
            )
        ax.axis('off')
        plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
        # ax.triplot(self._triangle, linewidth=1.5, color="k")
        return ax

    def plot_points(self, ax, X, barycentric=True, border=True, **kwargs):
        '''Plots a set of points in the simplex.
        Arguments:
            `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                           (if in barycentric coords) of points to plot.
            `barycentric` (bool): Indicates if `X` is in barycentric coords.
            `border` (bool): If True, the simplex border is drawn.
            kwargs: Keyword args passed on to `plt.plot`.
        '''
        if barycentric is True:
            X = X.dot(self._corners)
        ax.scatter(X[:, 0], X[:, 1], **kwargs)
        ax.axis('equal')
        ax.set_xlim(0 - 0.1, 1 + 0.1)
        ax.set_ylim(0 - 0.1, 0.75 ** 0.5 + 0.1)
        ax.axis('off')
        if border is True:
            ax.triplot(self._triangle, linewidth=1, c="k")

# Experiments
if __name__ == "__main__":

    K = 50 # number of particles
    D = 20 # number of dimensions
    alpha = np.ones(D, dtype=np.float64)
    alpha[:3] += np.array([90., 5., 5.])

    target = DirichletTarget(alpha)

    # true samples
    ground_truth_set = np.random.dirichlet(alpha, size=1000).astype(np.float64)
    ground_truth_set = tf.constant(ground_truth_set[:, :-1], dtype=tf.float64)

    if D == 3:
        ground_truth_set_np = ground_truth_set.numpy()
    else:
        ground_truth_set_np = ground_truth_set.numpy()[:, :2]

    # plot ground truth samples
    plot_samples = True
    if plot_samples:
        if D == 3:
            f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
            plot_simplex = PlotSimplex(corners)
            plot_simplex.draw_pdf_contours(ax, target)

        if D == 3:
            f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            plot_simplex.plot_points(ax, ground_truth_set.numpy(), alpha=.6, c="g", s=20)
        else:
            f, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            ax.scatter(ground_truth_set_np[:, 0], ground_truth_set_np[:, 1], alpha=.6, c="g", s=20)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 0.2])
        plt.show()
        plt.close("all")

    # all methods
    def run(target, ground_truth_set, method="smvd", lr=0.005, eds_freq=5, n_iters=500):
        q0 = tfp.distributions.Dirichlet(tf.ones(D, dtype=tf.float64) * 5)
        theta_full = q0.sample(K)
        # theta: [K, D - 1]
        theta0 = theta_full[:, :-1]
        if method == "proj_svgd" or method == "proj-coin-svgd":
            theta = tf.Variable(theta0)
        else:
            theta = theta0
            eta0 = target.mirror_map.nabla_psi(theta)
            # eta: [K, D - 1]
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
        #     optimizer = tf.keras.optimizers.SGD(lr)
        for t in trange:
            if method == "svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.98)
            # seem to need bigger eigenvalue cut-off to ensure stability for coin
            elif method == "coin-svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.9)
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
                theta_ext = tf.concat([theta, 1 - tf.reduce_sum(theta, axis=-1, keepdims=True)], -1)
                theta.assign(euclidean_proj_simplex(theta_ext)[..., :-1])
            elif method == "proj-coin-svgd":
                abs_theta_grad = abs(theta_grad)
                L = tf.maximum(abs_theta_grad, L)
                theta_grad_sum += theta_grad
                abs_theta_grad_sum += abs_theta_grad
                reward = tf.maximum(reward + tf.multiply(theta - theta0, theta_grad), 0)
                theta = theta0 + theta_grad_sum / (L * (abs_theta_grad_sum + L)) * (L + reward)
                theta = tf.Variable(theta)
                theta_ext = tf.concat([theta, 1 - tf.reduce_sum(theta, axis=-1, keepdims=True)], -1)
                theta.assign(euclidean_proj_simplex(theta_ext)[..., :-1])
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
                ed = energy_dist(ground_truth_set, theta)
                eds.append(ed.numpy())
        theta_ext = tf.concat([theta, 1 - tf.reduce_sum(theta, axis=-1, keepdims=True)], -1)
        return theta_ext, eds

    # methods to run
    methods = ["coin-svgd", "proj-coin-svgd", "svgd", "svmd", "proj_svgd"] # "coin-svmd"

    # names for plotting
    name_map = {
        "svgd": "MSVGD",
        "svmd": "SVMD",
        "proj_svgd": "Projected SVGD",
        "coin-svgd": "Coin MSVGD",
        "coin-svmd": "Coin SVMD",
        "proj-coin-svgd": "Projected Coin SVGD"
    }

    # compare all methods (energy distance vs iterations)
    compare_methods_plot = False
    if compare_methods_plot:

        methods = ["coin-svgd", "proj-coin-svgd", "svgd", "svmd", "proj_svgd"]

        linestyle_map = {
            "coin-svgd": "solid",
            "proj-coin-svgd": "dotted",
            "svgd": "dashed",
            "svmd": "dashdot",
            "proj_svgd": (0, (3, 5, 1, 5)),
        }
        color_map = {
            "coin-svgd": "C0",
            "proj-coin-svgd": "C1",
            "svgd": "C2",
            "svmd": "C3",
            "proj_svgd": "C4"
        }
        lr_map = {
            "svgd": 0.1,
            "svmd": 0.1,
            "proj_svgd": 1e-4,
            "coin-svgd": 1,  # placeholder
            "coin-svmd": 1,  # placeholder
            "proj-coin-svgd": 1  # placeholder
        }

        samples_dict = defaultdict(list)
        eds_dict = defaultdict(list)

        eds_freq = 5
        n_iters = 500

        reps = 5
        for rep in range(reps):
            print("Rep: " + str(rep + 1) + "/" + str(reps))
            for method in methods:
                theta, eds = run(target, ground_truth_set, method=method, lr=lr_map[method], eds_freq=eds_freq,
                                 n_iters=n_iters)
                eds_dict[method].append(eds)
                samples_dict[method].append(theta)

        f, ax = plt.subplots(1, 1, figsize=(6, 4))

        for i, method in enumerate(methods):
            if "coin" in method:
                linestyle = "solid"
                zorder = 10
            if "coin" not in method:
                linestyle = "dashed"
                zorder =1
            iter = np.arange(len(eds_dict[method][0])) * eds_freq
            eds_mean = np.mean(np.array(eds_dict[method]), axis=0)
            eds_upper = np.max(np.array(eds_dict[method]), axis=0)
            eds_lower = np.min(np.array(eds_dict[method]), axis=0)
            ax.plot(iter, eds_mean, linestyle=linestyle_map[method],
                    label="{}".format(name_map[method]), color=color_map[method], zorder=zorder)
            ax.fill_between(iter, eds_upper, eds_lower, color=color_map[method], alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Energy Distance")
        ax.legend()
        #plt.savefig("results/dirichlet/iter_vs_eds.pdf", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close("all")

    # compare methods for different step sizes (energy distance vs iterations)
    step_size_plot = False
    if step_size_plot:
        methods = ["coin-svgd", "svgd"]

        linestyle_map = {
            5e-1: "solid",
            1e-1: "dotted",
            1e-2: "dashed",
            1e-3: "dashdot",
            1e-4: (0, (3, 5, 1, 5)),
        }

        color_map = {
            5e-1: "C1",
            1e-1: "C2",
            1e-2: "C3",
            1e-3: "C4",
            1e-4: "C5",
        }
        search_lr = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
        all_lr_samples_dict = defaultdict(list)
        all_lr_eds_dict = defaultdict(list)

        eds_freq = 5
        n_iters = 500

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

        f, axes = plt.subplots(1, 1, figsize=(6, 4))

        for i, method in enumerate(methods):
            if "coin" in method:
                iter = np.arange(len(all_lr_eds_dict[5e-1][0][method][0])) * eds_freq
                eds_mean = np.mean(np.array(all_lr_eds_dict[5e-1][0][method]), axis=0)
                eds_upper = np.max(np.array(all_lr_eds_dict[5e-1][0][method]), axis=0)
                eds_lower = np.min(np.array(all_lr_eds_dict[5e-1][0][method]), axis=0)
                axes.plot(iter, eds_mean, label="Coin MSVGD", zorder=10)
                axes.fill_between(iter, eds_upper, eds_lower, alpha=0.2)
            if "coin" not in method:
                for j, lr in enumerate(search_lr):
                    eds_mean = np.mean(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                    eds_upper = np.max(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                    eds_lower = np.min(np.array(all_lr_eds_dict[lr][0][method]), axis=0)
                    axes.plot(iter, eds_mean, linestyle=linestyle_map[lr], label="MSVGD (LR={})".format(lr),
                              color=color_map[lr], zorder=1)
                    axes.fill_between(iter, eds_upper, eds_lower, alpha=0.2, color=color_map[lr])
                # axes.set_title(name_map[method])
                axes.set_ylim(top=1.8, bottom=0)
                axes.set_xlabel("Iterations")
                if i == 0:
                    axes.set_ylabel("Energy Distance")
            axes.legend(ncol=2)
        plt.savefig("results/dirichlet/iter_vs_eds_wrt_lr.pdf", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close("all")

    # compare methods as a function of learning rate (plot energy distance vs LR after 500 iterations)
    step_size_plot_2 = False
    if step_size_plot_2:

        methods = ["coin-svgd", "proj-coin-svgd", "svgd", "svmd", "proj_svgd"]

        linestyle_map = {
            "coin-svgd": "solid",
            "proj-coin-svgd": "dotted",
            "svgd": "dashed",
            "svmd": "dashdot",
            "proj_svgd": (0, (3, 5, 1, 5)),
        }
        color_map = {
            "coin-svgd": "C0",
            "proj-coin-svgd": "C1",
            "svgd": "C2",
            "svmd": "C3",
            "proj_svgd": "C4"
        }

        search_lr = np.logspace(-5, 0, 20)

        all_method_samples_dict = defaultdict(list)
        all_method_eds_dict = defaultdict(list)

        eds_freq = 100
        n_iters = 500

        reps = 5

        for j, method in enumerate(methods):
            print("method: " + str(j + 1) + "/" + str(len(methods)))
            samples_dict = defaultdict(list)
            eds_dict = defaultdict(list)
            for rep in range(reps):
                print("Repeat: " + str(rep + 1) + "/" + str(reps))
                for lr in search_lr:
                    theta, eds = run(target, ground_truth_set, method=method, lr=lr, eds_freq=eds_freq, n_iters=n_iters)
                    eds_dict[lr].append(eds[-1])
                    samples_dict[lr].append(theta)

            all_method_samples_dict[method].append(samples_dict)
            all_method_eds_dict[method].append(eds_dict)

        f, ax = plt.subplots(1, 1, figsize=(6, 4))

        for ii, method in enumerate(methods):
            if "coin" in method:
                eds_mean = np.mean(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                eds_upper = np.max(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                eds_lower = np.min(np.array(all_method_eds_dict[method][0][search_lr[0]]), axis=0)
                ax.axhline(eds_mean, label=name_map[method], color=color_map[method],
                           linestyle=linestyle_map[method])
                ax.fill_between(search_lr, eds_upper, eds_lower, color=color_map[method], alpha=0.2)
            if "coin" not in method:
                eds_mean = np.mean(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                eds_upper = np.max(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                eds_lower = np.min(np.array([all_method_eds_dict[method][0][lr] for lr in search_lr]), axis=1)
                ax.plot(search_lr, eds_mean, label=name_map[method], color=color_map[method],
                        linestyle=linestyle_map[method], marker=".")
                ax.fill_between(search_lr, eds_upper, eds_lower, alpha=0.2, color=color_map[method])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Energy Distance")
        ax.legend()
        ax.margins(x=0)
        plt.savefig("results/dirichlet/eds_vs_lr.pdf", bbox_inches="tight", dpi=150)
        plt.show()
        plt.close("all")
