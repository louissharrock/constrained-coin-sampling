import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
# os.environ['R_HOME'] = "/usr/lib/R"
import time

import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from rpy2 import rinterface
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from psi.target import SelectiveTarget
from psi.data import HIV_NRTI
from constrained.nonnegative.entropic import NonnegativeEntropicMap
from constrained.sampling.kernel import imq
from constrained.sampling.svgd import svgd_update
from constrained.sampling.svmd import svmd_update_v2

# run experiments
if __name__ == "__main__":

    devtools = importr("devtools")
    selectiveInference = importr("selectiveInference")
    r("load_all('R-software/selectiveInference')")
    glmnet = importr("glmnet")
    r("set.seed(1)")

    # all methods
    def run(target, theta_init, method="svmd", K=50, n_chain=100):
        D = theta_init.shape[-1]
        eta0 = .1*target.mirror_map.nabla_psi(theta_init[None, :]) + .1*tf.random.normal([n_chain, K, D], dtype=tf.float64)
        # eta: [n_chain, K, D - 1]
        eta = tf.Variable(eta0)
        theta = target.mirror_map.nabla_psi_star(eta)
        n_iters = 2000
        kernel = imq
        if "coin" in method:
            L = 0
            eta_grad_sum = 0
            reward = 0
            abs_eta_grad_sum = 0
        trange = tqdm(range(n_iters))
        optimizer = tf.keras.optimizers.RMSprop(0.01) #0.01 main results, 0.001 small lr, 1.0 big lr
        for t in trange:
            if method == "svmd" or method == "coin-svmd":
                eta_grad = svmd_update_v2(target, theta, kernel, n_eigen_threshold=0.98)
            elif method == "svgd" or method == "coin-svgd":
                eta_grad = svgd_update(target, eta, theta, kernel)
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
        return tf.reshape(theta, [-1, theta.shape[-1]]).numpy()

    # get data
    # X, _, _ = HIV_NRTI("3TC")
    X, y, NRTI_muts = HIV_NRTI(drug="3TC", datafile="psi/NRTI_DATA.txt")
    n, p = X.shape
    s = 10
    sigma = 1

    # truth = np.zeros(p)
    # truth[:s] = np.linspace(0.5, 1, s)
    # np.random.shuffle(truth)
    # print(np.nonzero(truth))
    # print(truth[np.nonzero(truth)[0]])
    # truth /= np.sqrt(n)
    # truth *= sigma
    # y = X.dot(truth) + sigma * np.random.standard_normal(n)

    r.assign("n", n)
    r.assign("p", p)
    r.assign("X", X)
    r.assign("y", y)
    r.assign("s", s)
    _ = r("""
    rho = 0.3
    lambda_frac = 1.
    """)

    X.shape, y.shape

    r("sigma_est = 1")
    # theoretical lambda
    r("lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, 'ls', sigma_est)")
    print("lambda:", r["lambda"])

    r("rand_lasso_soln = selectiveInference:::randomizedLasso(X, y, lambda*n, family='gaussian')")
    rand_lasso_soln = r["rand_lasso_soln"]
    active_vars = rand_lasso_soln.rx2["active_set"] - 1
    print("active_vars:", active_vars)
    print(active_vars.shape)

    ci = {}
    methods = ["Coin MSVGD", "MSVGD", "SVMD", "Unadjusted", "Standard"] #,"Coin SVMD"]

    r("fit = rand_lasso_soln$unpen_reg")
    coeff = r["fit"].rx2["coefficients"]
    ci["Unadjusted"] = r("confint(fit, level=0.95)")
    print("coeff:", coeff.shape)
    print("unadjusted_ci:", ci["Unadjusted"].shape)

    r("targets = selectiveInference:::compute_target(rand_lasso_soln, type='selected', sigma_est=sigma_est)")
    r("target_samples = mvrnorm(5000, rep(0,length(rand_lasso_soln$active_set)), targets$targets$cov_target)")

    r("linear = rand_lasso_soln$law$sampling_transform$linear_term")
    r("offset = rand_lasso_soln$law$sampling_transform$offset_term")
    r("theta_init = rand_lasso_soln$law$observed_opt_state")
    r("noise_scale = rand_lasso_soln$noise_scale")

    r("opt_samples = get_opt_samples(rand_lasso_soln, sampler='norejection', nsample=7000, burnin=2000)")
    r("""
    PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                  targets=targets,
                                                  level=0.95,
                                                  opt_samples=opt_samples,
                                                  target_samples=target_samples)
    """)
    ci["Standard"] = r["PVS"].rx2["ci"]

    A = np.asarray(r["linear"])
    print("A:", A.shape)
    b = np.squeeze(np.asarray(r["offset"]), -1)
    print("b:", b.shape)
    theta_init = np.asarray(r["theta_init"])
    print("theta_init:", theta_init.shape)
    nonneg_map = NonnegativeEntropicMap()
    target = SelectiveTarget(nonneg_map, A, b, np.asarray(r["noise_scale"])[0])

    name_map = {
        "SVMD": "svmd",
        "MSVGD": "svgd",
        "Coin SVMD": "coin-svmd",
        "Coin MSVGD": "coin-svgd"
    }

    # run experiment on HIV data
    run_hiv_experiment = False
    if run_hiv_experiment:
        for method in ["Coin MSVGD", "MSVGD", "SVMD"]: #, "Coin SVMD"]:
            opt_samples = run(target, theta_init, method=name_map[method], K=50, n_chain=100)
            nr, nc = opt_samples.shape
            opt_samples_r = r.matrix(opt_samples, nrow=nr, ncol=nc)
            r.assign("opt_samples", opt_samples_r)
            r("""
            PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                          targets=targets,
                                                          level=0.95,
                                                          opt_samples=opt_samples,
                                                          target_samples=target_samples)
            """)
            ci[method] = r["PVS"].rx2["ci"]

        # main plot (plot all mutations)
        main_plot = True
        if main_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
            ax2.axhline(y=0, color='r', linestyle=':')
            fig.subplots_adjust(hspace=0.05)  # adjust space between axes

            for ax in (ax1, ax2):
                for i, method in enumerate(methods):
                    markers, caps, bars = ax.errorbar(np.arange(active_vars.shape[0]) + (i - 2) * 0.15, coeff,
                                                      yerr=np.abs(coeff - ci[method].T), ls='none', lw=4, label=method)
                    for bar in bars:
                        bar.set_alpha(0.7)
                _ = ax.set_xticks(np.arange(active_vars.shape[0]))
                _ = ax.set_xticklabels(NRTI_muts[active_vars])
                ax.tick_params(axis='both', length=0)
            # zoom-in / limit the view to different portions of the data
            ax1.set_ylim(1.85, 2.55)  # outliers only
            ax2.set_ylim(-0.2, 0.5)  # most of the data
            ax1.legend()

            # hide the spines between ax and ax2
            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

            # plt.ylim(ymax=1)
            fig.autofmt_xdate()
            plt.savefig("results/selection_inference_hiv/hiv_ci.pdf", bbox_inches="tight", dpi=300)
            plt.show()


        # sub plots (plot subset of the mutations)
        sub_plots = True
        if sub_plots:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            fig.subplots_adjust(hspace=0.05)  # adjust space between axes

            for i, method in enumerate(methods):
                markers, caps, bars = ax.errorbar(np.arange(3) + (i - 2) * 0.15, coeff[0:3],
                                                  yerr=np.abs(coeff[0:3] - ci[method][0:3].T), ls='none', lw=10, label=method)
                for bar in bars:
                    bar.set_alpha(0.7)
            _ = ax.set_xticks(np.arange(3))
            _ = ax.set_xticklabels(NRTI_muts[active_vars][0:3])
            ax.tick_params(axis='both', length=0)
            ax.set_ylim(-0.2, 0.6)  # most of the data
            ax.legend(loc=2)
            ax.xaxis.tick_bottom()
            ax.axhline(y=0, color='r', linestyle=':')
            plt.savefig("results/selection_inference_hiv/hiv_ci_sub_plot.pdf", bbox_inches="tight", dpi=300)
            plt.show()