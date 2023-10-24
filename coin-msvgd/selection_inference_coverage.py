import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
#os.environ['R_HOME'] = "/usr/lib/R"
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from rpy2 import rinterface
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from tqdm import tqdm

import matplotlib.pyplot as plt

from psi.target import SelectiveTarget
from psi.data import HIV_NRTI
from constrained.nonnegative.entropic import NonnegativeEntropicMap
from constrained.sampling.kernel import imq
from constrained.sampling.svgd import svgd_update
from constrained.sampling.svmd import svmd_update_v2

if __name__=="__main__":

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
        n_iters = 1000
        kernel = imq
        if "coin" in method:
            L = 0
            eta_grad_sum = 0
            reward = 0
            abs_eta_grad_sum = 0
        trange = tqdm(range(n_iters))
        optimizer = tf.keras.optimizers.RMSprop(0.01) # 0.01 for main results, 0.001 small, 0.1 big
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

    #data = "hiv"
    data = None # synthetic data for this experiment

    if data == "hiv":
        X, _, _ = HIV_NRTI(datafile="psi/NRTI_DATA.txt")
        n, p = X.shape
        s = 10
        sigma = 1
        r.assign("n", n)
        r.assign("p", p)
        r.assign("X", X)
        r.assign("s", s)
        r("""
        rho = 0.3
        lambda_frac = 0.7
        """)
    else:
        r("""
        n = 100
        p = 40
        s = 0
        rho = 0.3
        lambda_frac = 0.7
        """)

    nrep = 100
    rng = np.random.RandomState(1)
    seeds = rng.randint(1, 1e6, size=nrep)
    nonneg_map = NonnegativeEntropicMap()

    methods = ["default", "svgd", "svmd", "coin-svgd"] #, "coin-svmd"]

    # empirical coverage vs nominal coverage experiment
    coverage = False
    if coverage:
        run_exp = False
        if run_exp:
            df = pd.DataFrame(columns=["target", "method", "covered", "width"])
            df_rows = []
            methods = ["default", "svgd", "svmd", "coin-svgd"] #, "coin-svmd"]
            # methods = ["default"]
            target_coverages = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
            for i in range(nrep):
                r.assign("seed", seeds[i])
                r("set.seed(seed)")
                np.random.seed(seeds[i])
                tf.random.set_seed(seeds[i])
                print("RUN {}".format(i) + "/" + str(nrep))

                if data == "hiv":
                    truth = np.zeros(p)
                    truth[:s] = np.linspace(0.5, 1, s)
                    np.random.shuffle(truth)
                    truth /= np.sqrt(n)
                    truth *= sigma
                    y = X.dot(truth) + sigma * np.random.standard_normal(n)
                    r.assign("y", y)
                    r.assign("beta", truth)
                else:
                    r("data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=sqrt(2*log(p)/n), design='equicorrelated', scale=TRUE)")

                    r("X = data$X")
                    r("y = data$y")
                    r("beta = data$beta")
                    r("cat('true nonzero:', which(beta != 0), '\n')")

                r("sigma_est = 1")
                # theoretical lambda
                r("lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, 'ls', sigma_est)")

                r("rand_lasso_soln = selectiveInference:::randomizedLasso(X, y, lambda*n, family='gaussian')")
                rand_lasso_soln = r["rand_lasso_soln"]
                active_vars = rand_lasso_soln.rx2["active_set"]
                if active_vars is rinterface.NULL:
                    continue
                print("active_vars:", active_vars)

                r("targets = selectiveInference:::compute_target(rand_lasso_soln, type='selected', sigma_est=sigma_est)")
                r("target_samples = mvrnorm(5000, rep(0,length(rand_lasso_soln$active_set)), targets$targets$cov_target)")

                r("linear = rand_lasso_soln$law$sampling_transform$linear_term")
                r("offset = rand_lasso_soln$law$sampling_transform$offset_term")
                r("theta_init = rand_lasso_soln$law$observed_opt_state")
                r("noise_scale = rand_lasso_soln$noise_scale")

                A = np.asarray(r["linear"])
                b = np.squeeze(np.asarray(r["offset"]), -1)
                theta_init = np.asarray(r["theta_init"])
                target = SelectiveTarget(nonneg_map, A, b, np.asarray(r["noise_scale"])[0])

                for method in methods:
                    print("sampler: {}".format(method))
                    if method == "default":
                        r("opt_samples = get_opt_samples(rand_lasso_soln, sampler='norejection', nsample=7000, burnin=2000)")
                    else:
                        timer = -time.time()
                        opt_samples = run(target, theta_init, method=method, K=50, n_chain=100)
                        print("time:", timer + time.time())
                        nr, nc = opt_samples.shape
                        opt_samples_r = r.matrix(opt_samples, nrow=nr, ncol=nc)
                        r.assign("opt_samples", opt_samples_r)

                    for target_coverage in target_coverages:
                        r.assign("target_coverage", target_coverage)
                        r("""
                        PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                                      targets=targets,
                                                                      level=target_coverage,
                                                                      opt_samples=opt_samples,
                                                                      target_samples=target_samples)
                        """)
                        r("sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[rand_lasso_soln$active_set])")
                        r("sel_length = as.vector(PVS$ci[,2] - PVS$ci[,1])")
                        for covered, width in zip(r["sel_coverage"], r["sel_length"]):
                            df_rows.append({"target": target_coverage,
                                            "method": method,
                                            "covered": covered,
                                            "width": width})

            df = pd.DataFrame(df_rows)

            # save
            save = False
            if save:
                if data == "hiv":
                    df.to_csv('results/hiv_coverage.csv')
                else:
                    df.to_csv("results/coverage.csv")

        # load
        load = True
        if load:
            df = pd.read_csv("results/selection_inference/coverage.csv")
        df.loc[df["method"] == "svgd", "method"] = "MSVGD"
        df.loc[df["method"] == "coin-svgd", "method"] = "Coin MSVGD"
        df.loc[df["method"] == "svmd", "method"] = "SVMD"
        df.loc[df["method"] == "coin-svmd", "method"] = "Coin SVMD"
        df.loc[df["method"] == "default", "method"] = "Standard"
        df.loc[df["method"] == "proj_svgd", "method"] = "Projected SVGD"

        target_coverages = np.array(sorted(set(df["target"].values)))
        methods = ["Coin MSVGD", "Coin SVMD", "MSVGD", "SVMD", "Standard"]

        def ci(df, method, z=1.96):
            means = []
            errs = []
            for target in target_coverages:
                covered = df.loc[(df["target"] == target) & (df["method"] == method), "covered"].values
                ns = covered.sum()
                n = covered.shape[0]
                means.append((ns + 0.5 * z**2) / (n + z**2))
                errs.append(z / (n + z**2) * np.sqrt(ns * (n - ns) / n + z**2 / 4.))
            return means, errs

        # plot
        plt.figure(figsize=(8, 6))
        lines = ["-", "-.", "--"]
        for i, method in enumerate(methods):
            if method != "Coin SVMD":
                means, errors = ci(df, method)
                # plt.errorbar(target_coverages + (i - 1) * 0.002, means, yerr=errors, label=method)
                plt.errorbar(target_coverages, means, yerr=errors, label=method, marker=".", ms=14,
                             capsize=4, capthick=1, zorder=5-i)
                plt.plot(target_coverages, target_coverages, ":", color="black")
        plt.xticks(np.linspace(0.8, 1, 5))
        plt.yticks(np.linspace(0.5, 1, 11))
        plt.grid(True)
        plt.xlabel("Nominal coverage", fontsize=20)
        plt.ylabel("Actual coverage", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 20})
        plt.ylim(ymin=0.75, ymax=1)
        plt.savefig("results/selection_inference/coverage.pdf", bbox_inches="tight", dpi=150)
        plt.show()

    # coverage vs number of samples experiment
    coverage_wrt_k = False
    methods = ["coin-svgd", "svgd", "svmd", "default"]
    if coverage_wrt_k:
        run_exp = False
        if run_exp:
            df = pd.DataFrame(columns=["target", "method", "n_samples", "covered", "width"])
            df_rows = []
            methods = ["coin-svgd", "svgd", "svmd", "default"] #"coin-svmd",
            target_coverage = 0.9
            r.assign("target_coverage", target_coverage)
            K = 50
            for i in range(nrep):
                r.assign("seed", seeds[i])
                r("set.seed(seed)")
                np.random.seed(seeds[i])
                tf.random.set_seed(seeds[i])
                print("RUN {}".format(i) + "/" + str(nrep))

                if data == "hiv":
                    truth = np.zeros(p)
                    truth[:s] = np.linspace(0.5, 1, s)
                    np.random.shuffle(truth)
                    truth /= np.sqrt(n)
                    truth *= sigma
                    y = X.dot(truth) + sigma * np.random.standard_normal(n)
                    r.assign("y", y)
                    r.assign("beta", truth)
                else:
                    r("data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=sqrt(2*log(p)/n), design='equicorrelated', scale=TRUE)")
                    r("X = data$X")
                    r("y = data$y")
                    r("beta = data$beta")

                    r("cat('true nonzero:', which(beta != 0), '\n')")

                r("sigma_est = 1")
                # theoretical lambda
                r("lambda = lambda_frac*selectiveInference:::theoretical.lambda(X, 'ls', sigma_est)")

                r("rand_lasso_soln = selectiveInference:::randomizedLasso(X, y, lambda*n, family='gaussian')")
                rand_lasso_soln = r["rand_lasso_soln"]
                active_vars = rand_lasso_soln.rx2["active_set"]
                if active_vars is rinterface.NULL:
                    continue
                print("active_vars:", active_vars)

                r("targets = selectiveInference:::compute_target(rand_lasso_soln, type='selected', sigma_est=sigma_est)")
                r("target_samples = mvrnorm(4000, rep(0,length(rand_lasso_soln$active_set)), targets$targets$cov_target)")

                r("linear = rand_lasso_soln$law$sampling_transform$linear_term")
                r("offset = rand_lasso_soln$law$sampling_transform$offset_term")
                r("theta_init = rand_lasso_soln$law$observed_opt_state")
                r("noise_scale = rand_lasso_soln$noise_scale")

                A = np.asarray(r["linear"])
                print("A:", A.shape)
                b = np.squeeze(np.asarray(r["offset"]), -1)
                print("b:", b.shape)
                theta_init = np.asarray(r["theta_init"])
                print("theta_init:", theta_init.shape)
                target = SelectiveTarget(nonneg_map, A, b, np.asarray(r["noise_scale"])[0])

                for method in methods:
                    print("sampler: {}".format(method))
                    if method == "default":
                        r("opt_samples = get_opt_samples(rand_lasso_soln, sampler='norejection', nsample=6000, burnin=2000)")
                    else:
                        timer = -time.time()
                        opt_samples = run(target, theta_init, method=method, K=K, n_chain=80)
                        print("time:", timer + time.time())
                        nr, nc = opt_samples.shape
                        opt_samples_r = r.matrix(opt_samples, nrow=nr, ncol=nc)
                        r.assign("opt_samples", opt_samples_r)

                    for n_chains in [10, 20, 30, 40, 50, 60]:
                        n_samples = K * n_chains
                        r.assign("n_samples", n_samples)
                        r("subset_target_samples = target_samples[1:n_samples,,drop=FALSE]")
                        r("subset_opt_samples = opt_samples[1:n_samples,,drop=FALSE]")
                        r("""
                        PVS = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                                      targets=targets,
                                                                      level=target_coverage,
                                                                      opt_samples=subset_opt_samples,
                                                                      target_samples=subset_target_samples)
                        """)
                        r("sel_coverage = selectiveInference:::compute_coverage(PVS$ci, beta[rand_lasso_soln$active_set])")
                        r("sel_length = as.vector(PVS$ci[,2] - PVS$ci[,1])")
                        for covered, width in zip(r["sel_coverage"], r["sel_length"]):
                            df_rows.append({"target": target_coverage,
                                            "method": method,
                                            "n_samples": n_samples,
                                            "covered": covered,
                                            "width": width})
            # save
            df = pd.DataFrame(df_rows)
            save = False
            if save:
                if data == "hiv":
                    df.to_csv('results/selection_inference/hiv_coverage_wrt_k.csv')
                else:
                    df.to_csv("results/selection_inference/coverage_wrt_k.csv")

        # load
        load = True
        if load:
            df = pd.read_csv("results/selection_inference/coverage_wrt_k.csv")
            # df = pd.read_csv("hiv_coverage_wrt_k.csv")

        name_map = {
            "svgd": "MSVGD",
            "coin-svgd": "Coin MSVGD",
            "svmd": "SVMD",
            "coin-svmd": "Coin SVMD",
            "default": "Standard",
            "proj_svgd": "Projected SVGD"
        }

        k_list = [1000, 1500, 2000, 2500, 3000]

        def ci_wrt_k(df, method, z=1.96):
            means = []
            errs = []
            for n_samples in k_list:
                covered = df.loc[(df["n_samples"] == n_samples) & (df["method"] == method), "covered"].values
                ns = covered.sum()
                n = covered.shape[0]
                means.append((ns + 0.5 * z ** 2) / (n + z ** 2))
                errs.append(z / (n + z ** 2) * np.sqrt(ns * (n - ns) / n + z ** 2 / 4.))
            return means, errs

        plt.figure(figsize=(8, 6))
        for i, method in enumerate(methods):
            if method != "coin-svmd":
                means, errors = ci_wrt_k(df, method)
                plt.errorbar(k_list, means, yerr=errors, label=name_map[method], marker=".", ms=14,
                             capsize=4, capthick=1, zorder=5-i)
        plt.plot(k_list, [0.9] * len(k_list), ":", color="black")
        plt.xticks(k_list)
        # plt.yticks(np.linspace(0.8, 1, 5))
        plt.grid(True)
        plt.xlabel("Number of Samples (N)", fontsize=20)
        plt.ylabel("Actual Coverage", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 20}, loc='lower left')
        plt.ylim(bottom=0.75, top=1.00)
        plt.savefig("results/selection_inference/coverage_wrt_k.pdf", bbox_inches="tight", dpi=300)
        plt.show()