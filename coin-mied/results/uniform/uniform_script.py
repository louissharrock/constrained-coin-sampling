# script to run MIED and MSVGD over a range of learning rates

import numpy as np
import itertools
import os
import torch
import argparse
from pathlib import Path
import math
import h5py
import wandb
import matplotlib.pyplot as plt
from csb.validators.particle import ParticleValidator
from csb.validators.metrics import compute_metric
from csb.utils.random import seed_all
from csb.utils.ec import ExperimentCoordinator
from csb.utils.h5_helpers import save_dict_h5
from csb.problems.analytical_problems import create_problem
from csb.solvers.mied import MIED, CoinMIED
from csb.solvers.svgd import SVGD, CoinSVGD
from csb.solvers.ksdd import KSDD, CoinKSDD
from csb.solvers.ipd import IPD, CoinIPD
from csb.solvers.dynamic_barrier import DynamicBarrier
from csb.solvers.no_op_projector import NoOpProjector

root_dir = Path("__file__").resolve().parent

lr_grid = np.logspace(-5, 0, 20)
n_lr_grid = lr_grid.shape[0]

seeds = [0, 1, 2, 3, 4]
n_seeds = len(seeds)

# run all methods over grid of learning rates
# MIED methods
methods = ['MIED', 'coin-mied']
kernels = ['riesz', 'laplace', 'gaussian']
methods_and_kernels = [methods, kernels]
permutations = list(itertools.product(*methods_and_kernels))
n_permutations = len(permutations)
run_mied_grid_search = False
if run_mied_grid_search:

    sinkhorn_array = np.zeros((n_permutations, n_lr_grid, n_seeds))
    energy_array = np.zeros((n_permutations, n_lr_grid, n_seeds))

    for ii, method_kernel in enumerate(permutations):

        for jj, lr in enumerate(lr_grid):

            for kk, seed in enumerate(seeds):

                ec = ExperimentCoordinator(root_dir)
                ec.add_temporary_arguments({
                    'num_itr': 250,
                    'traj_freq': 1,
                    'val_freq': 250,
                    'plot_update': False,
                    'num_trial': 1,
                    'gt_multiplier': 10,
                    'device': 'cpu',
                })

                ec.add_common_arguments({
                    'optimizer': 'RMSprop',
                    'lr': lr,
                    'prob': 'uniform_box_2d',
                    'reparam': 'box_tanh',
                    'filter_range': -1,
                    'wandb': False,
                    'num_particle': 100,
                    'alpha': 0,
                    'L_init': 0,
                    'seed': seed,
                })

                ec.add_projector_arguments(DynamicBarrier, {
                    'alpha_db': 1.0,
                    'merge_eq': True,
                    'max_proj_itr': 20
                })

                method = method_kernel[0]
                kernel = method_kernel[1]

                if method == "MIED" and kernel == "riesz":
                    ec.add_common_arguments({
                        'method': 'MIED',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(MIED, {
                        'kernel': 'riesz',
                        'eps': 1e-8,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })
                elif method == "coin-mied" and kernel == "riesz":
                    ec.add_common_arguments({
                        'method': 'coin-mied',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(CoinMIED, {
                        'kernel': 'riesz',
                        'eps': 1e-8,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })
                elif method == "MIED" and kernel == 'laplace':
                    ec.add_common_arguments({
                        'method': 'MIED',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(MIED, {
                        'kernel': 'laplace',
                        'eps': 1e-2,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })
                elif method == "coin-mied" and kernel == 'laplace':
                    ec.add_common_arguments({
                        'method': 'coin-mied',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(CoinMIED, {
                        'kernel': 'laplace',
                        'eps': 1e-2,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })
                elif method == "MIED" and kernel == 'gaussian':
                    ec.add_common_arguments({
                        'method': 'MIED',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(MIED, {
                        'kernel': 'gaussian',
                        'eps': 1e-3,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })
                elif method == "coin-mied" and kernel == 'gaussian':
                    ec.add_common_arguments({
                        'method': 'coin-mied',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(CoinMIED, {
                        'kernel': 'gaussian',
                        'eps': 1e-3,
                        'riesz_s': -1.0,
                        'alpha_mied': 0.5,
                        'include_diag': 'nnd_scale',
                        'diag_mul': 1.3,
                    })

                # for coin, don't need to repeat for each learning rate
                if (method == "coin-mied" and jj == 0) or (method != "coin-mied" and jj >=0):
                    run_exp = True
                else:
                    run_exp = False
                if run_exp:
                    ec_result = ec.parse_args()

                    tmp_args, config = ec_result.tmp_args, ec_result.config

                    seed_all(config['seed'])

                    problem = create_problem(ec_result.tmp_args.device,
                                             config['prob'],
                                             config['reparam'])

                    solver = ec.create_solver(problem)
                    validator = ParticleValidator(problem=problem)

                    def post_step_fn(i):
                        if tmp_args.traj_freq <= 0:
                            return
                        if i == 0 or (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
                            metrics = ['sinkhorn', 'energy_dist']

                            result = validator.run(samples=solver.get_samples(),
                                                   updates=solver.compute_update(
                                                       i, solver.get_samples()),
                                                   include_density=False,
                                                   metrics=metrics,
                                                   num_trial=tmp_args.num_trial,
                                                   gt_multipler=tmp_args.gt_multiplier,
                                                   filter_range=config['filter_range'],
                                                   save_path=None)
                            log_dict = {
                                'metrics': {m: result[m] for m in metrics},
                            }
                            if tmp_args.num_trial > 1:
                                log_dict['metrics_std'] = {m: result[m + '_std']
                                                           for m in metrics}

                    solver.run(num_itr=tmp_args.num_itr,
                               post_step_fn=post_step_fn)

                    print('Validating ...')
                    final = validator.run(samples=solver.get_samples(),
                                          include_gt=True,
                                          include_density=problem.in_dim == 2,
                                          density_bbox=problem.bbox,
                                          save_path=None,
                                          metrics=['sinkhorn', 'energy_dist']
                                          )

                    if method == "coin-mied" and jj == 0:
                        for ll in range(n_lr_grid):
                            sinkhorn_array[ii, ll, kk], energy_array[ii, ll, kk] = final['sinkhorn'], final['energy_dist']
                            print("Method: ", method)
                            print("Energy Distance: ", final['energy_dist'])
                    else:
                        sinkhorn_array[ii, jj, kk], energy_array[ii, jj, kk] = final['sinkhorn'], final['energy_dist']
                        print("Method: ", method)
                        print("LR: ", lr)
                        print("Energy Distance: ", final['energy_dist'])

    save_results = True
    if save_results:
        results_dir = "results/uniform"
        if not os.path.exists(results_dir):
           os.makedirs(results_dir)

        np.save(results_dir + "/" + "energy_dist", energy_array)

    mied_riesz_lr = lr_grid[np.argmin(np.mean(energy_array, axis=2)[0, :])]
    mied_laplace_lr = lr_grid[np.argmin(np.mean(energy_array, axis=2)[1, :])]
    mied_gaussian_lr = lr_grid[np.argmin(np.mean(energy_array, axis=2)[2, :])]
    best_lr = np.array([mied_riesz_lr, mied_laplace_lr, mied_gaussian_lr])
    save_lr = True
    if save_lr:
        np.save("results/uniform/best_lr", best_lr)

# Run grid search over learning rates for MSVGD
msvgd_methods = ["SVGD", "CoinSVGD"]
energy_array_svgd = np.zeros((2, n_lr_grid, n_seeds))
run_msvgd_grid_search = False
if run_msvgd_grid_search:

    for ii, method in enumerate(msvgd_methods):

        for jj, lr in enumerate(lr_grid):

            for kk, seed in enumerate(seeds):

                ec = ExperimentCoordinator(root_dir)
                ec.add_temporary_arguments({
                    'num_itr': 250,
                    'traj_freq': 1,
                    'val_freq': 250,
                    'plot_update': False,
                    'num_trial': 1,
                    'gt_multiplier': 10,
                    'device': 'cpu',
                })

                ec.add_common_arguments({
                    'optimizer': 'RMSprop',
                    'lr': lr,
                    'prob': 'uniform_box_2d',
                    'reparam': 'box_mirror_entropic',
                    'filter_range': -1,
                    'wandb': False,
                    'num_particle': 100,
                    'alpha': 0,
                    'L_init': 0,
                    'seed': seed,
                })

                ec.add_projector_arguments(DynamicBarrier, {
                    'alpha_db': 1.0,
                    'merge_eq': True,
                    'max_proj_itr': 20
                })

                if method == "SVGD":
                    ec.add_common_arguments({
                        'method': 'SVGD',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(SVGD, {
                        'gaussian_bw': 1e-2,
                    })
                elif method == "CoinSVGD":
                    ec.add_common_arguments({
                        'method': 'CoinSVGD',
                        'projector': 'DB',
                    })
                    ec.add_method_arguments(CoinSVGD, {
                        'gaussian_bw': 1e-2,
                    })

                # for coin, don't need to repeat for each learning rate
                if (method == "CoinSVGD" and jj == 0) or (method != "CoinSVGD" and jj >=0):
                    run_exp = True
                else:
                    run_exp = False
                if run_exp:
                    ec_result = ec.parse_args()

                    tmp_args, config = ec_result.tmp_args, ec_result.config

                    seed_all(config['seed'])

                    problem = create_problem(ec_result.tmp_args.device,
                                             config['prob'],
                                             config['reparam'])

                    solver = ec.create_solver(problem)
                    validator = ParticleValidator(problem=problem)

                    def post_step_fn(i):
                        if tmp_args.traj_freq <= 0:
                            return
                        if i == 0 or (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
                            metrics = ['sinkhorn', 'energy_dist']

                            result = validator.run(samples=solver.get_samples(),
                                                   updates=solver.compute_update(
                                                       i, solver.get_samples()),
                                                   include_density=False,
                                                   metrics=metrics,
                                                   num_trial=tmp_args.num_trial,
                                                   gt_multipler=tmp_args.gt_multiplier,
                                                   filter_range=config['filter_range'],
                                                   save_path=None)
                            log_dict = {
                                'metrics': {m: result[m] for m in metrics},
                            }
                            if tmp_args.num_trial > 1:
                                log_dict['metrics_std'] = {m: result[m + '_std']
                                                           for m in metrics}

                    solver.run(num_itr=tmp_args.num_itr,
                               post_step_fn=post_step_fn)

                    print('Validating ...')
                    final = validator.run(samples=solver.get_samples(),
                                          include_gt=True,
                                          include_density=problem.in_dim == 2,
                                          density_bbox=problem.bbox,
                                          save_path=None,
                                          metrics=['sinkhorn', 'energy_dist']
                                          )

                    if method == "CoinSVGD" and jj == 0:
                        for ll in range(n_lr_grid):
                            energy_array_svgd[ii, ll, kk] = final['energy_dist']
                            print("Method: ", method)
                            print("Energy Distance: ", final['energy_dist'])
                    else:
                        energy_array_svgd[ii, jj, kk] = final['energy_dist']
                        print("Method: ", method)
                        print("LR: ", lr)
                        print("Energy Distance: ", final['energy_dist'])

    save_results = True
    if save_results:
        results_dir = "results/uniform"
        if not os.path.exists(results_dir):
           os.makedirs(results_dir)

        np.save(results_dir + "/" + "energy_dist_svgd", energy_array_svgd)

    best_lr_svgd = lr_grid[np.argmin(np.mean(energy_array_svgd, axis=2)[0, :])]
    save_lr = True
    if save_lr:
        np.save("results/uniform/best_lr_svgd", best_lr_svgd)


# now fix LR and generate a full set of results for the best learning rates
plot_best_vs_iter = False
if plot_best_vs_iter:
    load_lr = True
    if load_lr:
        best_lr = np.load("results/uniform/best_lr.npy")
    # now fix LR and plot energy distance vs iterations
    mied_riesz_lr = best_lr[0]
    mied_laplace_lr = best_lr[1]
    mied_gaussian_lr = best_lr[2]

    for ii, method_kernel in enumerate(permutations):

        ec = ExperimentCoordinator(root_dir)
        ec.add_temporary_arguments({
            'num_itr': 250,
            'traj_freq': 1,
            'val_freq': 1,
            'plot_update': False,
            'num_trial': 10, # 10 trials
            'gt_multiplier': 10,
            'device': 'cpu',
        })

        ec.add_common_arguments({
            'optimizer': 'RMSprop',
            'prob': 'uniform_box_2d',
            'reparam': 'box_tanh',
            'filter_range': -1,
            'wandb': True,
            'num_particle': 100,
            'alpha': 0,
            'L_init': 0,
            'seed': 42,
        })

        ec.add_projector_arguments(DynamicBarrier, {
            'alpha_db': 1.0,
            'merge_eq': True,
            'max_proj_itr': 20
        })

        method = method_kernel[0]
        kernel = method_kernel[1]

        if method == "MIED" and kernel == "riesz":
            ec.add_common_arguments({
                'lr': mied_riesz_lr
            })
            ec.add_common_arguments({
                'method': 'MIED',
                'projector': 'DB',
            })
            ec.add_method_arguments(MIED, {
                'kernel': 'riesz',
                'eps': 1e-8,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })
        elif method == "coin-mied" and kernel == "riesz":
            ec.add_common_arguments({
                'lr': mied_riesz_lr
            })
            ec.add_common_arguments({
                'method': 'coin-mied',
                'projector': 'DB',
            })
            ec.add_method_arguments(CoinMIED, {
                'kernel': 'riesz',
                'eps': 1e-8,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })
        elif method == "MIED" and kernel == 'laplace':
            ec.add_common_arguments({
                'lr': mied_laplace_lr
            })
            ec.add_common_arguments({
                'method': 'MIED',
                'projector': 'DB',
            })
            ec.add_method_arguments(MIED, {
                'kernel': 'laplace',
                'eps': 1e-2,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })
        elif method == "coin-mied" and kernel == 'laplace':
            ec.add_common_arguments({
                'lr': mied_laplace_lr
            })
            ec.add_common_arguments({
                'method': 'coin-mied',
                'projector': 'DB',
            })
            ec.add_method_arguments(CoinMIED, {
                'kernel': 'laplace',
                'eps': 1e-2,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })
        elif method == "MIED" and kernel == 'gaussian':
            ec.add_common_arguments({
                'lr': mied_gaussian_lr
            })
            ec.add_common_arguments({
                'method': 'MIED',
                'projector': 'DB',
            })
            ec.add_method_arguments(MIED, {
                'kernel': 'gaussian',
                'eps': 1e-3,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })
        elif method == "coin-mied" and kernel == 'gaussian':
            ec.add_common_arguments({
                'lr': mied_gaussian_lr
            })
            ec.add_common_arguments({
                'method': 'coin-mied',
                'projector': 'DB',
            })
            ec.add_method_arguments(CoinMIED, {
                'kernel': 'gaussian',
                'eps': 1e-3,
                'riesz_s': -1.0,
                'alpha_mied': 0.5,
                'include_diag': 'nnd_scale',
                'diag_mul': 1.3,
            })

        ec_result = ec.parse_args()

        tmp_args, config = ec_result.tmp_args, ec_result.config

        seed_all(config['seed'])

        problem = create_problem(ec_result.tmp_args.device,
                                 config['prob'],
                                 config['reparam'])

        solver = ec.create_solver(problem)
        validator = ParticleValidator(problem=problem)

        gt_samples = problem.sample_gt(1000, refresh=False)
        init_energy = compute_metric(solver.get_samples(), problem, metric='energy_dist', gt_samples = gt_samples,
                                     refresh=False, gt_multiplier=1)
        print(init_energy)

        def post_step_fn(i):
            if tmp_args.traj_freq <= 0:
                return
            if i == 0 or (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
                metrics = ['sinkhorn', 'energy_dist']

                result = validator.run(samples=solver.get_samples(),
                                       include_density=False,
                                       metrics=metrics,
                                       num_trial=tmp_args.num_trial,
                                       gt_multipler=tmp_args.gt_multiplier,
                                       filter_range=config['filter_range'],
                                       save_path=None)
                log_dict = {
                    'metrics': {m: result[m] for m in metrics},
                    # 'samples': wandb.Image(fig),
                }
                if i == 0:
                    print(log_dict)
                if tmp_args.num_trial > 1:
                    log_dict['metrics_std'] = {m: result[m + '_std']
                                               for m in metrics}

                wandb.log(log_dict, commit=True)

        solver.run(num_itr=tmp_args.num_itr,
                   post_step_fn=post_step_fn)

        print('Validating ...')
        final = validator.run(samples=solver.get_samples(),
                              include_gt=True,
                              include_density=problem.in_dim == 2,
                              density_bbox=problem.bbox,
                              save_path=None,
                              metrics=['sinkhorn', 'energy_dist']
                              )



# now fix LR and plot energy distance vs iterations, now MSVGD methods
plot_best_vs_iter_svgd = False
if plot_best_vs_iter_svgd:
    load_lr = True
    if load_lr:
        best_lr_svgd = np.load("results/uniform/best_lr_svgd.npy")

    # now fix LR and plot energy distance vs iterations
    for ii, method in enumerate(['CoinSVGD']):

        ec = ExperimentCoordinator(root_dir)
        ec.add_temporary_arguments({
            'num_itr': 250,
            'traj_freq': 1,
            'val_freq': 1,
            'plot_update': False,
            'num_trial': 10, # 10 trials
            'gt_multiplier': 10,
            'device': 'cpu',
        })

        ec.add_common_arguments({
            'optimizer': 'RMSprop',
            'prob': 'uniform_box_2d',
            'reparam': 'box_mirror_entropic',
            'filter_range': -1,
            'wandb': True,
            'num_particle': 100,
            'alpha': 0,
            'L_init': 0,
            'seed': 42,
        })

        ec.add_projector_arguments(DynamicBarrier, {
            'alpha_db': 1.0,
            'merge_eq': True,
            'max_proj_itr': 20
        })

        if method == "SVGD":
            ec.add_common_arguments({
                'lr': best_lr_svgd
            })
            ec.add_common_arguments({
                'method': 'SVGD',
                'projector': 'DB',
            })
            ec.add_method_arguments(SVGD, {
                'gaussian_bw': 1e-2
            })
        elif method == "CoinSVGD":
            ec.add_common_arguments({
                'lr': best_lr_svgd
            })
            ec.add_common_arguments({
                'method': 'CoinSVGD',
                'projector': 'DB',
            })
            ec.add_method_arguments(CoinSVGD, {
                'gaussian_bw': 1e-2
            })

        ec_result = ec.parse_args()

        tmp_args, config = ec_result.tmp_args, ec_result.config

        seed_all(config['seed'])

        problem = create_problem(ec_result.tmp_args.device,
                                 config['prob'],
                                 config['reparam'])

        solver = ec.create_solver(problem)
        validator = ParticleValidator(problem=problem)

        gt_samples = problem.sample_gt(1000, refresh=False)
        init_energy = compute_metric(solver.get_samples(), problem, metric='energy_dist', gt_samples = gt_samples,
                                     refresh=False, gt_multiplier=1)

        def post_step_fn(i):
            if tmp_args.traj_freq <= 0:
                return
            if i == 0 or (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
                metrics = ['sinkhorn', 'energy_dist']

                result = validator.run(samples=solver.get_samples(),
                                       include_density=False,
                                       metrics=metrics,
                                       num_trial=tmp_args.num_trial,
                                       gt_multipler=tmp_args.gt_multiplier,
                                       filter_range=config['filter_range'],
                                       save_path=None)
                log_dict = {
                    'metrics': {m: result[m] for m in metrics},
                    # 'samples': wandb.Image(fig),
                }
                if i == 0:
                    print(log_dict)
                if tmp_args.num_trial > 1:
                    log_dict['metrics_std'] = {m: result[m + '_std']
                                               for m in metrics}

                wandb.log(log_dict, commit=True)

        solver.run(num_itr=tmp_args.num_itr,
                   post_step_fn=post_step_fn)

        print('Validating ...')
        final = validator.run(samples=solver.get_samples(),
                              include_gt=True,
                              include_density=problem.in_dim == 2,
                              density_bbox=problem.bbox,
                              save_path=None,
                              metrics=['sinkhorn', 'energy_dist']
                              )
