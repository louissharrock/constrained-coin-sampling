# Coin MIED
python results/fairness_bnn/fairness_bnn_run.py --thres=0.01 --method=CoinMIED --num_particle=50 --projector=DB --reparam=id --num_itr=2000 --val_freq=10 --save_freq=-1 --wandb=true --exp_name='coinMIED_t_0_01'

# MIED
python results/fairness_bnn/fairness_bnn_run.py --thres=0.01 --method=MIED --num_particle=50 --projector=DB --lr=0.001 --reparam=id --num_itr=2000 --val_freq=10 --save_freq=-1 --wandb=true --exp_name='MIED_t_0_01'