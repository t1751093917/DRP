import os
import configparser

print("expr.py")
weights = {'cifar80n_symm20': "Results/cifar100/expr-sample-knn/resnet18-symm0.2-bestAcc_55.3500/epoch_50.pth",
           'cifar100n_symm20': "Results/cifar100/cifar100n_symm20_methods/resnet18-cifar100n_symm20-bestAcc_53.7400/epoch_40.pth",
           'cifar80n_asym40': "Results/cifar100/cifar80n_asym40_methods/resnet18-pairflip_closeset0.4_openset0.2-josrc-20250409_160238-bestAcc_43.2625/epoch_40.pth",
           'cifar100n_asym40': "Results/cifar100/cifar100n_asym40_methods/resnet18-cifar100n_asym40-bestAcc_40.4500/epoch_40.pth",
           'cifar80n_symm80': "Results/cifar100/cifar80n_symm80_methods/resnet18-cifar80n_symm80-bestAcc_29.9750/epoch_50.pth",
           'cifar100n_symm80': "Results/cifar100/cifar100n_symm80_methods/resnet18-cifar100n_symm80-bestAcc_29.5700/epoch_50.pth",
           'ablation': "Results/cifar100/ablation-model-lambda/resnet18-cifar80n_symm20-bestAcc_55.7875/epoch_{}.pth"}
rate_ls = [0.1, 0.15, 0.2, 0.25, 0.3] #,
method_ls = ['Boosting', 'Boosting', 'Boosting'] # ,  #['taylor', 'l2', 'clsa', 'l1', 'fpgm', 'lamp', 'l2_0', 'taylor_0']
lambda_ls = [20, 4, 1, 0.5, 0.05]
# delta_ls = [20, 50, 200, 500, 1000]
delta_ls = [1000, 2500, 5000, 7500, 10000]
k_ls = [5, 10, 20, 50, 80]


GPU = 3
cfg_file = 'config/cifar100_resnet_prune.cfg'
cfg_tmp_file = 'config/cifar100_resnet_prune_tmp.cfg'
project_root = 'boosting'
result_file = f'Results/cifar100/{project_root}/result.txt'
dataset_tag = '80n_a40'
cnt = 0  # start 22

weight_root = weights['cifar80n_asym40']
weight_root = weight_root[:7] + '_NAP' + weight_root[7:]
# weight_path = weight_root.format('50')
# for id in ['10', '30', '50', '70', '90']:
#     weight_path = weight_root.format(id)
for rate in rate_ls:
# for lambda_ in lambda_ls:
# for delta in delta_ls:
    # for k in k_ls:
# for _ in [1, 1, 1, 1, 1]:
    for method in method_ls:
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)
        cfg['network']['weights'] = weight_root
        cfg['network']['epochs'] = str(int(rate*100 *3/4))
        cfg['logging']['project'] = project_root
        cfg['logging']['result_file'] = result_file
        cfg['prune']['prune_method'] = method  # '*Taylor'
        cfg['prune']['prune_rate'] = str(rate)
        cfg['prune']['tag'] = dataset_tag
        # cfg['prune']['theta'] = str(lambda_)
        # cfg['prune']['delta'] = str(delta)
        # cfg['prune']['k'] = str(k)
        with open(cfg_tmp_file, 'w') as f:
            cfg.write(f)

        cnt += 1
        # if cnt < 36:
        #     continue
        cmd = f"python prune_resnet.py --config {cfg_tmp_file} --gpu {GPU}"
        print(f"\n----------------------- {cnt} ----------------------------------")
        print(cmd)
        res = os.system(cmd)
        if res != 0:
            exit(-1)



