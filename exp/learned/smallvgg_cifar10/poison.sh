

for num_poison in 500 1000 1500 2000 2500; do
    log_dir=exp/learned/smallvgg_cifar10/poison_${num_poison}/full
    python3 train.py exp/learned/smallvgg_cifar10/config_poison.yaml \
            --paths.log_name $log_dir \
            --dataset.train.params.num_poison $num_poison \
            --dataset.train.params.data_section full

    log_dir=exp/learned/smallvgg_cifar10/poison_${num_poison}/clean
    python3 train.py exp/learned/smallvgg_cifar10/config_poison.yaml \
            --paths.log_name $log_dir \
            --dataset.train.params.num_poison $num_poison \
            --dataset.train.params.data_section clean

done