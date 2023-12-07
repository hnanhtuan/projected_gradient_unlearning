
python3 unlearn.py  unlearn_submit/exp/unlearned/smallvgg_cifar10/data_removal/exp1/config.yaml


python3 depoisoning.py unlearn_submit/exp/unlearned/smallvgg_cifar10/depoison/500/config.yaml
python3 depoisoning.py unlearn_submit/exp/unlearned/smallvgg_cifar10/depoison/1000/config.yaml
python3 depoisoning.py unlearn_submit/exp/unlearned/smallvgg_cifar10/depoison/1500/config.yaml
python3 depoisoning.py unlearn_submit/exp/unlearned/smallvgg_cifar10/depoison/2000/config.yaml
python3 depoisoning.py unlearn_submit/exp/unlearned/smallvgg_cifar10/depoison/2500/config.yaml


python3 unlearn_class.py unlearn_submit/exp/unlearned/resnet18_tinyimagenet/forget_5classes/exp1/config.yaml
python3 unlearn_class.py unlearn_submit/exp/unlearned/resnet18_tinyimagenet/forget_10classes/exp1/config.yaml
python3 unlearn_class.py unlearn_submit/exp/unlearned/resnet18_tinyimagenet/forget_15classes/exp1/config.yaml