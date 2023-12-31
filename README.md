Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection
===========

This is the Python implements of our methods accepted in [WACV 2024](https://wacv2024.thecvf.com/). [[pdf]](https://arxiv.org/abs/2312.04095)


**'scripts/learned.sh'**: Script to train models.  
**'scripts/unlearned.sh'**: Script unlearn models. Please note that the trained (full training dataset) and retrained (with retaining dataset only) models are needed to be trained before the unlearning process in our code. The retrained model is used for evaluation only.


### BibTex
If you find that our works is useful, please cite our paper as follows: 
``` 
@INPROCEEDINGS{GradientProjectionUnlearning,
 author    = {Tuan Hoang and Santu Rana and Sunil Gupta and Svetha Venkatesh},
 title     = {Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection},
 bookTitle = {WACV},
 year      = {2024},
 month     = {Jan},
}
```