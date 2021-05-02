# TransUNet project (UT project - NN)

* Setting up HPC. Create python 3.7 virtual environment [https://docs.hpc.ut.ee/cluster/python_envs/](https://docs.hpc.ut.ee/cluster/python_envs/). Install all requirements  `pip install -r requirements.txt`
* Running HPC with gpu: Use `TransUNet/start_*.sh` scripts. Note that max running time is given in parameter `SBATCH -t 01:00:00` (example 1 hour)
* To execute job in HPC: `sbatch start_*.sh`. To monitor the execution `tail -f slurm-*.out`
* Testing logs (and performance measures DICE, HD95) available in `TransUNet/test_log`.
* Some example predictions available in `predictions`


Currently executed:
* Existing paper main model training with 150 epochs, batch size 24, image patches: 224 `Synapse` `Testing performance in best val model: mean_dice : 0.765989 mean_hd95 : 30.480154`
* Overfit small dataset of university data `University_dev`. `performance in best val model: mean_dice : 0.894027 mean_hd95 : 1.000000`
* Train same model as in existing paper on full `University` dataset (training + val used for training). `Testing performance in best val model: mean_dice : 0.551625 mean_hd95 : 19.641420`


* University preprocessed dataset uploaded to Kaarel's drive. I'll share link in slack


-----
This is the dataset, preprocessed by TransUNet, originated from Synapse multi-organ Dataset.


This preprocessing is implemented and introduced in the work:
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}

The original data can be accessed through "https://www.synapse.org/#!Synapse:syn3193805/wiki/". 
Please refer to the included license in official Synapse websit for information regarding the allowed use of the dataset.

Please note that the preprocessed dataset provided by TransUNet is for research purpose only and please do not redistribute this preprocessed dataset
