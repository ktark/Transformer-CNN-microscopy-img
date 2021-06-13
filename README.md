# TransUNet project (UT project - NN)

* python 3.7 virtual environment [https://docs.hpc.ut.ee/cluster/python_envs/](https://docs.hpc.ut.ee/cluster/python_envs/). Install all requirements  `pip install -r requirements.txt`
* Running HPC with gpu: Use `TransUNet/start_*.sh` scripts. 
* To execute job in HPC: `sbatch start_*.sh`. To monitor the execution `tail -f slurm-*.out`
* Testing logs (and performance measures DICE/pixel-wise F1, HD95) available in `TransUNet/test_log`.

-----
Branches:
* master - Initial model with cropping 
* additional_CNN - additional CNN or ResNet to bottleneck changes
* resnet_skip - additional skip-connection from ResNet hidden features to bottleneck
* transformer-to-skip - moving transformers from encoder to skip-connection.

-----
Initial architecture source code repo: https://github.com/Beckschen/TransUNet
