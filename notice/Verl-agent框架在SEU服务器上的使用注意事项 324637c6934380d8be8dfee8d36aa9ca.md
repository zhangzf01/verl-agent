# Verl-agent框架在SEU服务器上的使用注意事项

1.  因为服务器断网， Wandb需要离线设置，学习如何设置，并学习如何利用wandb查看训练曲线，分析效果。学习如何从离线wandb文件上传到个人wandb账号
[https://blog.csdn.net/weixin_42336668/article/details/147918322](https://blog.csdn.net/weixin_42336668/article/details/147918322)

1. Run shell脚本需要设置一些路径：
trainer.default_local_dir=''\
actor_rollout_ref.model.path=\
等，需要放到机械硬盘/seu_share2/（机械硬盘）/seu_share/是默认目录固态硬盘

1. Free -h 查看服务器内存大小，每个verl-agent程序需要消耗多少内存（学习如何计算），不然大家的程序跑上去爆内存的话，服务器会挂掉。

1. **H100服务器不要在vscode作业在跑代码！学习如何使用sbatch提交作业，squeue查看作业, （在用户手册查看详细操作）不然会导致其他用户无法正常使用显卡。短期调试可用vscode,不要长期在vscode跑，影响其他用户作业分配显卡资源**

1. Verl-agent可以在seu服务器上debug,查看如何使用https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html

1. huggingface模型（qwen2.5）可以放在公共模型/seu_nvme/ogai/model 下（然后在相应文件更改路径），不用每个用户分别下载。。。现在好像公共目录有权限问题，先忽略

1. h005节点是全院公共节点，也可以去排队跑

1. Checkpoints断点保存需要注意更改到机械硬盘trainer.default_local_dir=''\，不然服务器空间不足，另外保存的频率trainer.save_freq=40左右，并且跑完到160epoch之后，需要自己去清理一下不用的前面epoch的checkpoints，不然跑多了，服务器空间不足。。。

1. 建议先用2卡再1.5b小模型上把idea做出效果，然后再申请使用4卡跑更大的7b模型

1. debug调试的时候可减少train_datasize, max_step等参数，快一点跑通就行

1. 有些baseline对比算法可能不用跑，需要找其他人问问有没有训练保存过的checkpoints，节省计算资源

1. 作业排队时候，修改文件（如参数）后，排队的作业的修改之处是会变的！只有作业运行之后，修改文件不影响正在运行的作业！

1. 在SEU服务器配置环境的时候不要连接V100或者H100，有时候tmp文件夹不可用，应该连接专用的配置环境ssh fat01

1. **修改代码，verl agent类强化学习框架，训练的时候只创建训练环境，在代码里面不要创建eval环境，可减少一半内存消耗。**需要在每个环境：alfworld/webshop等设置，训练时候只创建训练环境，测试时候只创建测试环境。

![image.png](Verl-agent%E6%A1%86%E6%9E%B6%E5%9C%A8SEU%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9/image.png)

**增加参数设置：trainer.train_only/trainer.eval_only,表示只创建train/eval 环境，trainner.test_freq=-1,不进行测试**

![image.png](Verl-agent%E6%A1%86%E6%9E%B6%E5%9C%A8SEU%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9/image%201.png)

./agent_system/environments/env_manager.py 在每个环境：alfworld,webshop设置 在训练时候只创建训练环境不创建测试环境，在测试时候只创建测试环境不创建训练环境

![image.png](Verl-agent%E6%A1%86%E6%9E%B6%E5%9C%A8SEU%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9/image%202.png)

![image.png](Verl-agent%E6%A1%86%E6%9E%B6%E5%9C%A8SEU%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E7%9A%84%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9/image%203.png)

配置文件相应增加参数：**trainer.train_only/trainer.eval_only等，默认false,在slurm里面再设置（val_out是alfworld_out_of_distribution）**

1. verl-agent训练时候的测试结果只能参考，因为val_data_size=128,比较少，更好的做法是训练完模型后，专门测试val_data_size=128*4, env.seed=三个随机种子。结合上一点，把训练和测试分开