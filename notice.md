# Verl-agent 框架在 SEU 服务器上的使用注意事项

## 1. Wandb 离线设置

服务器断网，需使用 wandb 离线模式。学习如何设置离线 wandb，并将离线文件上传到个人账号。

参考：https://blog.csdn.net/weixin_42336668/article/details/147918322

## 2. 路径配置

运行 shell 脚本需设置以下路径参数：

```bash
trainer.default_local_dir=''
actor_rollout_ref.model.path=
```

- `/seu_share2/` — 机械硬盘（推荐存 checkpoint、大文件）
- `/seu_share/` — 固态硬盘（默认目录）

## 3. 内存管理

- 用 `free -h` 查看服务器当前内存使用情况
- 估算每个 verl-agent 程序的内存消耗，防止多人程序同时跑时爆内存导致服务器挂掉

## 4. H100 作业提交（重要）

**不要在 VSCode 上长期跑训练作业！** 短期调试可用 VSCode，长期训练必须用 Slurm 提交：

```bash
sbatch <job_script>   # 提交作业
squeue                # 查看作业队列
```

详见用户手册。违规会导致其他用户无法正常分配显卡资源。

## 5. Debug 调试

verl-agent 支持 Ray debug，参考：https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html

调试时可减小参数加速跑通：

```bash
train_datasize=<小值>
max_step=<小值>
```

## 6. 公共模型目录

HuggingFace 模型（如 Qwen2.5）可放在公共目录，避免每人重复下载：

```
/seu_nvme/ogai/model
```

> 注意：目前公共目录存在权限问题，暂时忽略，各自下载。

## 7. h005 节点

h005 节点是全院公共节点，资源紧张时可排队使用。

## 8. Checkpoint 管理

- 保存路径改为机械硬盘：`trainer.default_local_dir='/seu_share2/...'`
- 保存频率建议：`trainer.save_freq=40`
- 跑完 160 epoch 后，手动清理早期不用的 checkpoint，防止磁盘空间不足

## 9. 模型规模建议

- 先用 **2 卡 + 1.5B 小模型** 验证 idea，跑出效果后再申请 4 卡跑 7B 模型

## 10. Baseline 复用

部分 baseline 对比算法可能已有人训练并保存了 checkpoint，先询问组内成员，避免重复消耗计算资源。

## 11. 排队作业的文件修改行为

- 作业**排队等待期间**修改代码/参数文件，修改会生效（作业运行时读取的是当前文件）
- 作业**已开始运行后**修改文件，不影响正在运行的作业

## 12. 配置环境节点

配置环境时，连接专用节点：

```bash
ssh fat01
```

不要在 V100 或 H100 节点上配置环境，`/tmp` 文件夹有时不可用会导致失败。

## 13. 训练时只创建训练环境（重要，节省一半内存）

verl-agent 默认同时创建训练和测试环境，内存消耗翻倍。训练时关闭测试环境：

**配置参数：**

```yaml
trainer.train_only: true    # 训练时只创建训练环境
trainer.test_freq: -1       # 不在训练中途进行测试
# 或
trainer.eval_only: true     # 测试时只创建测试环境
```

**修改文件：** `agent_system/environments/env_manager.py`

在每个环境（alfworld、webshop 等）中设置：训练时只创建训练环境，测试时只创建测试环境。

在 Slurm 脚本中通过命令行参数传入（`val_out` 对应 alfworld out-of-distribution）。

## 14. 训练与测试分离

训练过程中的 val 结果仅供参考（`val_data_size=128` 样本量太少，结果不稳定）。

正式评估方式：

- 训练完成后，单独跑测试
- `val_data_size=512`（128×4）
- 使用 3 个不同随机种子，取平均结果
