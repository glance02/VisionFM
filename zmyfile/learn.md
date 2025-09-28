## 训练指令

1. `CUDA_VISIBLE_DEVICES=2,3,4,5`: 指定使用的GPU设备ID为2,3,4,5。这样可以控制程序在哪些GPU上运行，避免占用其他GPU。

2. `nohup`: 使程序在后台运行，即使关闭终端也不会停止。

3. `python3 -m torch.distributed.launch`: 启动PyTorch分布式训练。

   - `--nnodes 1`: 使用1个节点

   - `--node_rank 0`: 当前节点的排名为0

   - `--nproc_per_node=4`: 每个节点使用4个进程（对应4个GPU）

   - `--master_addr=127.0.0.1`: 主节点地址

   - `--master_port=29500:` 主节点端口

4. `main_pretrain.py`: 训练的主程序文件。

5. 参数部分：

   - `--data_path ./dataset/pretrain_random:` 训练数据路径

   - `--modality Fundus:` 指定模态为眼底图像

   - `--norm_last_layer true`: 对最后一层进行归一化

   - `--epochs 400:` 训练400个epoch

   - `--batch_size_per_gpu 12`: 每个GPU的batch size为12

   - `--shared_head true:` 共享head网络

   - `--out_dim 8192:` 输出维度为8192

   - `--output_dir ./results: `输出目录

   - `--global_crops_scale 0.32 1.0:` 全局裁剪的缩放范围

   - `--pred_ratio 0 0.3: `预测比例

   - `--pred_ratio_var 0 0.2: `预测比例的方差

   - `--name Train_Random_Fundus:` 实验名称

   - `--load_pretrain: `加载预训练权重

6. `> train_fundus.log 2>&1 &`: 将输出重定向到日志文件并在后台运行。

## python相关知识

### argparse库

[示例文件](draft.py)

argparse 是Python标准库中专门用于命令行参数解析的模块，它的主要作用就是：

1. 定义参数 - 声明程序可以接受哪些命令行参数

2. 解析参数 - 自动解析用户传入的命令行参数

3. 生成帮助信息 - 自动生成 -h/--help 帮助信息

4. 参数验证 - 验证参数类型和值的有效性

## 大致思路

1. **下游任务解耦**：项目采用“编码器-解码器”结构。编码器负责提取不同模态的图像特征，下游任务（如分类、分割、回归等）由不同的解码器完成。特征提取和任务训练分开，便于灵活组合。

2. **典型流程**：

   * 预训练 VisionFM 编码器（多模态图像）

   * 用预训练权重提取图像特征

   * 用提取的特征训练下游任务解码器（如分类、分割、回归等）

   * 可根据任务和数据灵活调整结构和参数