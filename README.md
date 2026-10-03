# EGC-PMCC 5.0 · 多负荷预测复现包

**简体中文** | [English](README_EN.md)

本仓库面向审稿人和复现者，提供模型源码、版本化处理数据、依赖和固定顺序的复现入口。原始数据、模型权重和本地运行缓存不在仓库中。

## 发布范围

为保护论文正式发表前的研究细节，仓库已移除未公开的实验方案、历史结果表、模型选择记录、审稿跟进材料和历史分析脚本。保留的机器可执行配置仅用于让审稿人按注册顺序重新训练、导出测试预测并生成统计结果。它们不是论文结果档案。

推荐只使用根目录的 `reviewer_recipe.py`。它会依次完成完整性检查、训练、验证分析、测试导出、离线统计、补充分析和效率检查，并将输出写入独立的 `runs/` 目录。

具体顺序见 [审稿人复现顺序](docs/REPRODUCTION_ORDER.md)。

## 安装

```bash
git clone https://github.com/Winter-SunshineO/EGC-PMCC-Multi-load-prediction.git EGC-PMCC5.0
cd EGC-PMCC5.0
conda create -n egc-pmcc-v7 python=3.11.15 -y
conda activate egc-pmcc-v7
python -m pip install -r requirements-public.txt
python reproduce.py check
```

GPU 运行请安装与驱动匹配的 PyTorch CUDA 构建，并将 `--device cuda:0` 替换为可用设备。CPU 也可以运行，但耗时和数值可能不同。

## 审稿人复现顺序

完整运行：

```bash
python reviewer_recipe.py --run-root runs/reviewer --device cuda:0 --execute
```

短接口检查：

```bash
python reviewer_recipe.py --run-root runs/smoke --device cuda:0 --smoke --execute
```

中断后可使用相同命令增加 `--resume`。如需直接调试，可使用 `reproduce.py`，但应保持 `reviewer_recipe.py` 中的顺序，并在修改源码、配置、数据或设备后使用新的 `--run-root`。

## 数据

处理后的 v7 数据位于 `data/preprocessed_forecasting_v7/`。训练入口会自动读取输入表、标签表、有效性掩码和预处理元数据。原始合并数据没有发布；数据条款见 [DATA_LICENSE.md](DATA_LICENSE.md)。

## 输出

每次运行会在指定根目录保存环境指纹、执行记录、模型检查点、验证输出、测试导出和离线统计。这些运行目录被 `.gitignore` 排除，不应提交到 GitHub。

## 许可证与引用

项目代码使用 [LICENSE](LICENSE)。第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；处理数据条款见 [DATA_LICENSE.md](DATA_LICENSE.md)。论文尚未正式发表，完整引用将在适当时补充。
