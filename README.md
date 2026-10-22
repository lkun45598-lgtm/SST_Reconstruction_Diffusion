# SST_Diffusion

OSTIA 原生分辨率预训练 -> JAXA 自监督适配 -> JAXA 0.02° 网格重建 的 U 形 FNO 扩散实验。所有本项目内容都放在这个目录下。

- `code/`   训练、评估代码(`ufno.py` 模型、`edm.py` 去噪器/目标/采样器、`data.py`、`train.py`、`eval_*.py`、`run_full.sh`)
- `cache/`  缓存数据(OSTIA 原生 0.05°,1993-2014,8004 日(缺 2014-07) x 500x600;JAXA L3 适配帧 3996 + 云洞库 1500);由 `code/cache_data.py` 生成
- `runs/`   训练与评估输出。`ostia_full`、`jaxa_adapt`、`eval_jaxa_full` 是 A100 上的完整训练(4 卡);`pilot_*` 是本地 4090 上的试点
- `docs/`   背景与方案说明

远端 A100 服务器上有同结构的工作副本,本地 runs 是从那里拷回的。
本地运行需 `FNO_CACHE` 默认指向 `cache/`;A100 上由 `run_full.sh` 设置。
