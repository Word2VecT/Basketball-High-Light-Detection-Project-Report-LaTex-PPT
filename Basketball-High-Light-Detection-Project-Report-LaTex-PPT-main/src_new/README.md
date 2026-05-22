# 代码简介
当前实现多个视角多线程处理，然后主线程进行轨迹的融合、id的统计可视化

原本速度很快 但hugestone上最近跑的特别慢，不知道是不是服务器问题

- track/: 人物追踪代码
- pipeline.py: 主处理流水线
- traj_*.py: 轨迹生成、平滑、匹配、ReID等模块

# 环境
把src_new 改名为src替换code下面的src 没有直接覆盖是为了保留原版，方便直接下载

环境准备（在code文件夹下）(如果之前装过环境就不用了）
> uv sync
> 
> source .venv/bin/activate

运行：
> export HF_ENDPOINT="https://hf-mirror.com"  # 如果有网络问题
> 
> python -m src.track.pipeline

# demo
一个效果一般的demo，缺失了很多轨迹，主要是想测试能否跑通获得初步效果

# Todo
- [ ] 各视角线程和主线程实现异步
- [ ] 追踪和reid的逻辑有重叠，能不能检测+reid，或追踪+极少帧的reid
