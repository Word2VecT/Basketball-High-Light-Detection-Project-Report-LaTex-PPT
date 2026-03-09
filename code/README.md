# HAR

此项目包含复杂动作行为识别与序列生成代码。

完成情况

- [x] 人物追踪
- [] 动作识别
- [] 高光片段评价与剪辑

## 项目结构

- `src/`: 源代码目录
  - `track/`: 人物追踪代码
    - `pipeline.py`: 主处理流水线
    - `traj_*.py`: 轨迹生成、平滑、匹配、ReID等模块
    - `siglip.py`: SigLIP 模型封装
- `assets/`: 资源文件
  - `court__bg.png`: 球场背景图
  - `homo/`: 单应性矩阵文件
  - `ref/`: 参考图片
  - `ref1/`: 11.19 参考图片

## 使用方法

此项目使用 `uv` 进行依赖管理。

在 `src/track/pipeline.py` 中配置视频路径等。

1.  安装依赖：
    ```bash
    uv sync
    source .venv/bin/activate
    ```

2.  运行流水线：
    在项目根目录下运行：
    ```bash
    export HF_ENDPOINT="https://hf-mirror.com"  # 如果有网络问题
    python -m src.track.pipeline
    ```