# Basketball Highlight Detection

此项目包含篮球精彩片段检测的流水线代码。

## 项目结构

- `src/track/`: 源代码目录
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

1.  安装依赖：
    ```bash
    uv sync
    source .venv/bin/activate
    ```

2.  运行流水线：
    在项目根目录下运行：
    ```bash
    HF_ENDPOINT="https://hf-mirror.com" python -m src.track.pipeline
    ```

## 注意事项

- 代码中的文件路径已更新为相对于项目根目录的 `assets/` 路径。请确保在项目根目录下运行脚本。
- 如需修改配置，请参考 `pipeline.py` 中的配置部分。
