import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from vllm import LLM

# -------------------------- Qwen3-VL 基础配置 --------------------------
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_RUNNER = "pooling"
TENSOR_PARALLEL_SIZE = 4  # 需与 GPU 数量一致


class Qwen3VLMatcher:
    """
    Qwen3-VL-Embedding-2B 封装类，用于图片相似度匹配。

    特性：
    - 支持传入参考图片文件夹，初始化时预加载所有参考图片并缓存嵌入向量。
    - 计算相似度时直接遍历预存向量，效率提升。
    - 核心功能：计算目标图片与所有参考球员的相似度，返回最高相似度的球员。
    """

    def __init__(
        self,
        reference_dir: str,
        model_name: str = QWEN_MODEL_NAME,
        runner: str = QWEN_RUNNER,
        tensor_parallel_size: int = TENSOR_PARALLEL_SIZE,
    ):
        """
        初始化：加载模型并预加载参考图片。

        Args:
            reference_dir: 参考图片文件夹路径（文件名=球员名，支持 jpg/png/jpeg）。
            model_name: Qwen3-VL 模型名称或路径。
            runner: Qwen runner 类型（通常为 "pooling"）。
            tensor_parallel_size: 多卡并行数（需与 GPU 数量一致）。
        """
        self.model_name = model_name
        self.runner = runner
        self.tensor_parallel_size = tensor_parallel_size
        self.image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"

        # 核心变量：缓存参考图片的 {球员名: 嵌入向量}
        self.reference_embeddings = {}  # {player_name: normalized_embedding}
        self.reference_dir = reference_dir

        if not os.path.exists(reference_dir):
            raise FileNotFoundError(f"参考文件夹不存在：{reference_dir}")

        print("🔍 加载Qwen3-VL模型（多卡）...")
        self.llm = self._load_qwen_model()

        print(f"🔍 预加载参考文件夹下所有图片：{reference_dir}")
        self._load_all_reference_images()

        print(f"✅ 成功加载 {len(self.reference_embeddings)} 个参考球员的向量")
        for player_name in self.reference_embeddings.keys():
            print(f"   - {player_name}")

    def _load_qwen_model(self):
        """加载 Qwen3-VL 模型（仅执行一次）。"""
        return LLM(
            model=self.model_name,
            runner=self.runner,
            max_model_len=144064,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="float16",
            enforce_eager=True,
        )

    def _load_all_reference_images(self):
        """遍历参考文件夹，加载所有图片并计算嵌入向量（缓存）。"""
        for img_name in os.listdir(self.reference_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            player_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.reference_dir, img_name)

            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f"⚠️  跳过无效图片：{img_path}")
                    continue
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)

                emb = self._get_image_embedding(pil_img)
                if emb is not None:
                    self.reference_embeddings[player_name] = emb
            except Exception as e:
                print(f"⚠️  处理参考图片失败 {img_path}：{e}")
                continue

    def _get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        计算单张图片的嵌入向量（L2 归一化）。

        Args:
            image: PIL.Image 对象（RGB 格式）。

        Returns:
            normalized_embedding: 归一化后的嵌入向量（np.ndarray），失败返回 None。
        """
        try:
            inputs = [{"prompt": self.image_placeholder, "multi_modal_data": {"image": image}}]
            outputs = self.llm.embed(inputs)
            # print(f"⚙️  获取图片嵌入，输出信息：{outputs}")
            emb = torch.tensor(outputs[0].outputs.embedding)
            # print(emb)
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=0)
            return emb_norm.cpu().numpy().squeeze()
        except Exception as e:
            print(f"⚠️  计算图片嵌入失败：{e}")
            return None

    def get_top_similar_player(self, target_image) -> Tuple[str, float]:
        """
        计算目标图片与所有参考球员的相似度，返回最高相似度的球员。

        Args:
            target_image: 目标图片（numpy.ndarray/BGR 或 PIL.Image/RGB）。

        Returns:
            (top_player_name, max_similarity): 最高相似度球员名 + 归一化相似度（0-1）。
        """
        if len(self.reference_embeddings) == 0:
            return ("无参考球员", 0.0)

        if isinstance(target_image, np.ndarray):
            if target_image.size == 0:
                return ("空图片", 0.0)
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            target_pil = target_image.convert("RGB")

        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return ("计算失败", 0.0)

        max_sim = -1.0
        top_player = "未匹配"
        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2

            if normalized_sim > max_sim:
                max_sim = normalized_sim
                top_player = player_name

        return (top_player, max_sim)

    def get_all_similarities(self, target_image) -> Dict[str, float]:
        """
        返回目标图片与所有参考球员的相似度字典。

        Args:
            target_image: 目标图片（numpy/PIL）。

        Returns:
            {player_name: normalized_similarity}
        """
        similarities = {}
        if len(self.reference_embeddings) == 0:
            return similarities

        if isinstance(target_image, np.ndarray):
            if target_image.size == 0:
                return similarities
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            target_pil = target_image.convert("RGB")

        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return similarities

        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2
            similarities[player_name] = normalized_sim

        return similarities


# -------------------------- SigLIP 基础配置 --------------------------
SIGLIP_CKPT = "/data/ljy23/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


class SigLIPMatcher:
    """
    SigLIP 模型封装类（与 Qwen3VLMatcher 接口完全对齐）。

    用于图片相似度匹配，效率极高。
    """

    def __init__(
        self,
        reference_dir: str,
        ckpt: str = SIGLIP_CKPT,
        device: str = DEVICE,
        torch_dtype: torch.dtype = TORCH_DTYPE,
    ):
        """
        初始化：加载 SigLIP 模型并预加载参考图片。

        Args:
            reference_dir: 参考图片文件夹路径。
            ckpt: SigLIP 模型权重名称或路径。
            device: 运行设备（"cuda" 或 "cpu"）。
            torch_dtype: 模型精度。
        """
        self.ckpt = ckpt
        self.device = device
        self.torch_dtype = torch_dtype

        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.reference_dir = reference_dir

        if not os.path.exists(reference_dir):
            raise FileNotFoundError(f"参考文件夹不存在：{reference_dir}")

        print("🔍 加载 SigLIP 模型和处理器...")
        self.model, self.processor = self._load_siglip_model()

        print(f"🔍 预加载参考文件夹下所有图片：{reference_dir}")
        self._load_all_reference_images()

        print(f"✅ 成功加载 {len(self.reference_embeddings)} 个参考球员的向量")
        for player_name in self.reference_embeddings.keys():
            print(f"   - {player_name}")

    def _load_siglip_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """加载 SigLIP 模型和处理器（仅执行一次）。"""
        try:
            processor = AutoProcessor.from_pretrained(self.ckpt)
            model = AutoModel.from_pretrained(
                self.ckpt,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else self.device,
            ).eval()
            return model, processor
        except Exception as e:
            raise RuntimeError(f"加载 SigLIP 模型失败：{e}")

    def _load_all_reference_images(self):
        """遍历参考文件夹，加载所有图片并计算嵌入向量（缓存）。"""
        for img_name in os.listdir(self.reference_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            player_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.reference_dir, img_name)

            try:
                image = load_image(img_path)
                emb = self._get_image_embedding(image)
                if emb is not None:
                    self.reference_embeddings[player_name] = emb
            except Exception as e:
                print(f"⚠️  处理参考图片失败 {img_path}：{e}")
                continue

    def _get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        计算单张图片的嵌入向量（L2 归一化）。

        Args:
            image: PIL.Image 对象（RGB 格式）。

        Returns:
            normalized_embedding: 归一化后的嵌入向量（np.ndarray），失败返回 None。
        """
        try:
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device, dtype=self.torch_dtype)

            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)

            emb_norm = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            return emb_norm.cpu().numpy().squeeze()
        except Exception as e:
            print(f"⚠️  计算图片嵌入失败：{e}")
            return None

    def get_top_similar_player(self, target_image) -> Tuple[str, float]:
        """
        计算目标图片与所有参考球员的相似度，返回最高相似度的球员。

        Args:
            target_image: 目标图片（numpy.ndarray/BGR 或 PIL.Image/RGB）。

        Returns:
            (top_player_name, max_similarity): 最高相似度球员名 + 归一化相似度（0-1）。
        """
        if len(self.reference_embeddings) == 0:
            return ("无参考球员", 0.0)

        if isinstance(target_image, np.ndarray):
            if target_image.size == 0:
                return ("空图片", 0.0)
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            target_pil = target_image.convert("RGB")

        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return ("计算失败", 0.0)

        max_sim = -1.0
        top_player = "未匹配"
        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2

            if normalized_sim > max_sim:
                max_sim = normalized_sim
                top_player = player_name

        return (top_player, max_sim)

    def get_all_similarities(self, target_image) -> Dict[str, float]:
        """
        返回目标图片与所有参考球员的相似度字典。

        Args:
            target_image: 目标图片（numpy/PIL）。

        Returns:
            {player_name: normalized_similarity}
        """
        similarities = {}
        if len(self.reference_embeddings) == 0:
            return similarities

        if isinstance(target_image, np.ndarray):
            if target_image.size == 0:
                return similarities
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            target_pil = target_image.convert("RGB")

        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return similarities

        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2
            similarities[player_name] = normalized_sim

        return similarities
