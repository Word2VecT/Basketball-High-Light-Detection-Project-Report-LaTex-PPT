import torch
import cv2
import os
import warnings
import numpy as np
from vllm import LLM
from PIL import Image
from io import BytesIO
from typing import Optional
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
from typing import Optional, Dict, Tuple
# -------------------------- Qwen3-VL基础配置（可单独配置） --------------------------
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_RUNNER = "pooling"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 多卡配置
TENSOR_PARALLEL_SIZE = 4  # 需与GPU数量一致

class Qwen3VLMatcher:
    """
    优化版Qwen3-VL-Embedding-2B封装类
    - 支持传入参考图片文件夹，初始化时预加载所有参考图片并缓存嵌入向量
    - 计算相似度时直接遍历预存向量，无需重复处理参考图片，效率提升
    - 核心输出：目标图片与所有参考球员的相似度，返回最高相似度的球员
    """
    
    def __init__(
        self,
        reference_dir: str,  # 参考图片文件夹路径（核心修改）
        model_name: str = QWEN_MODEL_NAME,
        runner: str = QWEN_RUNNER,
        tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    ):
        """
        初始化：加载模型 + 预加载参考文件夹所有图片并缓存向量
        Args:
            reference_dir: 参考图片文件夹路径（文件名=球员名，支持jpg/png/jpeg）
            model_name: Qwen3-VL模型名称/路径
            runner: Qwen runner类型（pooling）
            tensor_parallel_size: 多卡并行数（与GPU数量一致）
        """
        self.model_name = model_name
        self.runner = runner
        self.tensor_parallel_size = tensor_parallel_size
        self.image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        
        # 核心变量：缓存参考图片的{球员名: 嵌入向量}
        self.reference_embeddings = {}  # {player_name: normalized_embedding}
        self.reference_dir = reference_dir
        
        # 1. 校验参考文件夹
        if not os.path.exists(reference_dir):
            raise FileNotFoundError(f"参考文件夹不存在：{reference_dir}")
        
        # 2. 加载Qwen3-VL模型（仅加载一次）
        print("🔍 加载Qwen3-VL模型（多卡）...")
        self.llm = self._load_qwen_model()
        
        # 3. 预加载并处理所有参考图片（缓存向量）
        print(f"🔍 预加载参考文件夹下所有图片：{reference_dir}")
        self._load_all_reference_images()
        
        # 4. 输出加载结果
        print(f"✅ 成功加载 {len(self.reference_embeddings)} 个参考球员的向量")
        for player_name in self.reference_embeddings.keys():
            print(f"   - {player_name}")

    def _load_qwen_model(self):
        """加载Qwen3-VL模型（仅执行一次）"""
        return LLM(
            model=self.model_name,
            runner=self.runner,
            max_model_len=144064,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="float16",
            enforce_eager=True
        )

    def _load_all_reference_images(self):
        """遍历参考文件夹，加载所有图片并计算嵌入向量（缓存）"""
        for img_name in os.listdir(self.reference_dir):
            # 过滤有效图片格式
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            # 球员名 = 文件名（不含后缀）
            player_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.reference_dir, img_name)
            
            # 读取并预处理图片（BGR→RGB，转PIL）
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f"⚠️  跳过无效图片：{img_path}")
                    continue
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # 计算该参考图片的嵌入向量并缓存
                emb = self._get_image_embedding(pil_img)
                if emb is not None:
                    self.reference_embeddings[player_name] = emb
            except Exception as e:
                print(f"⚠️  处理参考图片失败 {img_path}：{e}")
                continue

    def _get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        内部方法：计算单张图片的嵌入向量（L2归一化）
        Args:
            image: PIL.Image对象（RGB格式）
        Returns:
            normalized_embedding: 归一化后的嵌入向量（np.ndarray），失败返回None
        """
        try:
            # 构造Qwen输入
            inputs = [{"prompt": self.image_placeholder, "multi_modal_data": {"image": image}}]
            # 计算嵌入
            outputs = self.llm.embed(inputs)
            emb = torch.tensor(outputs[0].outputs.embedding)
            # L2归一化（保证相似度计算准确性）
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
            # 转为numpy数组（方便后续遍历计算）
            return emb_norm.cpu().numpy().squeeze()  # 形状：(dim,)
        except Exception as e:
            print(f"⚠️  计算图片嵌入失败：{e}")
            return None

    def get_top_similar_player(self, target_image) -> tuple[str, float]:
        """
        核心方法：计算目标图片与所有参考球员的相似度，返回最高相似度的球员
        Args:
            target_image: 目标图片（numpy.ndarray/BGR 或 PIL.Image/RGB）
        Returns:
            (top_player_name, max_similarity): 最高相似度球员名 + 归一化相似度（0-1）
        """
        # 0. 校验参考向量是否为空
        if len(self.reference_embeddings) == 0:
            return ("无参考球员", 0.0)
        
        # 1. 处理目标图片格式并计算嵌入向量
        if isinstance(target_image, np.ndarray):
            # numpy数组（BGR）→ PIL（RGB）
            if target_image.size == 0:
                return ("空图片", 0.0)
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            # PIL图片 → 确保RGB格式
            target_pil = target_image.convert("RGB")
        
        # 2. 计算目标图片的嵌入向量（仅计算一次）
        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return ("计算失败", 0.0)
        
        # 3. 遍历所有参考向量，计算余弦相似度（0-1）
        max_sim = -1.0
        top_player = "未匹配"
        for player_name, ref_emb in self.reference_embeddings.items():
            # 计算余弦相似度（numpy版，效率更高）
            sim = np.dot(ref_emb, target_emb)
            # 归一化到0-1区间（原余弦相似度范围[-1,1]）
            normalized_sim = (sim + 1) / 2
            
            # 更新最高相似度
            if normalized_sim > max_sim:
                max_sim = normalized_sim
                top_player = player_name
        
        return (top_player, max_sim)

    def get_all_similarities(self, target_image) -> dict[str, float]:
        """
        扩展方法：返回目标图片与所有参考球员的相似度字典
        Args:
            target_image: 目标图片（numpy/PIL）
        Returns:
            {player_name: normalized_similarity}
        """
        similarities = {}
        if len(self.reference_embeddings) == 0:
            return similarities
        
        # 处理目标图片并计算嵌入
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
        
        # 遍历所有参考球员计算相似度
        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2
            similarities[player_name] = normalized_sim
        
        return similarities


# -------------------------- SigLIP 基础配置（可自定义） --------------------------
SIGLIP_CKPT = "/data/ljy23/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 多卡配置（如需多卡，设置 device_map="auto" 即可）
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

class SigLIPMatcher:
    """
    SigLIP 模型封装类（与 Qwen3VLMatcher 接口完全对齐）
    - 支持传入参考图片文件夹，初始化时预加载所有参考图片并缓存嵌入向量
    - 计算相似度时直接遍历预存向量，效率极高
    - 核心输出：目标图片与所有参考球员的相似度，返回最高相似度的球员
    """
    
    def __init__(
        self,
        reference_dir: str,  # 参考图片文件夹路径（核心参数，文件名=球员名）
        ckpt: str = SIGLIP_CKPT,
        device: str = DEVICE,
        torch_dtype: torch.dtype = TORCH_DTYPE
    ):
        """
        初始化：加载 SigLIP 模型 + 预加载参考文件夹所有图片并缓存向量
        Args:
            reference_dir: 参考图片文件夹路径（支持 jpg/png/jpeg，文件名=球员名）
            ckpt: SigLIP 模型权重名称/本地路径（如 google/siglip2-giant-opt-patch16-384）
            device: 运行设备（cuda/cpu）
            torch_dtype: 模型精度（float16/float32）
        """
        self.ckpt = ckpt
        self.device = device
        self.torch_dtype = torch_dtype
        
        # 核心变量：缓存参考图片的 {球员名: 归一化嵌入向量}
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.reference_dir = reference_dir
        
        # 1. 校验参考文件夹
        if not os.path.exists(reference_dir):
            raise FileNotFoundError(f"参考文件夹不存在：{reference_dir}")
        
        # 2. 加载 SigLIP 模型和处理器（仅加载一次）
        print("🔍 加载 SigLIP 模型和处理器...")
        self.model, self.processor = self._load_siglip_model()
        
        # 3. 预加载并处理所有参考图片（缓存向量）
        print(f"🔍 预加载参考文件夹下所有图片：{reference_dir}")
        self._load_all_reference_images()
        
        # 4. 输出加载结果
        print(f"✅ 成功加载 {len(self.reference_embeddings)} 个参考球员的向量")
        for player_name in self.reference_embeddings.keys():
            print(f"   - {player_name}")

    def _load_siglip_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """加载 SigLIP 模型和处理器（仅执行一次）"""
        try:
            # 加载处理器
            processor = AutoProcessor.from_pretrained(self.ckpt)
            # 加载模型（自动适配多卡/单卡）
            model = AutoModel.from_pretrained(
                self.ckpt,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else self.device
            ).eval()  # 推理模式
            return model, processor
        except Exception as e:
            raise RuntimeError(f"加载 SigLIP 模型失败：{e}")

    def _load_all_reference_images(self):
        """遍历参考文件夹，加载所有图片并计算嵌入向量（缓存）"""
        for img_name in os.listdir(self.reference_dir):
            # 过滤有效图片格式
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            # 球员名 = 文件名（不含后缀）
            player_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.reference_dir, img_name)
            
            # 读取并预处理图片
            try:
                # 读取图片（支持本地路径）
                image = load_image(img_path)
                # 计算该参考图片的嵌入向量并缓存
                emb = self._get_image_embedding(image)
                if emb is not None:
                    self.reference_embeddings[player_name] = emb
            except Exception as e:
                print(f"⚠️  处理参考图片失败 {img_path}：{e}")
                continue

    def _get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        内部方法：计算单张图片的嵌入向量（L2 归一化）
        Args:
            image: PIL.Image 对象（RGB 格式）
        Returns:
            normalized_embedding: 归一化后的嵌入向量（np.ndarray，形状 (dim,)），失败返回 None
        """
        try:
            # 预处理图片（模型要求的格式）
            inputs = self.processor(
                images=[image],
                return_tensors="pt"
            ).to(self.device, dtype=self.torch_dtype)
            
            # 推理计算嵌入（无梯度）
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)
            
            # L2 归一化（保证余弦相似度计算准确性）
            emb_norm = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            
            # 转为 numpy 数组（方便后续遍历计算）
            return emb_norm.cpu().numpy().squeeze()  # 形状：(dim,)
        except Exception as e:
            print(f"⚠️  计算图片嵌入失败：{e}")
            return None

    def get_top_similar_player(self, target_image) -> Tuple[str, float]:
        """
        核心方法：计算目标图片与所有参考球员的相似度，返回最高相似度的球员
        Args:
            target_image: 目标图片（numpy.ndarray/BGR 或 PIL.Image/RGB）
        Returns:
            (top_player_name, max_similarity): 最高相似度球员名 + 归一化相似度（0-1）
        """
        # 0. 校验参考向量是否为空
        if len(self.reference_embeddings) == 0:
            return ("无参考球员", 0.0)
        
        # 1. 处理目标图片格式
        if isinstance(target_image, np.ndarray):
            # numpy 数组（BGR，OpenCV 格式）→ PIL（RGB）
            if target_image.size == 0:
                return ("空图片", 0.0)
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            # PIL 图片 → 确保 RGB 格式
            target_pil = target_image.convert("RGB")
        
        # 2. 计算目标图片的嵌入向量（仅计算一次）
        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return ("计算失败", 0.0)
        
        # 3. 遍历所有参考向量，计算余弦相似度（归一化到 0-1 区间）
        max_sim = -1.0
        top_player = "未匹配"
        for player_name, ref_emb in self.reference_embeddings.items():
            # 计算余弦相似度（numpy 版，效率更高）
            sim = np.dot(ref_emb, target_emb)
            # 归一化到 0-1 区间（原余弦相似度范围 [-1,1]）
            normalized_sim = (sim + 1) / 2
            
            # 更新最高相似度
            if normalized_sim > max_sim:
                max_sim = normalized_sim
                top_player = player_name
        
        return (top_player, max_sim)

    def get_all_similarities(self, target_image) -> Dict[str, float]:
        """
        扩展方法：返回目标图片与所有参考球员的相似度字典
        Args:
            target_image: 目标图片（numpy/PIL）
        Returns:
            {player_name: normalized_similarity}
        """
        similarities = {}
        if len(self.reference_embeddings) == 0:
            return similarities
        
        # 处理目标图片格式
        if isinstance(target_image, np.ndarray):
            if target_image.size == 0:
                return similarities
            rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(rgb_img)
        else:
            target_pil = target_image.convert("RGB")
        
        # 计算目标图片嵌入
        target_emb = self._get_image_embedding(target_pil)
        if target_emb is None:
            return similarities
        
        # 遍历所有参考球员计算相似度
        for player_name, ref_emb in self.reference_embeddings.items():
            sim = np.dot(ref_emb, target_emb)
            normalized_sim = (sim + 1) / 2
            similarities[player_name] = normalized_sim
        
        return similarities