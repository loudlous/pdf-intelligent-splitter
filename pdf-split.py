#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用PDF智能拆分工具（基于大模型和OCR）
适用于将合并的PDF文档拆分为多个独立文档

核心功能：
1. 自动OCR识别PDF文本（或读取已有OCR JSON）
2. 优先使用目录进行精确拆分（零成本，不调用模型）
3. 目录不可用时，使用大模型进行智能拆分
4. 自动拆分PDF并规范化命名
5. 支持所有文档类型（不依赖预设规则）

使用场景：
- 多个小文档被合并为一个大的PDF文档
- 需要将PDF按文档边界拆分为独立文件
- 支持有目录和无目录的PDF文档
"""

import os
import re
import json
import gc
import argparse
import logging
import subprocess
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv可选

from openai import OpenAI

# PDF处理库
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# OCR库
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    # 捕获所有可能的导入错误（包括系统库缺失）
    # 注意：此时logger可能还未定义，使用print输出
    import sys
    print(f"警告: PaddleOCR导入失败: {e}", file=sys.stderr)
    OCR_AVAILABLE = False

# 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('criminal_case_splitter_llm.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentSplit:
    """文档拆分结果（只包含切点信息）"""
    start_page: int  # 起始页（1-based）
    end_page: int    # 结束页（1-based）
    title: str       # 文档标题（用于后续识别）


@dataclass
class TextBlock:
    """文本块信息"""
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    page_num: int


class LLMDocumentSplitter:
    """基于大模型的通用文档拆分器"""
    
    # 通用配置：补充材料关键词（可扩展）
    SUPPLEMENT_KEYWORDS = {
        'appendix': ['appendix', '附录'],
        'references': ['references', 'bibliography', '参考文献'],
        'supplementary': ['supplementary', '补充材料', 'supplement']
    }
    
    # 通用配置：页面类型检测关键词
    PAGE_TYPE_KEYWORDS = {
        'toc': ['目录', 'contents', 'table of contents', '目 录'],
        'abstract': ['abstract', '摘要'],
        'references': ['references', 'bibliography', '参考文献'],
        'title_page': ['abstract', '摘要', 'introduction', '引言', 'author', '作者', 
                      'university', '大学', 'institute', '学院', 'department', '系']
    }
    
    @staticmethod
    def is_blank_page(page: "fitz.Page",
                      text_threshold: int = 5,
                      scale: float = 0.5,
                      sample_step: int = 23,
                      white_threshold: int = 250,
                      blank_ratio_threshold: float = 0.995) -> bool:
        """
        检测整页是否为空白页（尽量轻量，避免额外依赖numpy）。
        - 先用PDF文本快速判断
        - 再用低分辨率灰度渲染，抽样计算"接近白色"的像素占比
        """
        try:
            text = page.get_text("text").strip()
            if len(text) > text_threshold:
                return False
        except Exception:
            pass

        try:
            m = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=m, colorspace=fitz.csGRAY, alpha=False)
            samples = memoryview(pix.samples)  # 1字节/像素
            n = len(samples)
            if n == 0:
                return True

            # 抽样统计，避免逐像素遍历
            step = max(1, int(sample_step))
            total = (n + step - 1) // step
            white = 0
            for i in range(0, n, step):
                if samples[i] >= white_threshold:
                    white += 1
            ratio = white / max(1, total)
            return ratio >= blank_ratio_threshold
        except Exception:
            try:
                text = page.get_text("text").strip()
                return len(text) <= text_threshold
            except Exception:
                return False
    
    def __init__(self, pdf_path: str, output_dir: str = "test_result", 
                 ocr_json_path: Optional[str] = None, use_gpu: Optional[bool] = None,
                 image_scale: float = 1.0, document_type: str = "general"):
        """
        初始化拆分器
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            ocr_json_path: OCR结果JSON文件路径（可选，如果不提供则自动OCR）
            use_gpu: 是否使用GPU（None=自动检测, True=强制GPU, False=强制CPU）
            image_scale: 图像缩放比例
            document_type: 文档类型（"general"通用, "legal"法律文书, "academic"学术文档等）
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.ocr_json_path = Path(ocr_json_path) if ocr_json_path else None
        self.image_scale = image_scale
        self.use_gpu_override = use_gpu
        self.document_type = document_type  # 文档类型，用于调整拆分策略
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 延迟初始化OpenAI客户端（只有在需要调用LLM时才初始化）
        self.client = None
        # API key：仅通过环境变量提供（不在代码中硬编码默认值，便于开源安全）
        # 支持 DEEPSEEK_API_KEY（旧）和 LLM_API_KEY（新）环境变量
        self._api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        # API base URL：优先使用环境变量，否则使用默认值
        # 注意：OpenAI客户端会自动添加 /chat/completions 路径，所以base_url只需要到 /v1
        self._api_base_url = os.getenv("LLM_API_BASE_URL") or "https://one-api.maas.com.cn/v1"
        
        # 打开PDF
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF未安装，请执行：pip install pymupdf")
        
        self.doc = fitz.open(str(self.pdf_path))
        self.total_pages = len(self.doc)
        
        # 初始化OCR（如果需要）
        self.ocr = None
        self._gpu_available = None
        
        # 检测空白页
        self.blank_pages: Set[int] = self._detect_blank_pages()
        
        logger.info(f"PDF文件: {self.pdf_path}")
        logger.info(f"总页数: {self.total_pages}")
        logger.info(f"空白页数量: {len(self.blank_pages)}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"文档类型: {self.document_type}")
        if self.ocr_json_path:
            logger.info(f"OCR JSON文件: {self.ocr_json_path}")
        else:
            logger.info("将自动进行OCR识别")
    
    def _detect_blank_pages(self) -> Set[int]:
        """检测整页空白页（页码1-based）"""
        blank: Set[int] = set()
        if self.total_pages <= 0:
            return blank

        iterable = range(self.total_pages)
        if TQDM_AVAILABLE:
            iterable = tqdm(iterable, desc="检测空白页", unit="页")

        for i in iterable:
            try:
                page = self.doc[i]
                if self.is_blank_page(page):
                    blank.add(i + 1)
            except Exception:
                continue

        return blank
    
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用（带缓存，优化版）"""
        if self._gpu_available is not None:
            return self._gpu_available
        
        # 如果用户明确指定，直接使用
        if self.use_gpu_override is not None:
            self._gpu_available = self.use_gpu_override
            logger.info(f"用户指定使用{'GPU' if self._gpu_available else 'CPU'}模式")
            return self._gpu_available
        
        # 自动检测GPU（优化版：先检查nvidia-smi，再检查PaddlePaddle）
        try:
            # 1. 先检查nvidia-smi（最快的方法）
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                gpu_info = result.stdout.decode('utf-8', errors='ignore')
                if any(kw in gpu_info for kw in ['NVIDIA', 'GeForce', 'Tesla', 'Quadro', 'RTX', 'GTX']):
                    # 2. 检查PaddlePaddle是否支持CUDA
                    try:
                        import paddle
                        # 检查PaddlePaddle版本和CUDA支持
                        if hasattr(paddle, 'is_compiled_with_cuda'):
                            if paddle.is_compiled_with_cuda():
                                logger.info("✓ 检测到NVIDIA GPU且PaddlePaddle支持CUDA，将使用GPU加速")
                                self._gpu_available = True
                                return True
                        elif hasattr(paddle, 'fluid') and hasattr(paddle.fluid, 'is_compiled_with_cuda'):
                            if paddle.fluid.is_compiled_with_cuda():
                                logger.info("✓ 检测到NVIDIA GPU且PaddlePaddle支持CUDA，将使用GPU加速")
                                self._gpu_available = True
                                return True
                        # GPU存在但PaddlePaddle不支持CUDA
                        logger.warning("检测到GPU但PaddlePaddle未编译CUDA支持，将使用CPU模式")
                        self._gpu_available = False
                        return False
                    except ImportError:
                        # PaddlePaddle未安装，但GPU存在，仍然尝试使用GPU（PaddleOCR可能会处理）
                        logger.info("✓ 检测到NVIDIA GPU，将尝试使用GPU加速（PaddlePaddle未安装）")
                        self._gpu_available = True
                        return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # nvidia-smi不存在或超时，说明没有GPU
            pass
        except Exception as e:
            logger.debug(f"GPU检测过程出错: {e}")
        
        # 没有检测到GPU
        logger.info("未检测到GPU或GPU不可用，将使用CPU模式")
        self._gpu_available = False
        return False
    
    def _init_ocr(self):
        """初始化OCR引擎（自动检测GPU，优化版）"""
        if self.ocr is not None:
            return

        use_gpu = self._check_gpu_available()
        
        logger.info(f"初始化PaddleOCR引擎... (GPU: {'启用' if use_gpu else '禁用'})")
        
        # 获取PaddleOCR版本
        try:
            import paddleocr
            version = getattr(paddleocr, '__version__', '2.8.0')
            logger.info(f"PaddleOCR版本: {version}")
        except:
            version = '2.8.0'  # 默认版本
        
        try:
            # PaddleOCR 2.8.0版本配置（当前使用的版本）
            if version.startswith('2.'):
                if use_gpu:
                    # GPU版本配置
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch',
                        show_log=False,
                        use_gpu=True,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                        max_text_length=50,
                        use_dilation=True,
                        det_limit_side_len=2560,
                        enable_mkldnn=False,
                        cpu_threads=1
                    )
                    logger.info("OCR引擎初始化成功（GPU模式）")
                else:
                    # CPU版本配置
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch',
                        show_log=False,
                        use_gpu=False,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                        max_text_length=25,
                        use_dilation=False,
                        det_limit_side_len=960
                    )
                    logger.info("OCR引擎初始化成功（CPU模式）")
            else:
                # PaddleOCR 3.x版本（使用更简单的配置）
                ocr_kwargs = {
                    'use_angle_cls': True,
                    'lang': 'ch',
                }
                # 3.x版本：尝试设置use_gpu参数
                if use_gpu:
                    ocr_kwargs['use_gpu'] = True
                    # 尝试设置GPU相关参数
                    try:
                        import paddle
                        if hasattr(paddle, 'set_device'):
                            paddle.set_device('gpu')
                            logger.info("已设置PaddlePaddle使用GPU")
                    except:
                        pass
                
                self.ocr = PaddleOCR(**ocr_kwargs)
                logger.info(f"OCR引擎初始化成功（{'GPU' if use_gpu else 'CPU'}模式，3.x版本）")
        except Exception as e:
            logger.error(f"OCR初始化失败: {e}")
            # 尝试使用最简配置（仅必需参数）
            try:
                logger.info("尝试使用最简配置初始化OCR...")
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                logger.info("OCR引擎初始化成功（最简配置，CPU模式）")
                self._gpu_available = False  # 降级到CPU
            except Exception as e2:
                logger.error(f"最简配置也失败: {e2}")
                raise
    
    def ocr_page_batch(self, page_nums: List[int]) -> Dict[int, List[TextBlock]]:
        """批量OCR处理页面"""
        results = {}

        for page_num in page_nums:
            try:
                # 跳过空白页
                if page_num in self.blank_pages:
                    results[page_num] = []
                    continue
            
                page = self.doc[page_num - 1]  # 0-based

                # 转换为图像
                matrix = fitz.Matrix(self.image_scale, self.image_scale)
                pix = page.get_pixmap(matrix=matrix)

                # 转换为numpy数组
                import numpy as np
                from PIL import Image

                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                if pix.n == 4:
                    pil_img = Image.fromarray(img, 'RGBA')
                else:
                    pil_img = Image.fromarray(img, 'RGB')

                # OCR识别
                self._init_ocr()
                # 新版本PaddleOCR可能不支持cls参数，尝试兼容
                try:
                    ocr_result = self.ocr.ocr(np.array(pil_img), cls=True)
                except TypeError:
                    # 如果不支持cls参数，使用不带cls的调用
                    ocr_result = self.ocr.ocr(np.array(pil_img))

                # 提取文本信息
                text_blocks = []
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        if line and len(line) >= 2:
                            bbox = line[0]
                            text_content = line[1][0]
                            confidence = line[1][1] if len(line[1]) > 1 else 0.0

                            if bbox and len(bbox) >= 4:
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                x_min, x_max = min(x_coords), max(x_coords)
                                y_min, y_max = min(y_coords), max(y_coords)
                                width = x_max - x_min
                                height = y_max - y_min

                                text_blocks.append(TextBlock(
                                    text=text_content,
                                    x=float(x_min),
                                    y=float(y_min),
                                    width=float(width),
                                    height=float(height),
                                    confidence=float(confidence),
                                    page_num=page_num
                                ))

                results[page_num] = text_blocks

                # 清理内存
                del img, pix, pil_img
                gc.collect()

            except Exception as e:
                logger.error(f"第{page_num}页OCR失败: {e}")
                results[page_num] = []

        return results
    
    def generate_ocr_json(self, output_json_path: Optional[str] = None) -> str:
        """生成OCR JSON文件"""
        if output_json_path is None:
            output_json_path = str(self.output_dir / f"{self.pdf_path.stem}_ocr.json")
        
        logger.info(f"开始为 {self.pdf_path} 生成OCR JSON...")
        
        # 获取所有页面（跳过空白页）
        all_pages = [p for p in range(1, self.total_pages + 1) if p not in self.blank_pages]
        logger.info(f"需要处理 {len(all_pages)} 页（总页数: {self.total_pages}，空白页: {len(self.blank_pages)}）")
        
        # 批量OCR处理
        ocr_results = {}
        batch_size = 2  # 减小批次大小，避免内存问题
        
        batches = [all_pages[i:i+batch_size] for i in range(0, len(all_pages), batch_size)]
        batch_iter = batches
        if TQDM_AVAILABLE:
            batch_iter = tqdm(batches, desc="OCR处理", unit="批")
        
        for batch in batch_iter:
            batch_results = self.ocr_page_batch(batch)
            ocr_results.update(batch_results)
        
        # 转换为JSON格式（极简优化：只保留页眉和标题区域文本，大幅减少token消耗）
        pages_data = []
        for page_num in sorted(ocr_results.keys()):
            page = self.doc[page_num - 1]
            page_height = page.rect.height
            
            # 按y坐标排序
            blocks = sorted(ocr_results[page_num], key=lambda b: (b.y, b.x))
            
            # 只保留页眉区域（顶部25%）和标题区域（25%-35%）的文本
            # 这些区域通常包含文档标题、章节名等拆分关键信息
            header_threshold = page_height * 0.25  # 页眉区域：顶部25%
            title_threshold = page_height * 0.35   # 标题区域：25%-35%
            
            # 提取关键文本（页眉+标题区域）
            key_texts = []
            for block in blocks:
                y = block.y
                text = block.text.strip()
                if not text:
                    continue
                # 只保留页眉和标题区域的文本
                if y <= title_threshold:
                    key_texts.append({
                        'text': text,
                        'y': round(y, 1)
                    })
            
            # 如果关键区域文本太少，补充前3行文本（通常是标题）
            if len(key_texts) < 3:
                for i, block in enumerate(blocks[:10]):  # 最多取前10个文本块
                    text = block.text.strip()
                    if text and not any(t['text'] == text for t in key_texts):
                        key_texts.append({
                            'text': text,
                            'y': round(block.y, 1)
                        })
                        if len(key_texts) >= 5:  # 最多保留5个文本块
                            break
            
            pages_data.append({
                'page_num': page_num,
                'page_height': round(page_height, 1),
                'texts': key_texts  # 只保存关键文本
            })
        
        # 保存JSON
        ocr_data = {'pages': pages_data}
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"OCR JSON已保存到: {output_json_path}")
        logger.info(f"共处理 {len(pages_data)} 页")
        
        return str(output_path)
    
    def load_ocr_json(self) -> Dict:
        """加载OCR JSON文件"""
        if not self.ocr_json_path or not self.ocr_json_path.exists():
            raise FileNotFoundError(f"OCR JSON文件不存在: {self.ocr_json_path}")
        
        logger.info(f"加载OCR JSON文件: {self.ocr_json_path}")
        try:
            with open(self.ocr_json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            logger.info(f"OCR JSON加载成功，包含 {len(ocr_data.get('pages', []))} 页")
            return ocr_data
        except Exception as e:
            logger.error(f"加载OCR JSON失败: {e}")
            raise
    
    def extract_page_texts(self, ocr_data: Dict) -> List[Tuple[int, str, str, str]]:
        """
        从OCR JSON中提取每页的文本（优化版：智能合并文本，提取关键区域，减少token消耗）
        
        返回: [(页码, 页眉文本, 关键区域文本, 页面类型和关键信息), ...]
        """
        page_texts = []
        pages_data = ocr_data.get('pages', [])
        
        # 获取所有在OCR JSON中的页码
        ocr_page_nums = set()
        for page_data in pages_data:
            page_num = page_data.get('page_num', 0)
            if page_num > 0:
                ocr_page_nums.add(page_num)
        
        # 处理OCR JSON中的页面
        for page_data in pages_data:
            page_num = page_data.get('page_num', 0)
            if page_num <= 0:
                continue
            
            texts = page_data.get('texts', [])
            page_height = page_data.get('page_height', 0)
            
            # 即使没有文本，也记录这一页（可能是空白页）
            if not texts:
                page_texts.append((page_num, "", "", "空白页"))
                continue

            # 极简优化：只提取页眉文本（顶部25%区域）
            header_text = self._build_header_text_from_ocr_json(texts, page_height, ratio=0.25)
            
            # 提取前3行文本（通常是标题或关键信息），而不是整个关键区域
            # 这样可以大幅减少token消耗，同时保留拆分所需的关键信息
            key_text = self._extract_top_lines_text(texts, page_height, max_lines=3)
            
            # 如果关键文本太短，补充页眉文本
            if len(key_text) < 50:
                key_text = header_text[:200] if len(header_text) > 200 else header_text
            
            # 检测页面类型（针对学术论文优化）
            page_type_info = self._detect_academic_page_type(texts, page_height, header_text, key_text)
            
            # 同时保存页眉、关键区域文本和页面类型信息
            page_texts.append((page_num, header_text, key_text, page_type_info))
        
        # 检查是否有缺失的页面（PDF总页数 vs OCR JSON中的页数）
        missing_pages = set(range(1, self.total_pages + 1)) - ocr_page_nums
        if missing_pages:
            logger.warning(f"OCR JSON中缺失以下页码（可能是空白页）: {sorted(missing_pages)}")
            # 为缺失的页面添加空文本条目，确保大模型知道这些页面的存在
            for page_num in sorted(missing_pages):
                page_texts.append((page_num, "__BLANK_PAGE__", "__BLANK_PAGE__", "空白页"))
            # 重新排序
            page_texts.sort(key=lambda x: x[0])
        
        logger.info(f"提取了 {len(page_texts)} 页的文本（PDF总页数: {self.total_pages}）")
        return page_texts
    
    def _merge_text_blocks(self, texts: List[Dict]) -> str:
        """智能合并文本块，去除冗余空白"""
        if not texts:
            return ""
        
        # 按y坐标分组（同一行的文本）
        lines = {}
        for t in texts:
            text = t.get('text', '').strip()
            if not text:
                continue
            y = t.get('y', 0)
            # 将y坐标四舍五入到最近的5像素，用于分组
            y_group = round(y / 5) * 5
            if y_group not in lines:
                lines[y_group] = []
            lines[y_group].append(text)
        
        # 合并每行的文本
        merged_lines = []
        for y_group in sorted(lines.keys()):
            line_text = " ".join(lines[y_group])
            merged_lines.append(line_text)
        
        # 合并所有行，去除多余空白
        merged_text = " ".join(merged_lines)
        # 去除多个连续空格
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        return merged_text
    
    def _extract_top_lines_text(self, texts: List[Dict], page_height: float, max_lines: int = 3) -> str:
        """提取前N行文本（通常是标题或关键信息），极简优化"""
        if not texts:
            return ""
        
        # 按y坐标排序，取前N行
        sorted_texts = sorted(texts, key=lambda t: t.get('y', 0))
        
        # 按y坐标分组（同一行的文本）
        lines = []
        current_line_y = None
        current_line_texts = []
        
        for t in sorted_texts[:20]:  # 最多检查前20个文本块
            text = t.get('text', '').strip()
            if not text:
                continue
            
            y = t.get('y', 0)
            # 如果y坐标相近（相差小于10），认为是同一行
            if current_line_y is None or abs(y - current_line_y) < 10:
                current_line_texts.append(text)
                if current_line_y is None:
                    current_line_y = y
            else:
                # 新的一行
                if current_line_texts:
                    lines.append(" ".join(current_line_texts))
                current_line_texts = [text]
                current_line_y = y
            
            # 如果已经收集了足够的行，停止
            if len(lines) >= max_lines and current_line_texts:
                lines.append(" ".join(current_line_texts))
                break
        
        # 添加最后一行
        if current_line_texts and len(lines) < max_lines:
            lines.append(" ".join(current_line_texts))
        
        # 合并前N行
        result = " | ".join(lines[:max_lines])
        return result.strip()
    
    def _extract_key_regions_text(self, texts: List[Dict], page_height: float) -> str:
        """提取关键区域文本：页眉30% + 标题区域10% + 页脚10%（保留作为备用方法）"""
        if not texts:
            return ""
        
        try:
            page_height = float(page_height) if page_height else 0.0
        except Exception:
            page_height = 0.0
        
        if page_height <= 0:
            # 如果没有页面高度，返回前3行文本
            return self._extract_top_lines_text(texts, 0, max_lines=3)
        
        # 定义关键区域
        header_threshold = page_height * 0.30  # 页眉区域：顶部30%
        title_threshold = page_height * 0.40   # 标题区域：30%-40%
        footer_threshold = page_height * 0.90  # 页脚区域：底部10%
        
        key_texts = []
        for t in texts:
            y = t.get('y', 0)
            text = t.get('text', '').strip()
            if not text:
                continue
            
            # 提取关键区域的文本
            if y <= header_threshold:  # 页眉区域
                key_texts.append(('header', y, text))
            elif y <= title_threshold:  # 标题区域
                key_texts.append(('title', y, text))
            elif y >= footer_threshold:  # 页脚区域
                key_texts.append(('footer', y, text))
        
        # 如果关键区域文本太少，补充前20%区域的文本
        if len(key_texts) < 5:
            supplement_threshold = page_height * 0.20
            for t in texts:
                y = t.get('y', 0)
                text = t.get('text', '').strip()
                if not text or y > supplement_threshold:
                    break
                if ('supplement', y, text) not in key_texts:
                    key_texts.append(('supplement', y, text))
        
        # 按y坐标排序并合并
        key_texts.sort(key=lambda x: x[1])
        merged_text = " ".join([item[2] for item in key_texts])
        # 去除多余空白
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        return merged_text
    
    def _analyze_page_info(self, header_text: str, full_text: str, texts: List[Dict]) -> str:
        """分析页面类型和关键信息（辅助大模型判断）"""
        info_parts = []
        
        # 检测案号/文号
        case_number = self._find_case_number(full_text)
        if case_number:
            info_parts.append(f"案号/文号: {case_number}")
        
        # 检测机构落款
        institution = self._find_institution(full_text)
        if institution:
            info_parts.append(f"机构: {institution}")
        
        # 检测页面类型
        page_type = self._detect_page_type_from_text(texts, full_text)
        if page_type:
            info_parts.append(f"页面类型: {page_type}")
        
        # 检测变体（正本/副本等）
        variant = self._find_variant(header_text + " " + full_text)
        if variant:
            info_parts.append(f"变体: {variant}")
        
        return "; ".join(info_parts) if info_parts else "正常文书"
    
    def _find_case_number(self, text: str) -> Optional[str]:
        """查找案号/文号"""
        patterns = [
            r'\(\d{4}\)\w+刑\w+\d+号',      # (2024)粤0304刑初1142号
            r'\(\d{4}\)\w+民\w+\d+号',      # (2024)粤0304民初13365号
            r'\w+检刑诉\[\d{4}\]\d+号',      # 深福检刑诉[2024]1089号
            r'\w+所法字\[\d{4}\]第\d+号',    # 广深所法字[2024]第106号
            r'\w+公\(\w+\)不立字\(\d{4}\)\d+号',  # 深福公（莲花）不立字（2025）83号
            r'\w+\[\d{4}\]\d+号',            # 其他格式
            r'第\d+号',                      # 简单编号
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def _find_institution(self, text: str) -> Optional[str]:
        """提取机构/单位名称（通用方法）"""
        # 通用的机构名称模式（适用于各种行业）
        patterns = [
            r'(\w+公司)',           # 公司
            r'(\w+有限公司)',        # 有限公司
            r'(\w+股份公司)',        # 股份公司
            r'(\w+集团)',            # 集团
            r'(\w+局)',              # 局（如建设局、教育局等）
            r'(\w+委员会)',          # 委员会
            r'(\w+办公室)',          # 办公室
            r'(\w+中心)',            # 中心
            r'(\w+所)',              # 所（如研究所、鉴定所等）
            r'(\w+院)',              # 院（如医院、学院、法院等）
            r'(\w+处)',              # 处
            r'(\w+科)',              # 科
            r'(\w+部)',              # 部
            r'(\w+协会)',            # 协会
            r'(\w+学会)',            # 学会
            r'(\w+基金会)',          # 基金会
            r'(\w+事务所)',          # 事务所（通用）
            r'(\w+企业)',            # 企业
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _find_variant(self, text: str) -> Optional[str]:
        """查找变体（正本/副本等）"""
        variants = ['正本', '副本', '图本', '原本', '抄本']
        for variant in variants:
            if variant in text:
                return variant
        return None
    
    def _detect_academic_page_type(self, texts: List[Dict], page_height: float, header_text: str, key_text: str) -> str:
        """检测学术论文页面类型（针对论文拆分优化）"""
        if not texts:
            return "空白页"
        
        # 检测标题页特征（学术论文）
        # 标题页通常：顶部区域有较长文本，可能是标题
        top_region_texts = [t for t in texts if t.get('y', 0) <= page_height * 0.30]
        
        if len(top_region_texts) >= 2:
            # 顶部区域有多个文本块，可能是标题页
            top_text = " ".join([t.get('text', '') for t in top_region_texts[:3]])
            if len(top_text) > 30:  # 标题通常较长
                # 检查是否包含标题页关键词（使用通用配置）
                title_keywords = self.PAGE_TYPE_KEYWORDS.get('title_page', [])
                if any(kw.lower() in top_text.lower() or kw in top_text for kw in title_keywords):
                    return "论文标题页"
                # 或者顶部文本很长，可能是标题
                if len(top_text) > 50:
                    return "可能的标题页"
        
        # 检测目录页（使用通用配置）
        toc_keywords = self.PAGE_TYPE_KEYWORDS.get('toc', [])
        if any(kw.lower() in key_text.lower() or kw in key_text for kw in toc_keywords):
            return "目录页"
        
        # 检测摘要页（使用通用配置）
        abstract_keywords = self.PAGE_TYPE_KEYWORDS.get('abstract', [])
        if any(kw.lower() in key_text.lower() or kw in key_text for kw in abstract_keywords):
            return "摘要页"
        
        # 检测参考文献页（使用通用配置）
        ref_keywords = self.PAGE_TYPE_KEYWORDS.get('references', [])
        if any(kw.lower() in key_text.lower() or kw in key_text for kw in ref_keywords):
            return "参考文献页"
        
        if len(key_text) < 20:
            return "内容较少"
        
        return ""  # 正常页面
    
    def _detect_page_type_from_text(self, texts: List[Dict], full_text: str) -> Optional[str]:
        """检测页面类型（保留原有逻辑）"""
        if not texts or not full_text.strip():
            return "图片/空白"
        
        text_count = len(texts)
        text_length = len(full_text)
        
        # 正式文书
        if text_count > 10 or text_length > 200:
            return None  # 正常文书，不需要特殊标记
        
        # 检查聊天记录特征
        has_chat_time = bool(re.search(r'(上午|下午)\s*\d{1,2}:\d{2}', full_text))
        has_chat_pattern = '@' in full_text
        has_wechat_chat = '微信' in full_text and has_chat_time
        
        if (has_chat_time and has_chat_pattern) or has_wechat_chat:
            return "聊天记录"
        if has_chat_time and text_count <= 3:
            return "聊天记录"
        
        # 证据材料
        if text_count <= 5 and text_length < 50:
            return "证据材料"
        
        return None
    
    def _build_header_text_from_ocr_json(self, texts: List[Dict], page_height: float, ratio: float = 0.25) -> str:
        """从OCR JSON的texts中提取页眉文本（优化版：适配新的简化JSON结构）"""
        if not texts:
            return ""
        
        try:
            threshold = float(page_height) * float(ratio) if page_height else 0.0
        except Exception:
            threshold = 0.0
        
        header_items = []
        for t in texts:
            try:
                y = float(t.get("y", 0))
            except Exception:
                y = 0.0
            if threshold <= 0 or y <= threshold:
                header_items.append(t)
        
        # 兜底：如果顶部区域太"空"，取全页"最靠上"的若干块
        if len(header_items) < 3:
            tmp = list(texts)
            tmp.sort(key=lambda x: x.get("y", 0))  # 现在只有y坐标，不需要x
            header_items = tmp[:15]
        
        header_items.sort(key=lambda x: x.get("y", 0))
        header_text = " ".join([str(x.get("text", "")).strip() for x in header_items if str(x.get("text", "")).strip()])
        # 去除多余空白
        header_text = re.sub(r'\s+', ' ', header_text).strip()
        return header_text[:600]
    
    def _init_llm_client(self):
        """延迟初始化LLM客户端（只有在需要时才初始化）"""
        if self.client is not None:
            return
        
        if not self._api_key:
            raise ValueError("未设置LLM_API_KEY或DEEPSEEK_API_KEY环境变量，无法调用大模型。请先设置API密钥，或在.env文件中配置。")
        
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base_url,
        )
        logger.info("LLM客户端初始化成功")
    
    def split_with_llm(self, page_texts: List[Tuple[int, str, str, str]]) -> List[DocumentSplit]:
        """
        使用大模型进行文档拆分和打标（增强版，包含后处理验证）
        
        Args:
            page_texts: [(页码, 页眉文本, 全页文本, 页面类型和关键信息), ...]
        
        Returns:
            拆分结果列表
        """
        # 初始化LLM客户端
        self._init_llm_client()
        
        logger.info("开始调用大模型进行文档拆分和打标...")
        
        # 构建提示词
        system_prompt = self._build_system_prompt()
        
        # 构建用户输入（包含所有页面的文本）
        user_prompt = self._build_user_prompt(page_texts)
        
        # 调用大模型
        try:
            logger.info("正在调用大模型...")
            # 模型名称：优先使用环境变量，否则使用默认值
            model_name = os.getenv("LLM_MODEL_NAME") or "deepseek-chat"
            
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # 降低温度，提高一致性
            )
            
            content = completion.choices[0].message.content
            logger.info("大模型返回结果")
            
            # 解析JSON结果
            splits = self._parse_llm_response(content)
            
            logger.info(f"大模型识别到 {len(splits)} 个文档")
            
            # 后处理：验证和修正拆分结果
            splits = self._post_process_splits(splits, page_texts)
            
            logger.info(f"后处理完成，最终识别到 {len(splits)} 个文档")
            return splits
            
        except Exception as e:
            logger.error(f"调用大模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _post_process_splits(self, splits: List[DocumentSplit], page_texts: List[Tuple[int, str, str, str]]) -> List[DocumentSplit]:
        """后处理：验证和修正拆分结果"""
        if not splits:
            return splits
        
        # 按起始页排序
        splits.sort(key=lambda x: x.start_page)
        
        # 1. 确保第一个文档从第1页开始
        if splits[0].start_page > 1:
            logger.warning(f"第一个文档从第{splits[0].start_page}页开始，修正为第1页")
            splits[0].start_page = 1
        
        # 2. 确保最后一个文档到PDF末尾结束
        # 使用实际PDF的页数，而不是page_texts的长度（可能包含重复或无效页面）
        total_pages = len(self.doc)
        if splits[-1].end_page < total_pages:
            logger.warning(f"最后一个文档到第{splits[-1].end_page}页结束，修正为第{total_pages}页")
            splits[-1].end_page = total_pages
        elif splits[-1].end_page > total_pages:
            logger.warning(f"最后一个文档到第{splits[-1].end_page}页结束，但PDF只有{total_pages}页，修正为第{total_pages}页")
            splits[-1].end_page = total_pages
        
        # 3. 修正重叠和间隙
        corrected_splits = []
        for i, split in enumerate(splits):
            if i == 0:
                # 第一个文档
                if split.start_page < 1:
                    split.start_page = 1
                corrected_splits.append(split)
            else:
                # 后续文档：确保连续且不重叠
                prev_split = corrected_splits[-1]
                if split.start_page <= prev_split.end_page:
                    # 有重叠，调整当前文档的起始页
                    new_start = prev_split.end_page + 1
                    logger.warning(f"文档重叠检测：第{i+1}个文档起始页从{split.start_page}调整为{new_start}")
                    split.start_page = new_start
                
                if split.start_page > split.end_page:
                    # 调整后无效，跳过
                    logger.warning(f"跳过无效的拆分结果：{split.title} (第{split.start_page}-{split.end_page}页)")
                    continue
                
                corrected_splits.append(split)
        
        # 4. 保守的合并策略：只合并明确的补充内容，避免错误拆分
        # 注意：为了不错误拆分，我们采用非常保守的策略，只合并非常明确的补充材料
        if self.document_type in ("academic", "general"):
            final_splits = []
            for i, split in enumerate(corrected_splits):
                title_lower = split.title.lower()
                pages_count = split.end_page - split.start_page + 1
                
                # 通用方法：检查是否是补充材料（使用配置的关键词）
                should_merge = False
                
                # 检查是否以补充材料关键词开头
                for category, keywords in self.SUPPLEMENT_KEYWORDS.items():
                    for keyword in keywords:
                        if title_lower.startswith(keyword.lower()):
                            # 如果是明确的补充材料，且页数很少（<3页），考虑合并
                            if pages_count < 3 and pages_count > 0:
                                should_merge = True
                                break
                    if should_merge:
                        break
                
                # 特殊处理：Appendix A, Appendix B 等格式
                if not should_merge:
                    import re
                    if re.match(r'^appendix\s+[a-z]', title_lower):
                        if pages_count < 3 and pages_count > 0:
                            should_merge = True
                
                if should_merge and final_splits:
                    prev_split = final_splits[-1]
                    logger.info(f"检测到明确的补充内容：{split.title}（{pages_count}页），合并到前面的文档：{prev_split.title}")
                    prev_split.end_page = split.end_page
                else:
                    # 不满足合并条件，保留为独立文档
                    final_splits.append(split)
            
            corrected_splits = final_splits
        
        # 5. 确保最后一个文档覆盖到末尾
        if corrected_splits:
            total_pages = len(self.doc)  # 使用实际PDF页数
            if corrected_splits[-1].end_page < total_pages:
                logger.warning(f"最后一个文档未覆盖到末尾，修正为第{total_pages}页")
                corrected_splits[-1].end_page = total_pages
            elif corrected_splits[-1].end_page > total_pages:
                logger.warning(f"最后一个文档超出PDF页数，修正为第{total_pages}页")
                corrected_splits[-1].end_page = total_pages
        
        # 6. 验证页面覆盖完整性
        covered = set()
        for split in corrected_splits:
            for page_num in range(split.start_page, split.end_page + 1):
                covered.add(page_num)
        
        total_pages = len(self.doc)
        missing_pages = set(range(1, total_pages + 1)) - covered
        if missing_pages:
            logger.warning(f"检测到缺失的页面: {sorted(missing_pages)}")
            # 将缺失的页面添加到最后一个文档
            if corrected_splits:
                last_split = corrected_splits[-1]
                max_missing = max(missing_pages)
                if max_missing > last_split.end_page:
                    logger.info(f"将缺失的页面添加到最后一个文档：第{last_split.end_page+1}-{max_missing}页")
                    last_split.end_page = max_missing
        
        # 7. 最终验证：确保没有错误拆分
        # 检查是否有明显的错误合并（比如一个文档包含了多个明显不同的标题）
        # 这里我们采用保守策略：不主动拆分，只验证
        logger.info(f"最终拆分结果：{len(corrected_splits)}个文档，覆盖{total_pages}页")
        
        return corrected_splits
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词（通用版本，适用于各种文档类型）"""
        # 根据文档类型调整提示词
        if self.document_type == "legal":
            doc_type_desc = "法律文书"
            doc_type_notes = """
- 注意识别法律文书特有的格式：案号、文号、机构落款等
- 区分正本和副本，应拆分为不同文档
- 注意识别"及"、"和"连接的多个文档，必须拆分
"""
        elif self.document_type == "academic":
            doc_type_desc = "学术文档"
            doc_type_notes = """
- 注意识别论文、报告、章节标题等
- 通常以标题页、摘要、目录、正文等为边界
- **论文标题页特征**：页面顶部（前30%区域）有较长的标题文本，通常包含论文标题
- **新论文开始**：当页面顶部出现新的长标题文本时，通常是新论文的开始
- **论文边界**：每篇论文通常从标题页开始，到参考文献或下一论文标题页结束
"""
        else:
            doc_type_desc = "通用文档"
            doc_type_notes = """
- 识别文档标题、封面、目录等作为拆分边界
- 注意区分不同文档的起始和结束
- **对于学术论文**：注意识别标题页（页面顶部有较长标题文本），每篇论文通常从标题页开始
- **标题页特征**：页面顶部区域（前30%）有较长的文本，可能是论文标题
- **论文边界**：当页面类型显示为"论文标题页"或"可能的标题页"时，通常是新论文的开始
"""
        
        return f"""你是一个专业的PDF文档智能拆分助手，用于将合并的PDF文档拆分为多个独立文档。

## 任务说明
你需要分析OCR识别出的PDF页面文本，识别出每个独立文档的起始和结束页码，**只返回切点信息**。

## 输出格式要求（严格）
你必须返回严格的JSON格式，不要任何解释、不要markdown代码块标记、不要额外文字。

**输出格式示例：**
{{
  "splits": [
    {{
      "start_page": 1,
      "end_page": 5,
      "title": "文档标题1"
    }},
    {{
      "start_page": 6,
      "end_page": 10,
      "title": "文档标题2"
    }}
  ]
}}

**重要：只返回JSON对象，不要任何其他内容！**

## 拆分规则（核心原则）

### 1. 文档边界识别（最重要）

#### 1.1 文档起始识别
以下情况通常表示新文档的开始：
- **标题页**：页面顶部（前25%区域）出现明显的文档标题
- **封面页**：包含文档标题、作者、日期等信息的独立页面
- **目录页后**：目录页之后通常是一个新文档的开始
- **格式变化**：页面格式、字体、布局发生明显变化
- **编号/序号**：出现新的文档编号、序号或标识符
- **机构/单位标识**：出现新的机构名称、单位标识等

#### 1.2 学术论文特殊规则（重要！）
**对于学术论文，必须特别注意：**
- **论文标题页识别**：
  * 当页面类型显示为"论文标题页"或"可能的标题页"时，**通常是新论文的开始**
  * 页面顶部（前30%区域）有较长的标题文本（通常>30字符），可能是论文标题
  * 标题文本可能包含论文相关关键词，如"Abstract"、"Introduction"、"Author"、"University"等
- **论文边界判断**：
  * **每篇论文通常从标题页开始**，当遇到新的"论文标题页"时，上一篇论文结束
  * 如果页面类型显示为"论文标题页"，该页通常是新论文的第1页
  * 论文通常包含：标题页、摘要、正文、参考文献等部分
- **不应单独拆分的内容（关键！）**：
  * **附录（Appendix）**：标题以"Appendix"、"附录"开头的内容，应该属于前面的论文，**不应单独拆分**
  * **参考文献（References/Bibliography）**：参考文献部分应该属于前面的论文，**不应单独拆分**
  * **补充材料（Supplementary Material）**：补充材料应该属于前面的论文，**不应单独拆分**
  * **规则**：如果标题以"Appendix"、"References"、"Bibliography"、"Supplementary"、"附录"、"参考文献"等开头，且页数较少（<10页），应该合并到前面的文档
- **异常情况处理**：
  * 如果某个文档只有1-2页，可能是识别错误，应该检查前后页面，可能应该合并到相邻文档
  * 如果某个文档页数异常多（如>100页），可能包含了多篇论文，需要进一步拆分
  * **特别注意**：如果第3个文档只有2页（如18-19页），很可能是识别错误，应该检查第17页和第20页的内容

#### 1.2 位置限制（关键）
- **文档标题通常出现在页眉区域（页面顶部25%区域，即页眉文本的前50个字符内）**
- 如果标题出现在正文区域（页眉文本50个字符之后），需要谨慎判断：
  - 如果是明显的文档标题（如"第X章"、"报告"、"通知"等），可能是新文档
  - 如果是正文中的引用或章节标题，通常不是新文档

#### 1.3 文档结束识别
- 下一个文档的起始页-1
- 最后一个文档到PDF末尾结束
- 如果文档有明确的结束标记（如"此致"、"特此"、"附："、"年月日"落款、页码结束等），通常表示文档结束

### 2. 拆分策略

#### 2.1 必须拆分的情况
- **不同文档类型**（如报告和合同必须分开）
- **同一类型但不同编号/标识**（如两个不同的报告）
- **同一类型但不同版本**（如正本和副本，必须分开）
- **独立的附件或清单**（如果明显是独立文档）
- **"及"、"和"、"与"连接的多个文档**（必须拆分）：
  - 如"合同及附件" → 应拆分为"合同"和"附件"两个文档
  - 如"报告及附录" → 应拆分为"报告"和"附录"两个文档
- **同一页面上出现多个独立文档标题时，必须拆分**（即使在同一页，也要识别出不同的起始页）

#### 2.2 可以合并的情况（谨慎使用，避免错误拆分）
- 连续的同类文档（如果编号相同且无明显分隔）→ **但如果有明确的标题分隔，仍应拆分**
- 同一文档的多个副本（如果不需要区分）→ **但正本和副本必须分开**
- 附件和主文档（如果它们是一个整体）→ **但如果是独立的附件，应单独拆分**

**重要原则：宁可多拆分，不要错误合并！**
- 如果无法确定两个内容是否应该合并，**应该拆分为两个独立的文档**
- 每个有明确标题的文档都应该独立拆分，即使页数很少（1-2页）
- 只有非常明确的补充材料（如"Appendix A"、"参考文献"等）才考虑合并

### 3. 标题提取（重要！）
- **必须提取文档的完整标题**，不要使用"可能的标题页"、"文档"等通用描述
- 优先从页面顶部（前3行文本）提取标题
- 如果页面顶部有明确的文档标题，必须使用该标题
- 如果文档有编号/标识，可以包含在title中
- 如果无法提取明确标题，使用页眉中的关键信息，但尽量避免使用"可能的标题页"等描述
- 标题应简洁明了，便于识别，反映文档的实际内容

### 4. 页码要求（严格）
- **第一个文档必须从第1页开始**
- **最后一个文档必须到PDF末尾结束**
- **页码范围必须连续且不重叠**（下一个文档的start_page = 上一个文档的end_page + 1）
- **必须覆盖所有页面，不能遗漏任何一页**

### 5. 页面类型识别（辅助判断）
- **封面页**：通常包含标题、作者、日期等信息，文字较少
- **目录页**：包含目录条目和页码
- **正文页**：包含大量文字内容
- **图片/图表页**：文字很少，主要是图片或图表
- **空白页**：几乎没有内容
- **论文标题页**：页面顶部有较长的标题文本，通常>30字符，可能包含论文相关关键词
- **可能的标题页**：页面顶部有较长文本，可能是论文标题

### 6. 学术论文拆分特别说明（重要！）
**当处理学术论文时，必须遵循以下规则：**
1. **论文标题页识别**：
   - 当页面类型显示为"论文标题页"或"可能的标题页"时，**该页通常是新论文的第1页**
   - 页面顶部（前30%区域）有较长的标题文本（通常>30字符）
   - 标题文本可能包含论文相关关键词，如"Abstract"、"Introduction"、"Author"、"University"等
2. **论文边界判断**：
   - **每篇论文通常从标题页开始**
   - 当遇到新的"论文标题页"时，上一篇论文结束于该页的前一页
   - 例如：如果第18页是"论文标题页"，那么第17页是上一篇论文的最后一页
3. **不应单独拆分的内容（关键！）**：
   - **附录（Appendix）**：如果标题以"Appendix"、"附录"开头，且页数较少（<10页），**应该合并到前面的论文**，不应单独拆分
   - **参考文献（References/Bibliography）**：参考文献部分应该属于前面的论文，**不应单独拆分**
   - **补充材料（Supplementary Material）**：补充材料应该属于前面的论文，**不应单独拆分**
   - **判断规则**：如果标题以"Appendix"、"References"、"Bibliography"、"Supplementary"、"附录"、"参考文献"等开头，且页数<10页，应该合并到前面的文档
4. **异常情况处理**：
   - 如果某个文档只有1-2页，很可能是识别错误，应该检查前后页面，可能需要合并
   - 如果某个文档页数异常多（如>100页），可能包含了多篇论文，需要进一步拆分
   - **特别注意**：如果第3个文档只有2页（如18-19页），很可能是识别错误，应该检查第17页和第20页的内容，可能需要调整拆分点

## 文档类型特定说明
当前处理的文档类型：{doc_type_desc}
{doc_type_notes}

## 注意事项
1. **严格遵循输出格式，只返回JSON，不要任何其他文字**
2. **确保页码范围连续且不重叠**
3. **必须覆盖所有页面**（从第1页到最后一页）
4. **必须细致拆分**（"及"、"和"、"与"连接的多个文档必须拆分，不要合并）
5. **不要遗漏页面**（每个页面都必须属于某个文档）
6. **标题要准确**（提取文档的真实标题，便于后续识别，不要使用"及"连接的复合标题）
7. **位置限制必须遵守**（标题通常在页眉区域）
8. **同一页出现多个独立文档时，必须识别并拆分**（即使在同一页，也要识别出不同的起始页）

## 禁止行为（严格执行）
- **绝对禁止**输出markdown代码块标记（如```json或```）
- **绝对禁止**输出任何解释性文字、注释、说明
- **绝对禁止**遗漏页面（必须覆盖所有页面）
- **绝对禁止**创建重叠的页码范围
- **绝对禁止**跳过页面（每个页面都必须有归属）
- **绝对禁止**将正文中的章节标题误判为独立文档
- **必须**确保第一个文档从第1页开始
- **必须**确保最后一个文档到文档末尾结束
- **必须**确保页码范围连续（下一个文档的start_page = 上一个文档的end_page + 1）

## 输出要求（再次强调）
**只返回纯JSON对象，格式如下，不要任何其他内容：**
{{"splits": [{{"start_page": 1, "end_page": 5, "title": "文档标题"}}]}}"""
    
    def _build_user_prompt(self, page_texts: List[Tuple[int, str, str, str]]) -> str:
        """构建用户提示词（极简优化版：只保留页眉和前3行文本，大幅减少token）"""
        total_pages = len(page_texts)
        
        doc_type_desc = {
            "legal": "法律文书",
            "academic": "学术文档",
            "general": "通用文档"
        }.get(self.document_type, "通用文档")
        
        prompt_parts = [
            f"分析PDF文档，识别每个独立的{doc_type_desc}并拆分。",
            f"总页数：{total_pages}",
            "格式：页码|页眉|前3行文本|页面类型",
            ""
        ]
        
        # 极简格式：只保留页眉和前3行文本，大幅减少token
        for page_num, header_text, key_text, page_info in page_texts:
            # 页眉文本限制为100字符
            header_display = header_text[:100] if len(header_text) > 100 else header_text
            
            # 关键文本（前3行）限制为150字符
            if len(key_text) > 150:
                truncated_key = key_text[:150]
            else:
                truncated_key = key_text
            
            # 使用极简格式：页码|页眉|前3行文本|页面类型
            # 页面类型信息对识别论文边界很重要
            page_line = f"{page_num}|{header_display}|{truncated_key}"
            if page_info:
                page_line += f"|{page_info}"
            prompt_parts.append(page_line)
        
        prompt_parts.append(f"返回JSON拆分结果。必须覆盖所有页面（1-{total_pages}页）。")
        
        prompt_text = "\n".join(prompt_parts)
        
        # 检查token数量，如果仍然太长，进一步压缩
        estimated_tokens = len(prompt_text) * 1.8
        if estimated_tokens > 100000:
            logger.warning(f"提示词仍然较长（估计{estimated_tokens:.0f} tokens），进一步压缩...")
            # 进一步压缩：页眉限制为50字符，关键文本限制为80字符
            prompt_parts = [
                f"分析PDF文档，识别每个独立的{doc_type_desc}并拆分。",
                f"总页数：{total_pages}",
                "格式：页码|页眉|前3行文本",
                ""
            ]
            
            for page_num, header_text, key_text, page_info in page_texts:
                header_display = header_text[:50] if len(header_text) > 50 else header_text
                truncated_key = key_text[:80] if len(key_text) > 80 else key_text
                page_line = f"{page_num}|{header_display}|{truncated_key}"
                prompt_parts.append(page_line)
            
            prompt_parts.append(f"返回JSON拆分结果。必须覆盖所有页面（1-{total_pages}页）。")
            prompt_text = "\n".join(prompt_parts)
            logger.info(f"压缩后提示词长度：{len(prompt_text)} 字符（估计 {len(prompt_text) * 1.8:.0f} tokens）")
        
        return prompt_text
    
    def _parse_llm_response(self, content: str) -> List[DocumentSplit]:
        """解析大模型返回的JSON结果（只包含切点）"""
        try:
            # 移除可能的markdown代码块标记和多余文字
            content = content.strip()
            
            # 移除markdown代码块标记
            if content.startswith("```"):
                lines = content.split("\n")
                # 移除第一行（```json或```）
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                # 移除最后一行（```）
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            # 尝试提取JSON部分（如果包含其他文字）
            # 查找第一个{和最后一个}
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]
            
            # 解析JSON
            data = json.loads(content)
            
            splits = []
            for item in data.get("splits", []):
                try:
                    start_page = int(item.get("start_page", 0))
                    end_page = int(item.get("end_page", 0))
                    title = str(item.get("title", "")).strip()
                    
                    # 验证页码有效性（只要页码有效就保留，不因标题问题跳过）
                    if start_page > 0 and end_page >= start_page:
                        split = DocumentSplit(
                            start_page=start_page,
                            end_page=end_page,
                            title=title if title else f"文档_{start_page}-{end_page}"
                        )
                        splits.append(split)
                    else:
                        logger.warning(f"跳过无效页码的拆分结果: start_page={start_page}, end_page={end_page}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"解析拆分结果项失败: {item}, 错误: {e}")
                    continue
            
            return splits
            
        except json.JSONDecodeError as e:
            logger.error(f"解析大模型返回的JSON失败: {e}")
            logger.error(f"返回内容: {content[:500]}")
            raise
        except Exception as e:
            logger.error(f"解析拆分结果失败: {e}")
            raise

    def split_pdf(self, splits: List[DocumentSplit]):
        """根据拆分结果拆分PDF"""
        logger.info("开始拆分PDF...")
        
        created_files = 0
        
        enumerated_splits = list(enumerate(splits, 1))
        if TQDM_AVAILABLE:
            progress_bar = tqdm(enumerated_splits, desc="拆分PDF", unit="个")
        else:
            progress_bar = enumerated_splits
        
        for seq_num, split in progress_bar:
            try:
                # 生成文件名（使用序号和文档标题）
                safe_title = "".join(c for c in split.title if c.isalnum() or c in (' ', '-', '_', '(', ')', '（', '）'))[:50]
                safe_title = safe_title.strip().replace(' ', '_')
                filename = f"{seq_num:02d}_{safe_title}.pdf"
                output_path = self.output_dir / filename
                
                # 创建新的PDF
                new_pdf = fitz.open()
                
                # 添加页面
                pages_added = 0
                for page_num in range(split.start_page, split.end_page + 1):
                    if page_num <= self.total_pages:
                        new_pdf.insert_pdf(self.doc, from_page=page_num - 1, to_page=page_num - 1)
                        pages_added += 1
                
                # 保存文件
                if pages_added > 0:
                    new_pdf.save(str(output_path), garbage=4, deflate=True, clean=True)
                    created_files += 1
                    
                    # 验证文件
                    saved_doc = fitz.open(str(output_path))
                    saved_pages = saved_doc.page_count
                    saved_doc.close()
                    
                    progress_desc = f"{filename}({saved_pages}页)"
                    if TQDM_AVAILABLE:
                        progress_bar.set_postfix_str(progress_desc)
                    else:
                        logger.info(f"  已创建: {progress_desc} - {split.title[:50]}")
                else:
                    logger.warning(f"  无页面可添加: {filename}")
                
                new_pdf.close()
                
            except Exception as e:
                logger.error(f"  ✗ 拆分失败 [{split.title[:30]}]: {e}")
        
        logger.info(f"拆分完成，共生成{created_files}个PDF文件")
    
    def _detect_table_of_contents(self, text_blocks: List[TextBlock], page_num: int) -> bool:
        """检测是否为目录页（改进版：支持目录延续页）"""
        if not text_blocks:
            return False
        
        full_text = " ".join([block.text.strip() for block in text_blocks])
        
        # 检查是否包含目录关键词
        toc_keywords = [
            '目录', '目 录', 'CONTENTS', '目    录', '卷内目录', '卷宗目录',
            '证据目录', '证据材料及说明', '证据材料', '证据清单', '证据说明',
            '证据材料与说明', '材料与说明', '材料说明', '证据与说明',
            '文书目录', '卷内文书目录', '预审卷宗封面及卷内文书目录'
        ]
        has_toc_keyword = any(keyword in full_text for keyword in toc_keywords)
        
        # 检查是否有目录条目格式：序号+标题+页码（支持页码范围）
        toc_entry_patterns = [
            r'\d+[\.、]\s*[^\d]{2,30}\s+\d+',  # 1. 标题 页码
            r'\d+[\.、]\s*[^\d]{2,30}\s+\d+\s*[-~]\s*\d+',  # 1. 标题 页码-页码（页码范围）
            r'[一二三四五六七八九十]+[、．]\s*[^\d]{2,30}\s+\d+',  # 一、标题 页码
            r'[（(]\d+[）)]\s*[^\d]{2,30}\s+\d+',  # (1)标题 页码
            r'证据\d+[：:]',  # 证据1：、证据二：等
            r'^\d+\s+[^\d]{2,30}\s+\d+',  # 1 标题 页码（无标点）
            r'^\d+\s+[^\d]{2,30}\s+\d+\s*[-~]\s*\d+',  # 1 标题 页码-页码（无标点，页码范围）
        ]
        has_toc_entry_format = any(re.search(pattern, full_text) for pattern in toc_entry_patterns)
        
        # 检查是否包含页码模式（支持页码范围）
        page_patterns = [
            r'\d+',  # 纯数字
            r'\d+\s*[-~]\s*\d+',  # 页码范围：1-4、5-9等
            r'第\d+页',
            r'P\d+',
            r'p\d+',
            r'第\d+张'
        ]
        has_page_numbers = any(re.search(pattern, full_text) for pattern in page_patterns)
        
        # 检查文本块排列（目录通常有标题和页码，页码通常在右侧）
        has_table_format = False
        if len(text_blocks) >= 5:
            sorted_blocks = sorted(text_blocks, key=lambda b: b.x)
            # 检查是否有明显的左右分布（页码在右侧）
            right_side_blocks = [b for b in sorted_blocks if b.x > 0.6]  # 右侧60%区域
            left_side_blocks = [b for b in sorted_blocks if b.x < 0.3]  # 左侧30%区域
            
            # 右侧主要是数字（页码）或页码范围，左侧有序号或标题
            right_numbers = sum(1 for b in right_side_blocks if (
                re.match(r'^\d+$', b.text.strip()) or 
                re.match(r'^\d+\s*[-~]\s*\d+', b.text.strip())  # 支持页码范围
            ))
            left_has_sequence = any(re.match(r'^\d+$', b.text.strip()) for b in left_side_blocks)
            
            if right_numbers >= 2 and (left_has_sequence or has_toc_entry_format):
                has_table_format = True
        
        # 如果有目录关键词+页码，或者有目录格式+页码，或者有表格格式，都认为是目录页
        return (has_toc_keyword and has_page_numbers) or (has_toc_entry_format and has_page_numbers) or (has_table_format and has_page_numbers)
    
    def _extract_toc_entries(self, text_blocks: List[TextBlock], page_height: float) -> List[Dict[str, any]]:
        """从目录页提取目录条目（优化版）"""
        entries = []
        if not text_blocks:
            return entries
        
        # 按Y坐标排序（从上到下）
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.y, b.x))
        
        # 过滤掉明显的表头和页脚
        filtered_blocks = []
        skip_keywords = [
            '目录', '目 录', '序号', '名称', '页码', '页数', '页号', 'CONTENTS',
            '证据目录', '证据材料及说明', '证据材料', '证据清单', '证据说明',
            '证据材料与说明', '材料与说明', '材料说明', '证据与说明',
            '组别', '证明内容', '备注', '编号', 'NO', 'No', '提交人',
            '是否', '原件', '否', '是', '廨', '件', '文书名称', '文书标题',
            '预审卷宗封面及卷内文书目录', '卷内文书目录', '卷内', '文书目录',
            '责任者', '文号', '标题', '日期'
        ]
        
        for block in sorted_blocks:
            text = block.text.strip()
            if not text or len(text) < 1:
                continue
            # 跳过表头关键词
            if text in skip_keywords:
                continue
            if len(text) <= 3 and any(kw in text for kw in skip_keywords):
                continue
            filtered_blocks.append(block)
        
        # 按行分组（使用更小的容差，提高精度）
        lines = []
        current_line = []
        current_y = None
        
        # 计算平均行高
        if filtered_blocks:
            y_diffs = []
            sorted_y = sorted([b.y for b in filtered_blocks])
            for i in range(len(sorted_y) - 1):
                diff = sorted_y[i+1] - sorted_y[i]
                if 5 < diff < page_height * 0.1:  # 合理的行间距
                    y_diffs.append(diff)
            avg_line_height = sum(y_diffs) / len(y_diffs) if y_diffs else page_height * 0.03
            tolerance = max(avg_line_height * 0.5, page_height * 0.02)  # 使用更小的容差
        else:
            tolerance = page_height * 0.02
        
        for block in filtered_blocks:
            if current_y is None or abs(block.y - current_y) > tolerance:
                if current_line:
                    lines.append(current_line)
                current_line = [block]
                current_y = block.y
            else:
                current_line.append(block)
        
        if current_line:
            lines.append(current_line)
        
        # 处理每一行，提取标题和页码
        page_width = max([b.x + b.width for b in filtered_blocks]) if filtered_blocks else page_height * 1.4
        
        for line_blocks in lines:
            if not line_blocks:
                continue
            
            line_blocks.sort(key=lambda b: b.x)
            
            # 分析这一行的结构
            # 1. 识别序号列（最左侧，通常是单个数字）
            # 2. 识别标题列（中间区域）
            # 3. 识别页码列（最右侧，可能是单个数字或范围）
            
            seq_num = None
            seq_block_idx = None
            page_num = None
            page_block_indices = []
            title_blocks = []
            
            # 按X坐标分区
            for idx, block in enumerate(line_blocks):
                text = block.text.strip()
                if not text:
                    continue
                
                x_ratio = block.x / page_width
                
                # 左侧20%区域：可能是序号
                if x_ratio < 0.2:
                    if re.match(r'^\d+$', text) and len(text) <= 3:
                        if seq_num is None:  # 取第一个匹配的序号
                            seq_num = int(text)
                            seq_block_idx = idx
                
                # 右侧20%区域：可能是页码
                elif x_ratio > 0.8:
                    # 检查是否是页码范围（可能是分散的，如"3", "-", "5"或完整的"1-4"）
                    if re.match(r'^\d+$', text):
                        page_block_indices.append((idx, int(text), 'single'))
                    elif re.match(r'^[-~]$', text):
                        page_block_indices.append((idx, None, 'separator'))
                    elif re.match(r'^\d+\s*[-~]\s*\d+[A-Za-z]*$', text):
                        # 完整的页码范围（可能后面有字母，如"22-23F"，或简单的"1-4"、"5-9"）
                        match = re.match(r'^(\d+)\s*[-~]\s*(\d+)', text)
                        if match:
                            page_num = int(match.group(1))  # 使用起始页作为拆分点
                            page_block_indices = [(idx, page_num, 'range')]
                            break
                    elif re.match(r'^\d+\s*[-~]\s*[A-Za-z]?\d+$', text):
                        # 支持"1-4"、"5-9"、"10-13"等格式（OCR可能识别为"1 0 -13"）
                        match = re.match(r'^(\d+)\s*[-~]\s*(\d+)', text)
                        if match:
                            page_num = int(match.group(1))
                            page_block_indices = [(idx, page_num, 'range')]
                            break
                
                # 中间区域：标题
                else:
                    title_blocks.append((idx, block))
            
            # 处理分散的页码范围（如"3", "-", "5"）
            if not page_num and len(page_block_indices) >= 2:
                # 尝试组合分散的页码
                sorted_page_blocks = sorted(page_block_indices, key=lambda x: x[0])
                numbers = [x[1] for x in sorted_page_blocks if x[1] is not None]
                if len(numbers) >= 2:
                    # 如果有两个数字，可能是页码范围
                    page_num = numbers[0]
                elif len(numbers) == 1:
                    page_num = numbers[0]
            
            # 如果还没有页码，尝试从右侧提取（包括带字母的页码，如"22-23F"，或页码范围"1-4"）
            if not page_num:
                for idx, block in enumerate(line_blocks):
                    text = block.text.strip()
                    x_ratio = block.x / page_width
                    if x_ratio > 0.7:
                        # 尝试匹配页码范围（可能带字母，或简单的"1-4"、"5-9"）
                        match = re.match(r'^(\d+)\s*[-~]\s*\d+', text)
                        if match:
                            try:
                                page_num = int(match.group(1))  # 使用起始页
                                break
                            except:
                                pass
                        # 尝试匹配单个数字
                        elif re.match(r'^\d+$', text):
                            try:
                                page_num = int(text)
                                if page_num <= 100:  # 合理的页码范围
                                    break
                            except:
                                pass
            
            # 提取标题（参考pdf-split.py的逻辑，先尝试整行匹配）
            title_parts = []
            
            # 先尝试从整行文本中提取目录条目（更准确）
            line_text = " ".join([block.text.strip() for block in line_blocks])
            
            # 尝试匹配目录条目格式：序号.标题 页码 或 序号 标题 页码（支持页码范围）
            toc_line_patterns = [
                r'^(\d+)[\.、]\s*([^\d]+?)\s+(\d+)\s*[-~]\s*(\d+)$',  # 1. 标题 页码-页码（页码范围）
                r'^(\d+)[\.、]\s*([^\d]+?)\s+(\d+)$',  # 1. 标题 页码
                r'^(\d+)\s+([^\d]+?)\s+(\d+)\s*[-~]\s*(\d+)$',  # 1 标题 页码-页码（无标点，页码范围）
                r'^(\d+)\s+([^\d]+?)\s+(\d+)$',  # 1 标题 页码（无标点）
                r'^([一二三四五六七八九十]+)[、．]\s*([^\d]+?)\s+(\d+)$',  # 一、标题 页码
                r'^[（(](\d+)[）)]\s*([^\d]+?)\s+(\d+)$',  # (1)标题 页码
                r'证据\d+[：:]\s*([^\d]+?)\s+(\d+)',  # 证据1：标题 页码
            ]
            
            matched_title = None
            matched_page_num = None
            for pattern in toc_line_patterns:
                try:
                    match = re.search(pattern, line_text.strip())
                    if match:
                        # 提取标题和页码
                        groups = match.groups()
                        if len(groups) >= 2:
                            # 判断是否是页码范围格式（有4个组：序号、标题、起始页、结束页）
                            if len(groups) == 4:
                                # 页码范围格式：序号、标题、起始页、结束页
                                matched_title = groups[1].strip() if groups[1] else None
                                try:
                                    matched_page_num = int(groups[2])  # 使用起始页
                                except:
                                    pass
                            else:
                                # 单页码格式：序号、标题、页码（或只有标题、页码）
                                if len(groups) == 3:
                                    # 序号、标题、页码
                                    matched_title = groups[1].strip() if groups[1] else None
                                    try:
                                        matched_page_num = int(groups[2])
                                    except:
                                        pass
                                elif len(groups) == 2:
                                    # 标题、页码（无序号）
                                    matched_title = groups[0].strip() if groups[0] else None
                                    try:
                                        matched_page_num = int(groups[1])
                                    except:
                                        pass
                            
                            if matched_title:
                                # 如果从整行匹配到了页码，使用匹配到的页码
                                if matched_page_num:
                                    page_num = matched_page_num
                                break
                except Exception as e:
                    # 正则匹配失败，继续尝试下一个模式
                    continue
            
            # 如果从整行匹配到标题，直接使用
            if matched_title:
                title_parts = [matched_title]
            else:
                # 否则逐个块提取标题
                for idx, block in title_blocks:
                    text = block.text.strip()
                    if not text or len(text) < 1:
                        continue
                    
                    # 跳过页码相关的文本（包括带字母的，如"22-23F"）
                    if re.match(r'^\d+[-~]\d+[A-Za-z]*$', text) or (re.match(r'^\d+$', text) and len(text) <= 2):
                        continue
        
                    # 跳过表头关键词（完全匹配）- 通用关键词
                    skip_keywords = [
                        '序号', '名称', '页号', '页码', '页数', '标题', '日期', '备注',
                        '目录', '目 录', 'CONTENTS', '卷内目录', '卷宗目录'
                    ]
                    if text in skip_keywords:
                        continue
                    
                    title_parts.append(text)
            
            # 清理标题（通用清理，不针对特定领域）
            if title_parts:
                # 移除标题中的页码引用（如"3-5"、"20-21"、"22-23F"等）
                cleaned_parts = []
                for part in title_parts:
                    # 跳过看起来像页码的部分（包括带字母的）
                    if re.match(r'^\d+[-~]\d+[A-Za-z]*$', part) or (re.match(r'^\d+$', part) and len(part) <= 2):
                        continue
                    cleaned_parts.append(part)
                title_parts = cleaned_parts
            
            # 如果有页码，添加条目
            if page_num and page_num > 0:
                if title_parts:
                    title = " ".join(title_parts)
                    title = re.sub(r'\s+', ' ', title).strip()
                    
                    # 进一步清理标题（更保守的清理，避免删除有效内容）
                    title = re.sub(r'^[一二三四五六七八九十]+[、．]\s*', '', title)
                    title = re.sub(r'^\d+[\.、]\s*', '', title)
                    title = re.sub(r'^[（(]\d+[）)]\s*', '', title)
                    title = re.sub(r'\s*第\d+页\s*$', '', title)
                    title = re.sub(r'\s*P\d+\s*$', '', title)
                    title = re.sub(r'\s*第\d+-\d+页\s*$', '', title)
                    # 移除标题中残留的页码格式（包括带字母的，如"22-23F"）
                    title = re.sub(r'\s*\d+[-~]\d+[A-Za-z]*\s*', '', title)
                    title = re.sub(r'\s+\d{1,2}\s+', ' ', title)  # 移除孤立的1-2位数字（可能是页码）
                    # 清理多余的空白字符
                    title = re.sub(r'\s+', ' ', title)  # 多个空格合并为一个
                    title = title.strip()
                    
                    # 如果标题仍然有效
                    if title and len(title) > 1 and not re.match(r'^\d+$', title):
                        entries.append({'title': title, 'page': page_num, 'seq': seq_num})
                    elif seq_num:
                        # 使用序号作为标题
                        entries.append({'title': f'文档{seq_num}', 'page': page_num, 'seq': seq_num})
                    else:
                        entries.append({'title': f'文档{page_num}', 'page': page_num, 'seq': seq_num})
                elif seq_num:
                    # 有序号但没有标题
                    entries.append({'title': f'文档{seq_num}', 'page': page_num, 'seq': seq_num})
                else:
                    # 只有页码
                    entries.append({'title': f'文档{page_num}', 'page': page_num, 'seq': seq_num})
        
        # 去重和排序
        seen = set()
        unique_entries = []
        for entry in entries:
            key = (entry.get('title', ''), entry.get('page', 0))
            if key not in seen and entry.get('page', 0) > 0:
                seen.add(key)
                unique_entries.append(entry)
        
        # 按页码排序
        unique_entries.sort(key=lambda x: x.get('page', 0))
        
        return unique_entries
    
    def _split_from_toc(self, ocr_data: Dict) -> Optional[List[DocumentSplit]]:
        """基于目录进行拆分（优先方法）"""
        logger.info("尝试检测目录并基于目录拆分...")
        
        # 检查前50页是否有目录
        toc_entries = []
        toc_start_page = None
        
        for page_num in range(1, min(51, self.total_pages + 1)):
            if page_num in self.blank_pages:
                continue
            
            try:
                page = self.doc[page_num - 1]
                page_height = page.rect.height
                
                # OCR识别这一页
                text_blocks = self.ocr_page_batch([page_num]).get(page_num, [])
                
                # 检测是否为目录页
                if self._detect_table_of_contents(text_blocks, page_num):
                    if toc_start_page is None:
                        toc_start_page = page_num
                        logger.info(f"检测到目录页：第 {page_num} 页")
                    
                    # 提取目录条目
                    page_toc_entries = self._extract_toc_entries(text_blocks, page_height)
                    toc_entries.extend(page_toc_entries)
                    logger.info(f"从第 {page_num} 页提取到 {len(page_toc_entries)} 个目录条目")
                    
                    # 检查后续页面是否也是目录页（改进：更宽松的检测，支持目录延续页）
                    consecutive_no_toc = 0
                    max_check_pages = min(30, self.total_pages - page_num)  # 增加检查范围
                    
                    for next_page_num in range(page_num + 1, min(page_num + max_check_pages + 1, self.total_pages + 1)):
                        if next_page_num in self.blank_pages:
                            consecutive_no_toc += 1
                            if consecutive_no_toc >= 2:
                                break
                            continue
                        
                        try:
                            next_page = self.doc[next_page_num - 1]
                            next_page_height = next_page.rect.height
                            next_text_blocks = self.ocr_page_batch([next_page_num]).get(next_page_num, [])
                            
                            # 检查是否是目录页（标准检测）
                            is_toc_page = self._detect_table_of_contents(next_text_blocks, next_page_num)
                            
                            # 如果标准检测失败，尝试更宽松的检测（检查是否有目录格式特征）
                            if not is_toc_page:
                                full_text = " ".join([b.text.strip() for b in next_text_blocks])
                                
                                # 检查是否有目录条目格式：序号+标题+页码
                                toc_entry_patterns = [
                                    r'\d+[\.、]\s*[^\d]{2,30}\s+\d+',  # 1. 标题 页码
                                    r'[一二三四五六七八九十]+[、．]\s*[^\d]{2,30}\s+\d+',  # 一、标题 页码
                                    r'[（(]\d+[）)]\s*[^\d]{2,30}\s+\d+',  # (1)标题 页码
                                    r'证据\d+[：:]',  # 证据1：、证据二：等
                                ]
                                has_toc_entry_format = any(re.search(pattern, full_text) for pattern in toc_entry_patterns)
                                
                                # 检查是否有页码模式（右侧有数字）
                                sorted_blocks = sorted(next_text_blocks, key=lambda b: b.x)
                                right_blocks = [b for b in sorted_blocks if b.x > next_page_height * 0.6]
                                has_page_numbers = any(re.match(r'^\d+$', b.text.strip()) for b in right_blocks if b.text.strip())
                                
                                # 检查是否有序号列（左侧有数字序号）
                                left_blocks = [b for b in sorted_blocks if b.x < next_page_height * 0.3]
                                has_sequence_numbers = any(re.match(r'^\d+$', b.text.strip()) for b in left_blocks if b.text.strip())
                                
                                # 检查表格格式：左侧有序号，右侧有页码
                                left_nums = sum(1 for b in sorted_blocks[:len(sorted_blocks)//3] if re.match(r'^\d+$', b.text.strip()))
                                right_nums = sum(1 for b in sorted_blocks[-len(sorted_blocks)//3:] if re.match(r'^\d+$', b.text.strip()))
                                has_table_format = left_nums >= 2 and right_nums >= 2
                                
                                # 如果同时有目录格式和页码，或者是目录的延续页（有序号和页码），或者有表格格式，认为是目录页
                                if (has_toc_entry_format and has_page_numbers) or (has_sequence_numbers and has_page_numbers and len(next_text_blocks) >= 5) or (has_table_format and len(next_text_blocks) >= 5):
                                    is_toc_page = True
                                    if has_table_format:
                                        logger.info(f"通过表格格式检测识别为目录延续页：第 {next_page_num} 页")
                                    else:
                                        logger.info(f"通过宽松检测识别为目录延续页：第 {next_page_num} 页")
                            
                            if is_toc_page:
                                next_toc_entries = self._extract_toc_entries(next_text_blocks, next_page_height)
                                if next_toc_entries:
                                    toc_entries.extend(next_toc_entries)
                                    logger.info(f"从第 {next_page_num} 页提取到 {len(next_toc_entries)} 个目录条目")
                                    consecutive_no_toc = 0
                                else:
                                    consecutive_no_toc += 1
                                    # 如果检测为目录页但提取不到条目，可能是目录格式特殊，继续检查
                                    if consecutive_no_toc >= 5:  # 增加到5页，给更多机会
                                        break
                            else:
                                # 即使未检测为目录页，如果左侧有序号，也尝试提取（可能是目录延续页但页码格式特殊）
                                sorted_blocks = sorted(next_text_blocks, key=lambda b: b.x)
                                left_nums = sum(1 for b in sorted_blocks[:len(sorted_blocks)//3] if re.match(r'^\d+$', b.text.strip()))
                                # 如果左侧有序号且文本块较多，尝试提取
                                if left_nums >= 2 and len(next_text_blocks) >= 10:
                                    next_toc_entries = self._extract_toc_entries(next_text_blocks, next_page_height)
                                    if next_toc_entries:
                                        toc_entries.extend(next_toc_entries)
                                        logger.info(f"从第 {next_page_num} 页（未检测为目录但有序号）提取到 {len(next_toc_entries)} 个目录条目")
                                        consecutive_no_toc = 0
                                        continue
                                
                                consecutive_no_toc += 1
                                # 增加容忍度：连续3页未检测为目录页才停止（因为目录可能跨多页）
                                if consecutive_no_toc >= 3:
                                    break
                        except Exception as e:
                            logger.debug(f"检测第 {next_page_num} 页目录失败: {e}")
                            consecutive_no_toc += 1
                            if consecutive_no_toc >= 2:
                                break
                            continue
                    
                    if toc_start_page == page_num:
                        break
            except Exception as e:
                logger.debug(f"检测第 {page_num} 页目录失败: {e}")
                continue
        
        if not toc_entries:
            logger.info("未检测到目录，将使用大模型拆分")
            return None
        
        # 如果目录条目太少（少于3个），可能是误检，回退到LLM拆分
        if len(toc_entries) < 3:
            logger.info(f"目录条目太少（{len(toc_entries)}个），可能是误检，将使用大模型拆分")
            return None
        
        logger.info(f"从目录提取到 {len(toc_entries)} 个条目")
        
        # 根据目录条目创建拆分结果
        # 先对条目按页码排序
        sorted_entries = sorted(toc_entries, key=lambda x: x['page'])
        
        # 去重
        seen_keys = set()
        unique_entries = []
        for entry in sorted_entries:
            page = entry['page']
            title = entry['title']
            key = (page, title)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_entries.append(entry)
        
        logger.info(f"去重后共有 {len(unique_entries)} 个唯一条目")
        
        # 检查页码是否连续
        pages = [e['page'] for e in unique_entries]
        pages_sorted = sorted(pages)
        is_page_sequence_valid = True
        
        if len(pages_sorted) > 1:
            page_diffs = [pages_sorted[i+1] - pages_sorted[i] for i in range(len(pages_sorted)-1)]
            # 允许较大的页码差值（某些文档可能很长），但检查是否有明显不合理的页码
            # 如果页码超出总页数，或者有多个相邻页码差值都很大，则认为不准确
            max_reasonable_page = min(self.total_pages * 1.1, self.total_pages + 10)  # 允许10%的误差
            if any(p > max_reasonable_page for p in pages_sorted):
                is_page_sequence_valid = False
                logger.info(f"检测到目录页码超出合理范围（总页数{self.total_pages}），将按目录条目顺序拆分")
            elif len([d for d in page_diffs if d > 50]) > len(page_diffs) * 0.5:
                # 如果超过一半的页码差值都很大（>50），可能页码不准确
                is_page_sequence_valid = False
                logger.info("检测到目录页码可能不准确（多个大差值），将按目录条目顺序拆分")
            else:
                # 页码看起来合理，使用原始页码拆分
                logger.info(f"目录页码验证通过，将使用原始页码拆分（共{len(unique_entries)}个条目）")
        
        # 创建拆分结果
        splits = []
        if not is_page_sequence_valid:
            # 页码不准确，按顺序拆分
            toc_end_page = toc_start_page
            for page_num in range(toc_start_page, min(toc_start_page + 20, self.total_pages + 1)):
                if page_num in self.blank_pages:
                    continue
                try:
                    page = self.doc[page_num - 1]
                    text_blocks = self.ocr_page_batch([page_num]).get(page_num, [])
                    if self._detect_table_of_contents(text_blocks, page_num):
                        toc_end_page = page_num
                except:
                    pass
            
            current_start_page = toc_end_page + 1
            for i, entry in enumerate(unique_entries):
                title = entry['title']
                
                if i + 1 < len(unique_entries):
                    remaining_pages = self.total_pages - current_start_page + 1
                    remaining_entries = len(unique_entries) - i
                    avg_pages_per_entry = max(1, remaining_pages // remaining_entries)
                    end_page = min(current_start_page + avg_pages_per_entry - 1, self.total_pages)
                else:
                    end_page = self.total_pages
                
                start_page = current_start_page
                current_start_page = end_page + 1
                
                splits.append(DocumentSplit(
                    start_page=start_page,
                    end_page=end_page,
                    title=title
                ))
                logger.info(f"  目录条目{i+1}: {title} (第{start_page}-{end_page}页)")
        else:
            # 页码准确，使用原始页码拆分
            # 检查第一个条目之前是否有页面
            if unique_entries and unique_entries[0]['page'] > 1:
                first_entry_page = unique_entries[0]['page']
                # 添加第一个条目之前的页面
                splits.append(DocumentSplit(
                    start_page=1,
                    end_page=first_entry_page - 1,
                    title="目录前内容"
                ))
                logger.info(f"  目录前内容: 第1-{first_entry_page - 1}页")
            
            for i, entry in enumerate(unique_entries):
                title = entry['title']
                start_page = entry['page']
                
                if i + 1 < len(unique_entries):
                    end_page = unique_entries[i + 1]['page'] - 1
                else:
                    end_page = self.total_pages
                
                if start_page > end_page or start_page < 1:
                    logger.warning(f"跳过无效页码范围: {title} (第{start_page}-{end_page}页)")
                    continue
                
                if start_page == end_page:
                    end_page = start_page
                
                splits.append(DocumentSplit(
                    start_page=start_page,
                    end_page=end_page,
                    title=title
                ))
                logger.info(f"  目录条目{i+1}: {title} (第{start_page}-{end_page}页)")
        
        logger.info(f"基于目录创建了 {len(splits)} 个文档边界")
        self._toc_used = True  # 标记使用了目录拆分
        return splits

    def run(self):
        """执行完整拆分流程"""
        logger.info("=" * 60)
        logger.info("开始PDF智能拆分")
        logger.info(f"文档类型: {self.document_type}")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            # 1. 加载或生成OCR JSON
            if self.ocr_json_path and self.ocr_json_path.exists():
                ocr_data = self.load_ocr_json()
            else:
                # 自动生成OCR JSON
                logger.info("未提供OCR JSON文件，将自动进行OCR识别...")
                if not OCR_AVAILABLE:
                    raise RuntimeError("PaddleOCR未安装，无法自动OCR。请先安装：pip install paddleocr")
                
                ocr_json_path = self.generate_ocr_json()
                self.ocr_json_path = Path(ocr_json_path)
                ocr_data = self.load_ocr_json()
            
            # 2. 优先检查目录，如果有目录则使用目录拆分（零成本，不调用模型）
            splits = None
            toc_splits = None
            
            # 尝试检测目录（通用目录检测，适用于各种文档类型）
            logger.info("尝试检测目录（目录、目次、CONTENTS等）...")
            try:
                toc_splits = self._split_from_toc(ocr_data)
                if toc_splits and len(toc_splits) > 0:
                    splits = toc_splits
                    logger.info(f"根据目录生成了 {len(splits)} 个拆分结果")
                else:
                    logger.info("未检测到目录，将使用大模型拆分")
            except Exception as e:
                logger.warning(f"目录检测失败（不影响主流程）: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # 如果是法律文书类型，额外尝试证据目录检测
            if (splits is None or len(splits) == 0) and self.document_type == "legal":
                try:
                    from evidence_directory_helper import detect_evidence_directory, split_by_evidence_directory
                    
                    # 检测证据目录
                    evidence_dir_info = detect_evidence_directory(ocr_data)
                    
                    if evidence_dir_info:
                        logger.info("检测到证据目录，将使用证据目录进行拆分...")
                        evidence_splits = split_by_evidence_directory(ocr_data, self.total_pages)
                        
                        if evidence_splits and len(evidence_splits) > 0:
                            # 转换为DocumentSplit对象
                            splits = []
                            for ev_split in evidence_splits:
                                splits.append(DocumentSplit(
                                    start_page=ev_split['start_page'],
                                    end_page=ev_split['end_page'],
                                    title=ev_split['title']
                                ))
                            logger.info(f"根据证据目录生成了 {len(splits)} 个拆分结果")
                except Exception as e:
                    logger.warning(f"证据目录检测失败（不影响主流程）: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # 3. 如果目录拆分失败，使用大模型拆分
            if splits is None or len(splits) == 0:
                logger.info("使用大模型进行智能拆分...")
                page_texts = self.extract_page_texts(ocr_data)
                
                if not page_texts:
                    logger.warning("未提取到任何页面文本")
                    return
                
                splits = self.split_with_llm(page_texts)
                
                if not splits:
                    logger.warning("大模型未返回任何拆分结果")
                    return
            
            # 4. 拆分PDF
            self.split_pdf(splits)
            
            # 5. 保存拆分切点JSON（只包含切点信息）
            result_json_path = self.output_dir / "split_points.json"
            
            # 确定拆分方法
            split_method = "llm"
            if toc_splits:
                split_method = "toc"
            
            result_data = {
                "pdf_path": str(self.pdf_path),
                "ocr_json_path": str(self.ocr_json_path) if self.ocr_json_path else None,
                "total_pages": self.total_pages,
                "document_type": self.document_type,
                "split_method": split_method,
                "splits": [
                    {
                        "start_page": s.start_page,
                        "end_page": s.end_page,
                        "title": s.title
                    }
                    for s in splits
                ]
            }
            
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            logger.info(f"拆分切点已保存到: {result_json_path}")

            elapsed_time = datetime.now() - start_time
            logger.info("=" * 60)
            logger.info("PDF拆分完成")
            logger.info(f"总耗时: {elapsed_time}")
            logger.info(f"输出目录: {self.output_dir}")
            logger.info(f"拆分文件数: {len(splits)}")
            logger.info(f"拆分方法: {split_method}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"执行失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()
        except:
            pass


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='通用PDF智能拆分工具（支持目录检测和LLM拆分）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基本使用（自动OCR，自动检测目录）
  python pdf-split.py input.pdf -o ./output

  # 使用已有OCR JSON文件
  python pdf-split.py input.pdf --ocr-json ocr_result.json -o ./output

  # 指定文档类型
  python pdf-split.py input.pdf --document-type legal -o ./output
  python pdf-split.py input.pdf --document-type academic -o ./output
  python pdf-split.py input.pdf --document-type general -o ./output

  # 指定GPU/CPU
  python pdf-split.py input.pdf --use-gpu -o ./output
  python pdf-split.py input.pdf --use-cpu -o ./output

  # 调整图像缩放比例（大文件或内存受限时）
  python pdf-split.py input.pdf --image-scale 0.8 -o ./output
        """
    )
    
    parser.add_argument('pdf_path', help='输入PDF文件路径或URL')
    parser.add_argument('-o', '--output', default='test_result',
                       help='输出目录（默认：test_result）')
    parser.add_argument('--ocr-json', type=str,
                       help='OCR结果JSON文件路径（可选，如果不提供则自动OCR）')
    parser.add_argument('--use-gpu', action='store_true',
                       help='强制使用GPU（如果可用）')
    parser.add_argument('--use-cpu', action='store_true',
                       help='强制使用CPU（即使GPU可用）')
    parser.add_argument('--image-scale', type=float, default=1.0,
                       help='图像缩放比例（默认1.0，大文件可降低到0.7-0.8）')
    parser.add_argument('--document-type', type=str, 
                       choices=['general', 'legal', 'academic'], 
                       default='general',
                       help='文档类型：general（通用，默认）、legal（法律文书）、academic（学术文档）')

    args = parser.parse_args()
    
    # 检查输入是URL还是本地文件路径
    pdf_path = args.pdf_path
    is_url = pdf_path.startswith('http://') or pdf_path.startswith('https://')
    temp_file = None
    
    if is_url:
        # 从URL下载文件
        try:
            import requests
            import tempfile
            logger.info(f"从URL下载PDF文件: {pdf_path}")
            response = requests.get(pdf_path, stream=True, timeout=300)
            response.raise_for_status()
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir='/tmp')
            with temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            
            pdf_path = temp_file.name
            logger.info(f"PDF文件已下载到临时文件: {pdf_path}")
        except Exception as e:
            logger.error(f"从URL下载PDF文件失败: {e}")
            return 1
    else:
        # 检查本地文件是否存在
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return 1
    
    # 检查环境变量（仅在需要调用LLM时检查，如果有目录可能不需要）
    # 注意：如果PDF有目录，会优先使用目录拆分，不需要API key
    # 只有在目录拆分失败时才会需要API key
    # API key 仅通过环境变量提供（不在代码中硬编码默认值）
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("未检测到 LLM_API_KEY 或 DEEPSEEK_API_KEY，目录拆分失败时将无法调用大模型")
    else:
        logger.info("API key已配置（通过环境变量提供）")
    
    # 处理GPU选项
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.use_cpu:
        use_gpu = False
    
    try:
        splitter = LLMDocumentSplitter(
            pdf_path=pdf_path,
            output_dir=args.output,
            ocr_json_path=args.ocr_json,
            use_gpu=use_gpu,
            image_scale=args.image_scale,
            document_type=args.document_type
        )
        
        splitter.run()
        
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"已删除临时文件: {temp_file.name}")
            except:
                pass
        
        return 0

    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return 1


if __name__ == '__main__':
    exit(main())
