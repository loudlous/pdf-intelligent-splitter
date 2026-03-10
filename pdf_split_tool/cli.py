#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF拆分工具命令行入口
"""

import os
import sys
import argparse
import logging
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .splitter import LLMDocumentSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_split_tool.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='通用PDF智能拆分工具（支持目录检测和LLM拆分）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基本使用（自动OCR，自动检测目录）
  pdf-split input.pdf -o ./output

  # 使用已有OCR JSON文件
  pdf-split input.pdf --ocr-json ocr_result.json -o ./output

  # 指定文档类型
  pdf-split input.pdf --document-type legal -o ./output
  pdf-split input.pdf --document-type academic -o ./output
  pdf-split input.pdf --document-type general -o ./output

  # 指定GPU/CPU
  pdf-split input.pdf --use-gpu -o ./output
  pdf-split input.pdf --use-cpu -o ./output

  # 调整图像缩放比例（大文件或内存受限时）
  pdf-split input.pdf --image-scale 0.8 -o ./output
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
    sys.exit(main())


