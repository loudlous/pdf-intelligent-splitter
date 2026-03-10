# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install paddlepaddle paddleocr pymupdf openai
```

### 2. 配置API

编辑 `.env` 文件或直接在代码中配置API密钥：

```bash
# .env 文件
LLM_API_KEY=your_api_key
LLM_API_BASE_URL=https://api.example.com/v1
```

### 3. 运行拆分

```bash
python pdf-split.py input.pdf -o output_dir
```

## 详细参数说明

### 必需参数

- `input.pdf`：输入的PDF文件路径

### 可选参数

- `-o, --output-dir`：输出目录（默认：`test_result`）
- `--document-type`：文档类型
  - `general`：通用文档（默认）
  - `legal`：法律文书
  - `academic`：学术论文
- `--ocr-json`：使用已有的OCR JSON文件（跳过OCR步骤）
- `--use-gpu`：强制使用GPU（如果可用）
- `--no-gpu`：强制使用CPU
- `--batch-size`：OCR批处理大小（默认：2，大文件建议1）
- `--image-scale`：图像缩放比例（默认：0.6，范围0.3-1.0）

## 使用场景

### 场景1：有目录的PDF

如果PDF包含目录，工具会自动使用目录进行精确拆分，无需调用LLM。

```bash
python pdf-split.py document_with_toc.pdf -o ./result
```

### 场景2：无目录的PDF

工具会使用LLM进行智能拆分：

```bash
python pdf-split.py document_no_toc.pdf -o ./result --document-type general
```

### 场景3：学术论文

对于学术论文，使用 `academic` 类型可以获得更好的拆分效果：

```bash
python pdf-split.py papers.pdf -o ./result --document-type academic
```

### 场景4：大文件处理

对于大文件（>100页），建议降低图像缩放和批处理大小：

```bash
python pdf-split.py large_file.pdf -o ./result \
    --image-scale 0.5 \
    --batch-size 1
```

### 场景5：复用OCR结果

如果已经进行过OCR，可以复用结果：

```bash
# 第一次运行（生成OCR JSON）
python pdf-split.py document.pdf -o ./result

# 后续运行（使用已有OCR JSON）
python pdf-split.py document.pdf -o ./result2 \
    --ocr-json ./result/document_ocr.json
```

## 输出文件说明

### split_points.json

包含拆分点的详细信息：

```json
{
  "pdf_path": "input.pdf",
  "total_pages": 100,
  "document_type": "general",
  "split_method": "llm",
  "splits": [
    {
      "start_page": 1,
      "end_page": 10,
      "title": "文档标题"
    }
  ]
}
```

### OCR JSON文件

包含OCR识别结果，格式简化：

```json
{
  "pages": [
    {
      "page_num": 1,
      "page_height": 842.0,
      "texts": [
        {"text": "文本内容", "y": 100.5}
      ]
    }
  ]
}
```

## 性能建议

### 内存优化

- 大文件（>200页）：`--image-scale 0.5 --batch-size 1`
- 中等文件（50-200页）：`--image-scale 0.6 --batch-size 2`
- 小文件（<50页）：默认设置即可

### 速度优化

- 有GPU时自动使用GPU加速OCR
- 使用已有OCR JSON跳过OCR步骤
- 有目录时优先使用目录拆分（最快）

### Token优化

工具已自动优化Token使用：
- OCR JSON仅包含关键信息
- 提示词仅包含页眉和前3行文本
- 大文件自动截断

## 常见问题

### Q: 拆分结果不准确怎么办？

A: 
1. 检查OCR质量（查看OCR JSON文件）
2. 尝试不同的文档类型参数
3. 检查LLM API是否正常工作
4. 查看日志文件了解详细错误信息

### Q: 内存不足怎么办？

A:
1. 降低 `--image-scale`（如 0.5）
2. 降低 `--batch-size`（如 1）
3. 使用已有OCR JSON跳过OCR步骤
4. 分批处理大文件

### Q: 如何提高拆分准确性？

A:
1. 确保PDF质量良好（文字清晰）
2. 使用合适的文档类型参数
3. 检查OCR识别结果
4. 对于特殊格式，可能需要调整关键词配置

### Q: 可以自定义关键词吗？

A: 可以，在 `pdf-split.py` 中修改类常量：
- `SUPPLEMENT_KEYWORDS`：补充材料关键词
- `PAGE_TYPE_KEYWORDS`：页面类型关键词


