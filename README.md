# PDF智能拆分工具

基于大模型和OCR的通用PDF文档智能拆分工具，适用于将合并的PDF文档拆分为多个独立文档。

## 核心功能

1. **自动OCR识别**：支持GPU/CPU自动切换，使用PaddleOCR进行文本识别
2. **智能拆分策略**：
   - 优先使用目录进行精确拆分（零成本，不调用模型）
   - 目录不可用时，使用大模型进行智能拆分
3. **通用文档支持**：支持所有文档类型（法律文书、学术论文、通用文档等）
4. **自动规范化命名**：根据文档标题自动生成规范的文件名
5. **配置化关键词**：补充材料和页面类型关键词可配置，便于扩展

## 系统要求

- Python 3.8+
- 支持GPU（可选，自动检测）
- 足够的磁盘空间存储OCR结果和拆分后的PDF

## 安装依赖

```bash
pip install -r requirements.txt
```

### 额外依赖（根据使用场景）

**OCR支持（必需）**：
```bash
pip install paddlepaddle paddleocr
```

**PDF处理（必需）**：
```bash
pip install pymupdf
```

**大模型API（必需）**：
```bash
pip install openai
```

**可选依赖**：
```bash
pip install python-dotenv  # 环境变量支持
```

## 配置

### 环境变量配置

创建 `.env` 文件（可选）：

```bash
# LLM API配置
LLM_API_KEY=your_api_key_here
LLM_API_BASE_URL=https://api.example.com/v1

# 或使用DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 代码中配置

出于安全和开源最佳实践考虑，本项目**不在代码中硬编码任何API密钥**。  
请务必通过环境变量（如 `LLM_API_KEY`、`LLM_API_BASE_URL`、`DEEPSEEK_API_KEY`）来配置大模型访问参数。

## 使用方法

### 基本用法

```bash
python pdf-split.py <input.pdf> -o <output_dir>
```

### 完整参数

```bash
python pdf-split.py <input.pdf> \
    -o <output_dir> \                    # 输出目录
    --document-type <type> \              # 文档类型: general/legal/academic
    --ocr-json <path> \                  # 使用已有的OCR JSON文件
    --use-gpu \                           # 强制使用GPU（如果可用）
    --no-gpu \                            # 强制使用CPU
    --batch-size <n> \                    # OCR批处理大小（默认2）
    --image-scale <scale>                 # 图像缩放比例（默认0.6，大文件可降低）
```

### 示例

```bash
# 基本拆分
python pdf-split.py document.pdf -o ./result

# 指定文档类型
python pdf-split.py academic_papers.pdf -o ./result --document-type academic

# 使用已有OCR结果（跳过OCR步骤）
python pdf-split.py document.pdf -o ./result --ocr-json ./ocr_result.json

# 大文件优化（降低内存使用）
python pdf-split.py large_document.pdf -o ./result --image-scale 0.5 --batch-size 1
```

## 输出结果

拆分完成后，输出目录包含：

- `split_points.json`：拆分点信息（JSON格式）
  - `total_pages`：总页数
  - `splits`：拆分结果列表
    - `start_page`：起始页
    - `end_page`：结束页
    - `title`：文档标题
- `*_ocr.json`：OCR识别结果（可选，用于后续处理）
- `01_<title>.pdf`、`02_<title>.pdf`...：拆分后的PDF文件

## 工作原理

### 1. OCR识别阶段

- 自动检测GPU可用性
- 使用PaddleOCR进行文本识别
- 生成简化的OCR JSON（仅包含关键信息：页码、页高、文本和Y坐标）

### 2. 拆分策略

**策略1：目录拆分（优先）**
- 自动检测PDF中的目录页
- 提取目录条目和页码
- 根据目录精确拆分（零成本，不调用模型）

**策略2：大模型智能拆分**
- 提取页面关键信息（页眉、前3行文本、页面类型）
- 构建紧凑的提示词发送给LLM
- LLM分析文档结构并返回拆分建议

### 3. 后处理

- 修正重叠页面
- 合并补充材料（附录、参考文献等）
- 验证页面覆盖完整性
- 规范化文件名

## 配置化关键词

工具使用配置化的关键词系统，便于扩展：

### 补充材料关键词

```python
SUPPLEMENT_KEYWORDS = {
    'appendix': ['appendix', '附录'],
    'references': ['references', 'bibliography', '参考文献'],
    'supplementary': ['supplementary', '补充材料', 'supplement']
}
```

### 页面类型关键词

```python
PAGE_TYPE_KEYWORDS = {
    'toc': ['目录', 'contents', 'table of contents', '目 录'],
    'abstract': ['abstract', '摘要'],
    'references': ['references', 'bibliography', '参考文献'],
    'title_page': ['abstract', '摘要', 'introduction', '引言', ...]
}
```

可以在代码中修改这些配置以适应不同的文档类型。

## 性能优化

### 内存优化

- 大文件自动降低图像缩放比例（`--image-scale 0.5-0.6`）
- 降低OCR批处理大小（`--batch-size 1-2`）
- 及时释放内存（使用 `gc.collect()`）

### Token优化

- OCR JSON仅包含关键信息（文本和Y坐标）
- 提示词仅包含页眉和前3行文本
- 大文件自动截断，确保所有页面都被处理

### GPU加速

- 自动检测GPU可用性
- 支持PaddleOCR GPU加速
- 无GPU时自动降级到CPU

## 故障排除

### 常见问题

1. **OCR初始化失败**
   - 检查PaddleOCR是否正确安装
   - 检查GPU驱动和CUDA版本
   - 尝试使用 `--no-gpu` 强制使用CPU

2. **内存不足（Exit code 137）**
   - 降低 `--image-scale`（如 0.5）
   - 降低 `--batch-size`（如 1）
   - 使用已有OCR JSON跳过OCR步骤

3. **拆分结果不准确**
   - 检查OCR质量（查看OCR JSON）
   - 尝试不同的文档类型参数
   - 检查LLM API是否正常工作

4. **API调用失败**
   - 检查API密钥和基础URL配置
   - 使用 `test_api.py` 测试API连接
   - 检查网络连接

## 测试API连接

```bash
python test_api.py
```

## 注意事项

1. **拆分原则**：工具采用"宁可多拆分，不要错误合并"的原则
2. **标题提取**：如果无法提取明确标题，会使用页眉信息或生成默认标题
3. **页面覆盖**：工具会验证所有页面都被覆盖，无遗漏无重叠
4. **文件命名**：文件名中的特殊字符会被替换为下划线，确保文件系统兼容性

## 许可证

本工具为内部使用工具，请遵守相关使用规范。

## 更新日志

### v1.0
- 初始版本
- 支持目录拆分和LLM智能拆分
- 支持GPU/CPU自动切换
- 配置化关键词系统
- Token优化和性能优化


