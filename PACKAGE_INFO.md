# Python包打包信息

## 包结构

```
pdf-split-tool/
├── pdf_split_tool/          # Python包目录
│   ├── __init__.py          # 包初始化文件
│   ├── splitter.py          # 核心拆分逻辑
│   └── cli.py               # 命令行入口
├── setup.py                 # 打包配置文件
├── README.md                 # 项目说明
├── requirements.txt         # 依赖列表
├── MANIFEST.in              # 包含文件清单
└── docs/                    # 文档目录
```

## 安装方式

### 开发模式安装（推荐）

```bash
cd pdf-split-tool
python setup.py develop
```

### 普通安装

```bash
cd pdf-split-tool
python setup.py install
```

### 从源码包安装

```bash
# 先创建源码包
python setup.py sdist

# 然后安装
pip install dist/pdf_split_tool-1.0.0.tar.gz
```

## 使用方式

安装后可以使用命令行工具：

```bash
pdf-split input.pdf -o ./output
```

或者作为Python模块使用：

```python
from pdf_split_tool import LLMDocumentSplitter

splitter = LLMDocumentSplitter(
    pdf_path="input.pdf",
    output_dir="./output"
)
splitter.run()
```

## 打包命令

```bash
# 检查配置
python setup.py check

# 创建源码包
python setup.py sdist

# 查看打包内容
tar -tzf dist/pdf_split_tool-1.0.0.tar.gz
```

## 注意事项

- 包名使用下划线：`pdf_split_tool`
- 命令行工具名使用连字符：`pdf-split`
- 所有代码都在 `pdf_split_tool/` 目录下
- 使用 `entry_points` 创建命令行工具
