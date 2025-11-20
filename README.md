# XGBoost 用户续报预测系统

基于 XGBoost 的用户续报概率预测系统，提供完整的机器学习工作流：数据预处理、特征工程、模型训练、超参数调优、预测推理和 RESTful API 服务。

## 项目特性

- 🚀 **完整的 ML 工作流**：从数据加载到模型部署的端到端解决方案
- 📊 **特征工程**：自动特征创建、类别特征处理、缺失值处理
- 🎯 **模型训练**：XGBoost 二分类模型，支持早停和交叉验证
- 🔧 **超参数调优**：支持随机搜索和网格搜索，自动寻找最佳参数
- 📈 **结果分析**：多维度预测分析、等级标签分配、可视化报告
- 🌐 **API 服务**：基于 FastAPI 的 RESTful API，支持单条和批量预测
- 📦 **模型管理**：模型和特征处理器的持久化保存与加载

## 项目结构

```
xgb_demo/
├── config.py                  # 全局配置文件（路径、参数、阈值等）
├── train.py                   # 模型训练主程序
├── predict.py                 # 测试集预测脚本
├── tune.py                    # 超参数调优脚本
├── api.py                     # FastAPI 服务入口
├── requirements.txt           # 项目依赖
├── README.md                  # 项目说明文档
├── data/                      # 数据目录
│   ├── raw.csv               # 训练数据
│   └── test.csv              # 测试数据
├── models/                    # 模型文件目录
│   ├── feature_processor.pkl # 特征处理器
│   ├── xgb_model.pkl         # XGBoost 模型
│   └── best_params.pkl       # 最佳超参数（调优后）
├── results/                   # 结果输出目录
│   ├── *.csv                 # 预测结果 CSV
│   └── *.png                 # 分析可视化图表
└── utils/                     # 工具模块
    ├── feature_processor.py   # 特征处理模块
    ├── model_trainer.py       # 模型训练模块
    ├── hyperparameter_tuner.py # 超参数调优模块
    ├── level_tagger.py        # 等级标签分配工具
    ├── prediction_analysis.py # 预测结果分析模块
    └── pipeline_utils.py      # 流程工具函数
```

## 快速开始

### 1. 环境准备

#### 创建虚拟环境

**Windows 系统：**

```bash
# 使用 venv 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

**Mac/Linux 系统：**

```bash
# 使用 venv 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

#### 安装依赖

```bash
# 确保虚拟环境已激活（命令行前应显示 (venv)）
# 升级 pip（推荐）
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 数据准备

将训练数据放置在 `data/raw.csv`，测试数据放置在 `data/test.csv`。
`raw.csv`的取数sql: https://git.corp.hetao101.com/snippets/1179
`test.csv`的取数sql: https://git.corp.hetao101.com/snippets/1180

数据应包含以下特征列：
- `city`: 城市名称
- `city_level`: 城市等级
- `city_score`: 城市评分
- `house_price`: 房价
- `grade`: 年级
- `refresh_num`: 复刷次数
- `device`: 设备类型
- `is_enable`: 是否激活课程
- `fns_cnt`: 完课计数
- `is_renewal`: 续报标签（0/1，训练数据必需）

### 3. 模型训练

```bash
# 执行完整训练流程
python train.py
```

训练流程包括：
1. 数据读取与统计
2. 特征处理与工程
3. 数据集划分（训练集/测试集）
4. XGBoost 模型训练
5. 模型评估与特征重要性分析
6. 预测结果保存与分析可视化

**输出文件：**
- `models/feature_processor.pkl`: 特征处理器
- `models/xgb_model.pkl`: 训练好的模型
- `results/train_test_predictions.csv`: 验证集预测结果
- `results/train_test_prediction_analysis.png`: 预测质量分析图
- `results/train_test_level_analysis.png`: 等级-学期分析图
- `results/feature_importance.png`: 特征重要性图

### 4. 超参数调优（可选）

```bash
# 执行超参数搜索
python tune.py
```

支持两种搜索方法：
- **随机搜索**：快速探索参数空间（默认）
- **网格搜索**：全面搜索所有参数组合

**输出文件：**
- `models/best_params.pkl`: 最佳超参数
- `results/cv_results.csv`: 交叉验证结果

### 5. 测试集预测

```bash
# 对测试集进行预测
python predict.py
```

**输出文件：**
- `results/test_predictions.csv`: 测试集预测结果
- `results/prediction_analysis.png`: 预测分析图
- `results/prediction_level_analysis.png`: 等级分析图

### 6. 启动 API 服务

```bash
# 启动 FastAPI 服务
python api.py
```

服务默认运行在 `http://0.0.0.0:8000`

**API 文档：**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API 使用说明

### 健康检查

```bash
curl http://localhost:8000/health
```

### 单条预测

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "扬州市",
    "city_level": "三线城市",
    "city_score": 120.0,
    "house_price": 4941.0,
    "grade": 3,
    "refresh_num": 1,
    "device": "电脑",
    "is_enable": 1,
    "fns_cnt": 5
  }'
```

**响应示例：**
```json
{
  "probability": 0.1234,
  "level_tag": "A",
  "success": true
}
```

### 批量预测

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "users": [
      {
        "city": "扬州市",
        "city_level": "三线城市",
        "city_score": 120.0,
        "house_price": 4941.0,
        "grade": 3,
        "refresh_num": 1,
        "device": "电脑",
        "is_enable": 1,
        "fns_cnt": 5
      },
      {
        "city": "扬州市",
        "city_level": "三线城市",
        "city_score": 120.0,
        "house_price": 4941.0,
        "grade": 3,
        "refresh_num": 1,
        "device": "电脑",
        "is_enable": 1,
        "fns_cnt": 5
      }
    ]
  }'
```

## 配置说明

主要配置项位于 `config.py`：

### 数据路径
- `DATA_PATH`: 训练数据路径
- `TEST_PATH`: 测试数据路径
- `MODEL_PATH`: 模型保存路径
- `PROCESSOR_PATH`: 特征处理器保存路径

### 模型参数
- `MODEL_PARAMS`: XGBoost 默认参数
- `HYPERPARAMETER_SEARCH_SPACE`: 超参数搜索空间
- `HYPERPARAMETER_TUNING_CONFIG`: 调优配置

### 等级标签规则
- `LEVEL_TAG_RULES`: 根据预测概率分配等级标签（S/A/B/C/D）

### API 配置
- `API_HOST`: 服务监听地址（默认：0.0.0.0）
- `API_PORT`: 服务端口（默认：8000）
- `API_RELOAD`: 是否启用自动重载（开发环境建议 True）
- `API_WORKERS`: 工作进程数（生产环境建议设置为 CPU 核心数）

### 特征配置
- `FEATURE_INCLUDE_COLUMNS`: 显式指定特征列（None 表示自动选择）
- `FEATURE_EXCLUDE_COLUMNS`: 排除的特征列

## 核心模块说明

### 特征处理器 (`utils/feature_processor.py`)

- **自动特征工程**：创建交互特征、非线性变换等
- **类别特征处理**：保持训练和预测时类别一致性
- **缺失值处理**：利用 XGBoost 对缺失值的原生支持
- **特征选择**：支持配置化特征包含/排除

### 模型训练器 (`utils/model_trainer.py`)

- **XGBoost 训练**：支持早停、交叉验证
- **模型评估**：准确率、精确率、召回率、F1、ROC-AUC
- **特征重要性**：自动计算和可视化
- **模型持久化**：保存和加载模型

### 超参数调优器 (`utils/hyperparameter_tuner.py`)

- **搜索方法**：随机搜索、网格搜索
- **交叉验证**：K 折交叉验证评估
- **结果保存**：保存最佳参数和 CV 结果
- **性能对比**：与默认参数对比

### 预测分析 (`utils/prediction_analysis.py`)

- **预测质量分析**：概率分布、校准曲线、ROC 曲线
- **等级分析**：按学期分析各等级表现
- **趋势分析**：续报率趋势、等级占比趋势
- **可视化**：自动生成分析图表

### 等级标签器 (`utils/level_tagger.py`)

- **规则匹配**：根据概率区间分配等级标签
- **灵活配置**：支持自定义等级规则

## 评估指标

模型训练和评估使用以下指标：

- **准确率 (Accuracy)**: 正确预测的比例
- **精确率 (Precision)**: 预测为正例中真正为正例的比例
- **召回率 (Recall)**: 真正例中被正确预测的比例
- **F1 分数**: 精确率和召回率的调和平均
- **ROC-AUC**: ROC 曲线下面积

## 等级标签说明

根据预测概率自动分配等级标签：

- **S 级**: 概率 > 0.13（高续报概率）
- **A 级**: 0.08 < 概率 ≤ 0.13（中高续报概率）
- **B 级**: 0.03 < 概率 ≤ 0.08（中等续报概率）
- **C 级**: 0.015 < 概率 ≤ 0.03（低续报概率）
- **D 级**: 概率 ≤ 0.015（极低续报概率）

可在 `config.py` 中修改 `LEVEL_TAG_RULES` 自定义规则。

## 注意事项

1. **数据格式**：确保训练数据和测试数据格式一致，包含必要的特征列
2. **模型文件**：运行 API 服务前需先训练模型，确保 `models/` 目录下有模型文件
3. **特征一致性**：预测时使用的特征应与训练时保持一致
4. **生产环境**：部署到生产环境时，建议设置 `API_RELOAD=False` 并调整 `API_WORKERS`
5. **超参数调优**：调优过程可能耗时较长，建议在充足的计算资源下运行

## 开发建议

1. **特征工程**：根据业务需求在 `FeatureProcessor.create_features()` 中添加新特征
2. **模型优化**：通过 `tune.py` 寻找最佳超参数，或手动调整 `config.py` 中的 `MODEL_PARAMS`
3. **结果分析**：查看 `results/` 目录下的可视化图表，分析模型表现
4. **API 扩展**：在 `api.py` 中添加新的 API 端点以满足业务需求
