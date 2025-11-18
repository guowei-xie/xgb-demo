# XGBoost 用户报名率预测模型 Demo

这是一个完整的 XGBoost 模型训练演示项目，展示了从数据生成到模型评估的完整机器学习流程。

## 项目结构

```
xgb_demo/
├── data_generator.py      # 数据生成模块（生成MOCK用户数据）
├── feature_processor.py   # 特征处理模块（数据预处理和特征工程）
├── model_trainer.py       # 模型训练模块（XGBoost训练和评估）
├── analyzer.py            # 效果分析模块（可视化分析）
├── main.py                # 主程序入口
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明文档
```

## 功能特性

### 1. 数据生成 (`data_generator.py`)
- 生成包含用户特征的MOCK数据集
- 特征包括：城市、年级、年龄、房价、收入、教育年限、家庭规模、是否有车、访问次数等
- 标签：是否报名（基于特征的概率分布生成）

### 2. 特征处理 (`feature_processor.py`)
- 分类特征编码（Label Encoding）
- 特征工程（创建新特征，如房价收入比、访问频率等）
- 数值特征标准化
- 训练集/测试集划分

### 3. 模型训练 (`model_trainer.py`)
- XGBoost 二分类模型训练
- 模型评估（准确率、精确率、召回率、F1分数、ROC-AUC）
- 特征重要性分析
- 模型保存和加载

### 4. 效果分析 (`analyzer.py`)
- 特征重要性可视化
- ROC曲线
- 精确率-召回率曲线
- 混淆矩阵热力图
- 预测概率分布
- 数据探索性分析

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行完整流程

直接运行主程序，将执行完整的模型训练流程：

```bash
python main.py
```

### 单独运行各模块

#### 1. 生成数据
```bash
python data_generator.py
```

#### 2. 特征处理
```bash
python feature_processor.py
```

#### 3. 模型训练
```bash
python model_trainer.py
```

#### 4. 效果分析
```bash
python analyzer.py
```

## 输出文件

运行完成后，将生成以下文件：

```
xgb_demo/
├── data/
│   └── mock_data.csv              # 生成的MOCK数据
├── models/
│   ├── feature_processor.pkl      # 保存的特征处理器
│   └── xgb_model.pkl              # 训练好的XGBoost模型
└── results/
    ├── feature_importance.png     # 特征重要性图
    ├── roc_curve.png              # ROC曲线
    ├── precision_recall_curve.png # PR曲线
    ├── confusion_matrix.png       # 混淆矩阵
    ├── prediction_distribution.png # 预测概率分布
    └── data_exploration.png       # 数据探索分析
```

## 数据说明

### 用户特征
- **city**: 城市（北京、上海、广州等10个城市）
- **grade**: 年级（小学、初中、高中、大学、研究生、在职）
- **age**: 年龄（18-50岁）
- **house_price**: 房价（对数正态分布）
- **income**: 收入（对数正态分布）
- **education_years**: 教育年限（9-22年）
- **family_size**: 家庭规模（1-5人）
- **has_car**: 是否有车（0/1）
- **visit_count**: 访问次数（泊松分布）
- **last_visit_days**: 距离上次访问天数（指数分布）

### 标签
- **is_enrolled**: 是否报名（0/1）

## 模型参数

默认XGBoost参数：
- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 3
- `gamma`: 0.1
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0

可在 `main.py` 中修改模型参数。

## 评估指标

模型将输出以下评估指标：
- **准确率 (Accuracy)**: 正确预测的比例
- **精确率 (Precision)**: 预测为正例中真正为正例的比例
- **召回率 (Recall)**: 真正例中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **ROC-AUC**: ROC曲线下面积

## 注意事项

1. 本项目使用MOCK数据，仅用于演示目的
2. 实际应用中需要根据真实数据调整特征工程和模型参数
3. 建议在训练前进行更深入的数据探索和特征选择
4. 可根据实际需求调整模型超参数以优化性能

## 依赖库版本

- pandas >= 2.0.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

## 许可证

本项目仅供学习和演示使用。

