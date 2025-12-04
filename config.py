"""
项目配置

集中维护数据路径、模型参数等，避免在各模块中重复硬编码。
"""
from __future__ import annotations

from pathlib import Path


# 工程目录
BASE_DIR = Path(__file__).resolve().parent

# 数据与模型文件
DATA_PATH = BASE_DIR / "data" / "raw.csv" # 训练数据（用于切分训练集和测试集）
TEST_PATH = BASE_DIR / "data" / "test.csv"
PROCESSOR_PATH = BASE_DIR / "models" / "feature_processor.pkl"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
PREDICTION_OUTPUT = BASE_DIR / "results" / "test_predictions.csv"
TRAIN_TEST_PREDICTIONS = BASE_DIR / "results" / "train_test_predictions.csv"  # 训练时切分的测试集预测结果
TRAIN_TEST_ANALYSIS_FIG = BASE_DIR / "results" / "train_test_prediction_analysis.png"
PREDICTION_ANALYSIS_FIG = BASE_DIR / "results" / "prediction_analysis.png"
TRAIN_TEST_LEVEL_ANALYSIS_FIG = BASE_DIR / "results" / "train_test_level_analysis.png"
TRAIN_TEST_LEVEL_RATIO_FIG = BASE_DIR / "results" / "train_test_level_ratio.png"
TRAIN_TEST_TERM_TREND_FIG = BASE_DIR / "results" / "train_test_term_trend.png"
PREDICTION_LEVEL_ANALYSIS_FIG = BASE_DIR / "results" / "prediction_level_analysis.png"
PREDICTION_LEVEL_RATIO_FIG = BASE_DIR / "results" / "prediction_level_ratio.png"
PREDICTION_TERM_TREND_FIG = BASE_DIR / "results" / "prediction_term_trend.png"
FEATURE_IMPORTANCE_FIG = BASE_DIR / "results" / "feature_importance.png"

# 等级标签区间配置
LEVEL_TAG_RULES = [
    {
        "label": "S",
        "min_prob": 0.15,
        "max_prob": None,
        "min_inclusive": False,
        "max_inclusive": False,
    },
    {
        "label": "A",
        "min_prob": 0.08,
        "max_prob": 0.15,
        "min_inclusive": False,
        "max_inclusive": True,
    },
    {
        "label": "B",
        "min_prob": 0.025,
        "max_prob": 0.08,
        "min_inclusive": False,
        "max_inclusive": True,
    },
    {
        "label": "C",
        "min_prob": 0.011,
        "max_prob": 0.025,
        "min_inclusive": False,
        "max_inclusive": True,
    },
    {
        "label": "D",
        "min_prob": None,
        "max_prob": 0.011,
        "min_inclusive": True,
        "max_inclusive": True,
    },
]

# 训练配置
TARGET_COLUMN = "is_renewal"
TERM_COLUMN = "l1_term_name"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30

# 特征列配置
# - FEATURE_INCLUDE_COLUMNS: 显式指定可用特征列（None 表示自动选择除排除项以外的所有列）
# - FEATURE_EXCLUDE_COLUMNS: 在默认排除列表的基础上额外排除的列
FEATURE_INCLUDE_COLUMNS: list[str] | None = None
FEATURE_EXCLUDE_COLUMNS: list[str] = [
    # 在此添加不希望进入模型的列，例如：
    # "example_column",
    "district",
    "is_enable_fns_interaction",
    "fns_cnt_per_grade",
    "refresh_fns_interaction",
    "city_score_house_price_ratio",
    "house_price_log"
]

# 模型参数（可按需调整）
# 版本1 默认
# MODEL_PARAMS = {
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "max_depth": 6,
#     "learning_rate": 0.1,
#     "n_estimators": 200,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "min_child_weight": 3,
#     "gamma": 0.1,
#     "reg_alpha": 0.1,
#     "reg_lambda": 1.0,
#     "random_state": 42,
#     "n_jobs": -1,
# }

# # 版本2 AI建议
# MODEL_PARAMS = {
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "max_depth": 8,                    # 增加深度捕捉更复杂模式
#     "learning_rate": 0.05,             # 降低学习率，更稳定
#     "n_estimators": 500,               # 增加迭代次数
#     "subsample": 0.9,                  # 增加样本使用率
#     "colsample_bytree": 0.7,           # 减少特征使用，增加多样性
#     "min_child_weight": 5,             # 加强防止过拟合
#     "gamma": 0.2,                      # 增加分裂难度
#     "reg_alpha": 0.05,                 # 减少L1约束
#     "reg_lambda": 1.5,                 # 适当增加L2约束
#     "random_state": 42,
#     "n_jobs": -1,
# }

# 版本3 网格搜索最优
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 8,                   
    "learning_rate": 0.01,             
    "n_estimators": 600,               
    "subsample": 0.7,
    "colsample_bytree": 0.6, 
    "min_child_weight": 7,            
    "gamma": 0.2,                      
    "reg_alpha": 0.01,                 
    "reg_lambda": 2.0,                
    "random_state": 42,
    "n_jobs": -1,
}

# 超参数调优配置
# 超参数搜索空间（用于随机搜索或网格搜索）
HYPERPARAMETER_SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10, 12],                    # 树的最大深度
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],    # 学习率
    "n_estimators": [200, 300, 400, 500, 600],         # 树的数量
    "subsample": [0.7, 0.8, 0.9, 1.0],                 # 样本采样比例
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],    # 特征采样比例
    "min_child_weight": [1, 3, 5, 7, 10],              # 叶子节点最小权重
    "gamma": [0, 0.1, 0.2, 0.3, 0.5],                  # 最小损失减少量
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.2],            # L1正则化
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],           # L2正则化
}

# 超参数调优方法配置
HYPERPARAMETER_TUNING_CONFIG = {
    "search_method": "random",          # 'random' 或 'grid'
    "n_iter": 50,                       # 随机搜索迭代次数（仅random有效）
    "cv": 5,                            # 交叉验证折数
    "scoring": "roc_auc",               # 评分指标
    "n_jobs": -1,                       # 并行任务数，-1表示使用所有CPU
    "random_state": 42,                 # 随机种子
    "verbose": 1,                       # 详细程度
}


# 超参数调优结果保存路径
BEST_PARAMS_PATH = BASE_DIR / "models" / "best_params.pkl"
CV_RESULTS_PATH = BASE_DIR / "results" / "cv_results.csv"

# API服务配置
API_HOST = "0.0.0.0"  # API服务监听地址
API_PORT = 8000  # API服务端口
API_RELOAD = True  # 是否启用自动重载（开发环境建议True，生产环境建议False）
API_WORKERS = 4  # 工作进程数（生产环境可设置为CPU核心数）
