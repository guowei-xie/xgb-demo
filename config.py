"""
项目配置

集中维护数据路径、模型参数等，避免在各模块中重复硬编码。
"""
from __future__ import annotations

from pathlib import Path


# 工程目录
BASE_DIR = Path(__file__).resolve().parent

# 数据与模型文件
DATA_PATH = BASE_DIR / "data" / "raw.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
PROCESSOR_PATH = BASE_DIR / "models" / "feature_processor.pkl"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
PREDICTION_OUTPUT = BASE_DIR / "results" / "test_predictions.csv"
TRAIN_TEST_PREDICTIONS = BASE_DIR / "results" / "train_test_predictions.csv"  # 训练时切分的测试集预测结果

# 训练配置
TARGET_COLUMN = "is_renewal"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20

# 模型参数（可按需调整）
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# 评估阈值
DEFAULT_THRESHOLD = 0.5

