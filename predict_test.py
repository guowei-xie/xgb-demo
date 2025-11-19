"""
测试集预测脚本

仅保留加载模型->生成特征->推理->保存结果的必要步骤。
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

from feature_processor import FeatureProcessor
from model_trainer import ModelTrainer
from config import PROCESSOR_PATH, MODEL_PATH, TEST_PATH, PREDICTION_OUTPUT
from pipeline_utils import load_dataset


def load_artifacts(processor_path: Path, model_path: Path) -> tuple[FeatureProcessor, ModelTrainer]:
    """
    加载特征处理器与模型。

    Args:
        processor_path: 特征处理器文件路径。
        model_path: 模型文件路径。

    Returns:
        (processor, trainer) 元组。
    """
    if not processor_path.exists():
        raise FileNotFoundError(f"未找到特征处理器: {processor_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    processor: FeatureProcessor = FeatureProcessor.load(str(processor_path))
    trainer: ModelTrainer = ModelTrainer.load(str(model_path))
    return processor, trainer


def prepare_test_features(test_path: Path, processor: FeatureProcessor) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取测试集并生成特征。

    Args:
        test_path: 测试集CSV路径。
        processor: 已拟合的特征处理器。

    Returns:
        (原始DataFrame, 特征DataFrame)。
    """
    test_df = load_dataset(test_path)
    features = processor.transform(test_df)
    return test_df, features


def run_prediction(trainer: ModelTrainer, features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    使用模型进行预测。

    Args:
        trainer: 已加载的模型训练器。
        features: 特征数据。

    Returns:
        (预测概率Series, 预测标签Series)。
    """
    proba = pd.Series(trainer.predict_proba(features), index=features.index, name="pred_probability")
    labels = pd.Series(trainer.predict(features), index=features.index, name="pred_label")
    return proba, labels


def save_predictions(test_df: pd.DataFrame, proba: pd.Series, labels: pd.Series, output_path: Path) -> None:
    """
    合并预测结果并保存。

    Args:
        test_df: 原始测试数据。
        proba: 预测概率。
        labels: 预测标签。
        output_path: 输出CSV路径。
    """
    os.makedirs(output_path.parent, exist_ok=True)
    result_df = test_df.copy()
    result_df["pred_probability"] = proba
    result_df["pred_label"] = labels
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")


def main():
    """
    主流程：加载资源 -> 生成特征 -> 预测 -> 保存结果。
    """
    print("加载特征处理器与模型...")
    processor, trainer = load_artifacts(PROCESSOR_PATH, MODEL_PATH)

    print("读取测试集并生成特征...")
    test_df, features = prepare_test_features(TEST_PATH, processor)
    print(f"测试集样本数: {len(test_df)}")

    print("执行预测...")
    proba, labels = run_prediction(trainer, features)

    print("保存预测结果...")
    save_predictions(test_df, proba, labels, PREDICTION_OUTPUT)
    print("全部流程完成！")


if __name__ == "__main__":
    main()

