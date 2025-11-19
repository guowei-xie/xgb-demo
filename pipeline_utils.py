"""
训练流程通用工具

封装数据读取、信息打印与切分逻辑，避免在各脚本中重复实现。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    读取指定路径的CSV数据。

    Args:
        csv_path: 数据文件路径。

    Returns:
        pandas DataFrame。

    Raises:
        FileNotFoundError: 当文件不存在时。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    return pd.read_csv(csv_path)


def log_basic_stats(df: pd.DataFrame, target_col: str) -> None:
    """
    打印基础数据概览，聚焦必要指标。

    Args:
        df: 原始数据。
        target_col: 目标列名。
    """
    print(f"✓ 数据读取完成，共 {len(df)} 条记录")
    if target_col in df.columns:
        mean_val = df[target_col].mean()
        print(f"✓ {target_col} 均值: {mean_val:.2%}")
    print(f"✓ 数据列: {', '.join(df.columns)}")

    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if missing_stats.empty:
        print("✓ 无缺失值")
    else:
        print("✓ 缺失值列（Top5）:")
        for col, count in missing_stats.sort_values(ascending=False).head(5).items():
            print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    按照配置划分训练集与测试集。

    Args:
        X: 特征。
        y: 标签。
        test_size: 测试集比例。
        random_state: 随机种子。

    Returns:
        划分后的 (X_train, X_test, y_train, y_test)。
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

