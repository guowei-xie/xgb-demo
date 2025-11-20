"""
超参数调优主程序

执行XGBoost模型的超参数搜索，找到最佳参数组合。
"""
import os
import pandas as pd
from utils.feature_processor import FeatureProcessor
from utils.hyperparameter_tuner import HyperparameterTuner
from config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    HYPERPARAMETER_SEARCH_SPACE,
    HYPERPARAMETER_TUNING_CONFIG,
    BEST_PARAMS_PATH,
    CV_RESULTS_PATH,
    MODEL_PARAMS,
    FEATURE_INCLUDE_COLUMNS,
    FEATURE_EXCLUDE_COLUMNS,
)
from utils.pipeline_utils import load_dataset, log_basic_stats, split_dataset


def main():
    """主函数：执行超参数调优流程"""
    print("=" * 60)
    print("XGBoost模型超参数调优")
    print("=" * 60)
    
    # 步骤1: 读取数据
    print("\n【步骤1】读取数据...")
    print("-" * 60)
    df = load_dataset(DATA_PATH)
    log_basic_stats(df, TARGET_COLUMN)
    
    # 步骤2: 特征处理
    print("\n【步骤2】特征处理与工程...")
    print("-" * 60)
    processor = FeatureProcessor(
        include_columns=FEATURE_INCLUDE_COLUMNS,
        exclude_columns=FEATURE_EXCLUDE_COLUMNS,
    )
    X, y = processor.fit_transform(df, target_col=TARGET_COLUMN)
    print(f"✓ 特征处理完成")
    print(f"✓ 特征数量: {X.shape[1]}")
    
    # 步骤3: 数据集划分（用于超参数调优，使用全部训练数据）
    print("\n【步骤3】准备调优数据集...")
    print("-" * 60)
    # 为了充分利用数据，超参数调优使用全部数据
    # 如果需要，也可以只使用训练集
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✓ 数据集划分完成")
    print(f"  - 训练集: {X_train.shape[0]} 条 (用于超参数调优)")
    print(f"  - 测试集: {X_test.shape[0]} 条 (保留用于最终评估)")
    
    # 步骤4: 超参数调优
    print("\n【步骤4】超参数搜索...")
    print("-" * 60)
    
    # 创建调优器
    tuner = HyperparameterTuner(
        search_space=HYPERPARAMETER_SEARCH_SPACE,
        search_method=HYPERPARAMETER_TUNING_CONFIG["search_method"],
        n_iter=HYPERPARAMETER_TUNING_CONFIG["n_iter"],
        cv=HYPERPARAMETER_TUNING_CONFIG["cv"],
        scoring=HYPERPARAMETER_TUNING_CONFIG["scoring"],
        n_jobs=HYPERPARAMETER_TUNING_CONFIG["n_jobs"],
        random_state=HYPERPARAMETER_TUNING_CONFIG["random_state"],
        verbose=HYPERPARAMETER_TUNING_CONFIG["verbose"],
    )
    
    # 基础参数（不参与搜索的参数）
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "enable_categorical": True,
        "tree_method": "hist",
    }
    
    # 执行搜索（使用训练集）
    search_results = tuner.search(X_train, y_train, base_params=base_params)
    
    # 步骤5: 保存结果
    print("\n【步骤5】保存搜索结果...")
    print("-" * 60)
    tuner.save_results(
        best_params_path=str(BEST_PARAMS_PATH),
        cv_results_path=str(CV_RESULTS_PATH),
    )
    
    # 步骤6: 显示Top结果
    print("\n【步骤6】Top 10 参数组合...")
    print("-" * 60)
    top_results = tuner.get_cv_results(top_n=10)
    print(top_results.to_string(index=False))
    
    # 步骤7: 性能比较
    print("\n【步骤7】性能比较...")
    print("-" * 60)
    comparison = tuner.compare_with_default(
        X_train,
        y_train,
        default_params=MODEL_PARAMS,
    )
    
    # 步骤8: 使用最佳参数在测试集上评估
    print("\n【步骤8】使用最佳参数在测试集上评估...")
    print("-" * 60)
    from utils.model_trainer import ModelTrainer
    
    # 合并基础参数和最佳参数
    best_params_full = {**base_params, **tuner.get_best_params()}
    
    # 训练模型
    trainer = ModelTrainer(params=best_params_full)
    trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        early_stopping_rounds=30,
    )
    
    # 评估
    print("\n测试集评估结果:")
    trainer.print_evaluation_report(X_test, y_test, "测试集")
    
    # 与默认参数对比
    print("\n默认参数测试集评估结果:")
    default_trainer = ModelTrainer(params=MODEL_PARAMS)
    default_trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        early_stopping_rounds=30,
    )
    default_trainer.print_evaluation_report(X_test, y_test, "测试集（默认参数）")
    
    # 最终摘要
    print("\n" + "=" * 60)
    print("超参数调优完成！")
    print("=" * 60)
    print("\n输出摘要:")
    print(f"  - 最佳参数: {BEST_PARAMS_PATH}")
    print(f"  - 交叉验证结果: {CV_RESULTS_PATH}")
    print(f"\n最佳参数组合:")
    for param, value in sorted(tuner.get_best_params().items()):
        print(f"  {param}: {value}")
    print("=" * 60)


if __name__ == '__main__':
    main()

