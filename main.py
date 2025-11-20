"""
XGBoost模型训练主程序
演示完整的机器学习流程：数据读取 -> 特征处理 -> 模型训练 -> 效果分析
"""
import os
import pandas as pd
from utils.feature_processor import FeatureProcessor
from utils.model_trainer import ModelTrainer
from utils.level_tagger import assign_level_tags
from utils.prediction_analysis import (
    analyze_prediction_quality,
    plot_level_performance_by_term,
    plot_level_ratio_trend,
    plot_term_renewal_trend,
    plot_feature_importance,
)
from config import (
    DATA_PATH,
    PROCESSOR_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TERM_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PARAMS,
    EARLY_STOPPING_ROUNDS,
    TRAIN_TEST_PREDICTIONS,
    TRAIN_TEST_ANALYSIS_FIG,
    TRAIN_TEST_LEVEL_ANALYSIS_FIG,
    TRAIN_TEST_LEVEL_RATIO_FIG,
    TRAIN_TEST_TERM_TREND_FIG,
    FEATURE_IMPORTANCE_FIG,
    LEVEL_TAG_RULES,
    FEATURE_INCLUDE_COLUMNS,
    FEATURE_EXCLUDE_COLUMNS,
)
from utils.pipeline_utils import load_dataset, log_basic_stats, split_dataset


def main():
    """主函数：执行完整的模型训练流程"""
    print("="*60)
    print("XGBoost用户续报率预测模型训练")
    print("="*60)
    
    print("\n【步骤1】读取实际数据...")
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
    print(f"✓ 特征列表: {', '.join(X.columns.tolist()[:10])}...")
    
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✓ 数据集划分完成")
    print(f"  - 训练集: {X_train.shape[0]} 条 (续报率: {y_train.mean():.2%})")
    print(f"  - 测试集: {X_test.shape[0]} 条 (续报率: {y_test.mean():.2%})")
    
    # 保存特征处理器
    processor.save(str(PROCESSOR_PATH))
    
    # 步骤3: 模型训练
    print("\n【步骤3】XGBoost模型训练...")
    print("-" * 60)
    
    trainer = ModelTrainer(params=MODEL_PARAMS)
    trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    
    # 保存模型
    trainer.save(str(MODEL_PATH))
    
    # 步骤4: 模型评估
    print("\n【步骤4】模型性能评估...")
    print("-" * 60)
    trainer.print_evaluation_report(X_train, y_train, "训练集")
    trainer.print_evaluation_report(X_test, y_test, "测试集")
    
    # 显示特征重要性
    print("\n特征重要性 Top 15:")
    print(trainer.get_feature_importance(15).to_string(index=False))
    
    # 可视化特征重要性
    print("\n【步骤4.5】生成特征重要性可视化...")
    print("-" * 60)
    feature_importance_df = trainer.get_feature_importance(top_n=30)
    plot_feature_importance(
        feature_importance_df=feature_importance_df,
        output_path=FEATURE_IMPORTANCE_FIG,
        top_n=30,
        title="XGBoost模型特征重要性分析",
    )
    
    # 步骤5: 保存测试集预测结果
    print("\n【步骤5】保存测试集预测结果...")
    print("-" * 60)
    
    # 获取预测结果
    test_indices = X_test.index
    y_test_proba = trainer.predict_proba(X_test)
    y_test_pred = trainer.predict(X_test)
    
    # 从原始数据中提取测试集对应的行
    test_df_original = df.loc[test_indices].copy()
    
    # 合并预测结果（格式与predict_test.py一致）
    result_df = test_df_original.copy()
    result_df["pred_probability"] = y_test_proba
    result_df["pred_label"] = y_test_pred
    result_df["level_tag"] = assign_level_tags(
        result_df["pred_probability"],
        LEVEL_TAG_RULES,
    )
    
    # 保存到results目录
    os.makedirs(TRAIN_TEST_PREDICTIONS.parent, exist_ok=True)
    result_df.to_csv(TRAIN_TEST_PREDICTIONS, index=False)
    print(f"✓ 测试集预测结果已保存到: {TRAIN_TEST_PREDICTIONS}")
    print(f"  - 共 {len(result_df)} 条记录")

    print("\n【步骤6】预测结果分析...")
    print("-" * 60)
    proba_series = pd.Series(y_test_proba, index=X_test.index, name="pred_probability")
    y_test_aligned = y_test.loc[X_test.index]
    analyze_prediction_quality(
        probabilities=proba_series,
        labels=y_test_aligned,
        output_path=TRAIN_TEST_ANALYSIS_FIG,
        dataset_name="验证集",
    )
    if TERM_COLUMN in result_df.columns:
        plot_level_performance_by_term(
            df=result_df,
            term_col=TERM_COLUMN,
            level_col="level_tag",
            label_col=TARGET_COLUMN,
            output_path=TRAIN_TEST_LEVEL_ANALYSIS_FIG,
            dataset_name="验证集",
        )
        plot_level_ratio_trend(
            df=result_df,
            term_col=TERM_COLUMN,
            level_col="level_tag",
            output_path=TRAIN_TEST_LEVEL_RATIO_FIG,
            dataset_name="验证集",
        )
        plot_term_renewal_trend(
            df=result_df,
            term_col=TERM_COLUMN,
            prob_col="pred_probability",
            label_col=TARGET_COLUMN,
            output_path=TRAIN_TEST_TERM_TREND_FIG,
            dataset_name="验证集",
        )
    else:
        print(f"⚠️ 数据缺少列 {TERM_COLUMN}，跳过等级-学期散点图。")
    
    print("\n" + "="*60)
    print("模型训练流程完成！")
    print("="*60)
    print("\n输出摘要:")
    print(f"  - 特征处理器: {PROCESSOR_PATH}")
    print(f"  - 训练模型: {MODEL_PATH}")
    print(f"  - 测试集预测结果: {TRAIN_TEST_PREDICTIONS}")
    print(f"  - 预测概率分析图: {TRAIN_TEST_ANALYSIS_FIG}")
    print(f"  - 等级-学期散点图: {TRAIN_TEST_LEVEL_ANALYSIS_FIG}")
    print(f"  - 等级占比趋势图: {TRAIN_TEST_LEVEL_RATIO_FIG}")
    print(f"  - 特征重要性图: {FEATURE_IMPORTANCE_FIG}")
    if TERM_COLUMN in result_df.columns:
        print(f"  - 学期续报率走势图: {TRAIN_TEST_TERM_TREND_FIG}")
    print("="*60)


if __name__ == '__main__':
    main()
