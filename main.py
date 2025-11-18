"""
XGBoost模型训练主程序
演示完整的机器学习流程：数据生成 -> 特征处理 -> 模型训练 -> 效果分析
"""
import os
import sys
from data_generator import generate_mock_data, save_data
from feature_processor import FeatureProcessor
from model_trainer import ModelTrainer
from analyzer import ModelAnalyzer


def main():
    """主函数：执行完整的模型训练流程"""
    print("="*60)
    print("XGBoost用户报名率预测模型训练")
    print("="*60)
    
    # 步骤1: 生成MOCK数据
    print("\n【步骤1】生成MOCK用户数据...")
    print("-" * 60)
    df = generate_mock_data(n_samples=10000, random_state=42)
    print(f"✓ 数据生成完成，共 {len(df)} 条记录")
    print(f"✓ 报名率: {df['is_enrolled'].mean():.2%}")
    print(f"✓ 数据列: {', '.join(df.columns.tolist())}")
    
    # 保存原始数据
    save_data(df, 'data/mock_data.csv')
    
    # 步骤2: 特征处理
    print("\n【步骤2】特征处理与工程...")
    print("-" * 60)
    processor = FeatureProcessor()
    X, y = processor.fit_transform(df)
    print(f"✓ 特征处理完成")
    print(f"✓ 特征数量: {X.shape[1]}")
    print(f"✓ 特征列表: {', '.join(X.columns.tolist()[:10])}...")
    
    # 划分训练集和测试集（获取测试集索引以便关联原始数据）
    X_train, X_test, y_train, y_test, test_indices = processor.split_data(
        X, y, test_size=0.2, return_indices=True
    )
    print(f"✓ 数据集划分完成")
    print(f"  - 训练集: {X_train.shape[0]} 条")
    print(f"  - 测试集: {X_test.shape[0]} 条")
    
    # 保存特征处理器
    processor.save('models/feature_processor.pkl')
    
    # 步骤3: 模型训练
    print("\n【步骤3】XGBoost模型训练...")
    print("-" * 60)
    
    # 设置模型参数
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    trainer = ModelTrainer(params=model_params)
    trainer.train(X_train, y_train, X_test, y_test, early_stopping_rounds=20)
    
    # 保存模型
    trainer.save('models/xgb_model.pkl')
    
    # 步骤4: 模型评估
    print("\n【步骤4】模型性能评估...")
    print("-" * 60)
    trainer.print_evaluation_report(X_train, y_train, "训练集")
    trainer.print_evaluation_report(X_test, y_test, "测试集")
    
    # 显示特征重要性
    print("\n特征重要性 Top 10:")
    print(trainer.get_feature_importance(10).to_string(index=False))
    
    # 步骤5: 效果分析与可视化
    print("\n【步骤5】生成效果分析报告...")
    print("-" * 60)
    analyzer = ModelAnalyzer(output_dir='results')
    analyzer.generate_full_report(
        trainer=trainer,
        processor=processor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        df_original=df,
        test_indices=test_indices
    )
    
    # 总结
    print("\n" + "="*60)
    print("模型训练流程完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - 数据文件: data/mock_data.csv")
    print("  - 特征处理器: models/feature_processor.pkl")
    print("  - 训练模型: models/xgb_model.pkl")
    print("  - 分析图表: results/ 目录下")
    print("\n主要分析图表:")
    print("  - feature_importance.png: 特征重要性")
    print("  - roc_curve.png: ROC曲线")
    print("  - precision_recall_curve.png: PR曲线")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - prediction_distribution.png: 预测概率分布")
    print("  - data_exploration.png: 数据探索分析")
    print("  - test_predictions.csv: 测试集预测结果表格")
    print("="*60)


if __name__ == '__main__':
    main()
