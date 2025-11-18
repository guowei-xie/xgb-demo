"""
效果分析模块
包含模型效果可视化分析功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os
from typing import Optional, Tuple
import warnings

def setup_chinese_font():
    """自动设置中文字体，优先常用字体，否则自动尝试包含中文的字体"""
    preferred_fonts = [
        'Songti SC', 'STSong', 'SimHei', 'STHeiti',
        'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS'
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    for font in available_fonts:
        if any(k in font for k in ['Song', 'Hei', 'PingFang', 'Hiragino', 'ST', 'Sim']):
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            warnings.warn(f"使用备选中文字体: {font}")
            return font
    warnings.warn("未找到合适的中文字体，中文显示可能异常。建议安装中文字体。")
    plt.rcParams['axes.unicode_minus'] = False
    return None

_chinese_font = setup_chinese_font()


class ModelAnalyzer:
    """模型分析器类"""
    
    def __init__(self, output_dir: str = 'results'):
        """
        初始化分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        # 确保字体设置在seaborn设置之后仍然生效
        if _chinese_font:
            plt.rcParams['font.sans-serif'] = [_chinese_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 15, figsize: Tuple[int, int] = (10, 8)):
        """
        绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性DataFrame
            top_n: 显示前N个特征
            figsize: 图片大小
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        # 使用color参数替代palette，避免seaborn版本警告
        sns.barplot(data=top_features, y='feature', x='importance', color='steelblue')
        plt.title(f'特征重要性 Top {top_n}', fontsize=16, fontweight='bold')
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {filepath}")
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      figsize: Tuple[int, int] = (8, 6)):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            figsize: 图片大小
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = np.trapz(tpr, fpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {filepath}")
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   figsize: Tuple[int, int] = (8, 6)):
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            figsize: 图片大小
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'PR曲线 (AUC = {pr_auc:.4f})')
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('精确率-召回率曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"PR曲线图已保存到: {filepath}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             figsize: Tuple[int, int] = (8, 6)):
        """
        绘制混淆矩阵热力图
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            figsize: 图片大小
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['未报名', '已报名'],
                   yticklabels=['未报名', '已报名'],
                   cbar_kws={'label': '数量'})
        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {filepath}")
        plt.close()
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    figsize: Tuple[int, int] = (10, 6)):
        """
        绘制预测概率分布图
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            figsize: 图片大小
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 按类别绘制概率分布
        for label in [0, 1]:
            mask = y_true == label
            label_name = '已报名' if label == 1 else '未报名'
            axes[0].hist(y_pred_proba[mask], bins=30, alpha=0.6, 
                        label=label_name, density=True)
        
        axes[0].set_xlabel('预测概率', fontsize=12)
        axes[0].set_ylabel('密度', fontsize=12)
        axes[0].set_title('预测概率分布', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        data_for_box = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
        axes[1].boxplot(data_for_box, labels=['未报名', '已报名'])
        axes[1].set_ylabel('预测概率', fontsize=12)
        axes[1].set_title('预测概率箱线图', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'prediction_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"预测概率分布图已保存到: {filepath}")
        plt.close()
    
    def plot_data_exploration(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
        """
        绘制数据探索性分析图
        
        Args:
            df: 原始数据DataFrame
            figsize: 图片大小
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 报名率按城市分布
        city_enroll = df.groupby('city')['is_enrolled'].mean().sort_values(ascending=False)
        axes[0, 0].barh(range(len(city_enroll)), city_enroll.values)
        axes[0, 0].set_yticks(range(len(city_enroll)))
        axes[0, 0].set_yticklabels(city_enroll.index)
        axes[0, 0].set_xlabel('报名率', fontsize=10)
        axes[0, 0].set_title('各城市报名率', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. 报名率按年级分布
        grade_enroll = df.groupby('grade')['is_enrolled'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(grade_enroll)), grade_enroll.values, color='steelblue')
        axes[0, 1].set_xticks(range(len(grade_enroll)))
        axes[0, 1].set_xticklabels(grade_enroll.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('报名率', fontsize=10)
        axes[0, 1].set_title('各年级报名率', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 房价分布
        axes[0, 2].hist(df['house_price'], bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 2].set_xlabel('房价', fontsize=10)
        axes[0, 2].set_ylabel('频数', fontsize=10)
        axes[0, 2].set_title('房价分布', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. 年龄分布
        axes[1, 0].hist(df['age'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('年龄', fontsize=10)
        axes[1, 0].set_ylabel('频数', fontsize=10)
        axes[1, 0].set_title('年龄分布', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. 报名率按是否有车
        car_enroll = df.groupby('has_car')['is_enrolled'].mean()
        axes[1, 1].bar(['无车', '有车'], car_enroll.values, color=['salmon', 'lightblue'])
        axes[1, 1].set_ylabel('报名率', fontsize=10)
        axes[1, 1].set_title('是否有车对报名率的影响', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. 整体报名率
        enroll_rate = df['is_enrolled'].mean()
        axes[1, 2].pie([1-enroll_rate, enroll_rate], 
                      labels=['未报名', '已报名'],
                      autopct='%1.1f%%',
                      startangle=90,
                      colors=['lightcoral', 'lightblue'])
        axes[1, 2].set_title('整体报名率分布', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'data_exploration.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"数据探索图已保存到: {filepath}")
        plt.close()
    
    def save_prediction_results(self, X_test: pd.DataFrame, y_test: pd.Series,
                               y_pred: np.ndarray, y_pred_proba: np.ndarray,
                               df_original: pd.DataFrame = None,
                               test_indices: pd.Index = None,
                               filepath: str = None):
        """
        保存测试集预测结果到CSV表格
        
        Args:
            X_test: 测试集特征
            y_test: 测试集真实标签
            y_pred: 测试集预测标签
            y_pred_proba: 测试集预测概率（正类概率）
            df_original: 原始数据DataFrame（可选，用于添加原始特征）
            test_indices: 测试集在原始数据中的索引（可选）
            filepath: 保存路径（可选）
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'test_predictions.csv')
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            '真实标签': y_test.values,
            '预测标签': y_pred,
            '预测概率': y_pred_proba,
            '预测正确': (y_test.values == y_pred).astype(int)
        })
        
        # 如果提供了原始数据和测试集索引，添加原始特征
        if df_original is not None and test_indices is not None:
            try:
                # 确保索引在原始数据范围内
                valid_indices = test_indices[test_indices < len(df_original)]
                if len(valid_indices) == len(X_test):
                    # 添加原始特征列
                    for col in ['user_id', 'city', 'grade', 'age', 'house_price', 
                               'income', 'education_years', 'family_size', 
                               'has_car', 'visit_count', 'last_visit_days']:
                        if col in df_original.columns:
                            results_df[col] = df_original.loc[valid_indices, col].values
            except Exception as e:
                # 如果无法匹配，只保存预测结果
                pass
        elif df_original is not None:
            # 如果没有提供索引，尝试使用X_test的索引
            try:
                if X_test.index.is_unique:
                    original_indices = X_test.index
                    valid_indices = original_indices[original_indices < len(df_original)]
                    if len(valid_indices) == len(X_test):
                        for col in ['user_id', 'city', 'grade', 'age', 'house_price', 
                                   'income', 'education_years', 'family_size', 
                                   'has_car', 'visit_count', 'last_visit_days']:
                            if col in df_original.columns:
                                results_df[col] = df_original.loc[valid_indices, col].values
            except Exception:
                pass
        
        # 重新排列列顺序，将预测相关列放在前面
        priority_cols = ['真实标签', '预测标签', '预测概率', '预测正确']
        other_cols = [col for col in results_df.columns if col not in priority_cols]
        results_df = results_df[priority_cols + other_cols]
        
        # 保存到CSV
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"测试集预测结果表格已保存到: {filepath}")
        print(f"  共 {len(results_df)} 条记录")
        print(f"  预测准确率: {results_df['预测正确'].mean():.2%}")
        
        # 显示前几行预览
        print(f"\n预测结果预览（前5行）:")
        print(results_df.head().to_string(index=False))
    
    def generate_full_report(self, trainer, processor, X_train, y_train, 
                            X_test, y_test, df_original: pd.DataFrame,
                            test_indices: pd.Index = None):
        """
        生成完整的分析报告
        
        Args:
            trainer: 模型训练器对象
            processor: 特征处理器对象
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            df_original: 原始数据DataFrame
            test_indices: 测试集在原始数据中的索引（可选）
        """
        print("\n" + "="*50)
        print("开始生成分析报告...")
        print("="*50)
        
        # 1. 特征重要性
        feature_importance = trainer.get_feature_importance(20)
        self.plot_feature_importance(feature_importance)
        
        # 2. ROC曲线
        y_test_proba = trainer.predict_proba(X_test)[:, 1]
        self.plot_roc_curve(y_test, y_test_proba)
        
        # 3. PR曲线
        self.plot_precision_recall_curve(y_test, y_test_proba)
        
        # 4. 混淆矩阵
        y_test_pred = trainer.predict(X_test)
        self.plot_confusion_matrix(y_test, y_test_pred)
        
        # 5. 预测概率分布
        self.plot_prediction_distribution(y_test, y_test_proba)
        
        # 6. 数据探索
        self.plot_data_exploration(df_original)
        
        # 7. 保存预测结果表格
        self.save_prediction_results(X_test, y_test, y_test_pred, y_test_proba, 
                                    df_original, test_indices)
        
        print("\n" + "="*50)
        print("分析报告生成完成！")
        print("="*50)


if __name__ == '__main__':
    from data_generator import generate_mock_data
    from feature_processor import FeatureProcessor
    from model_trainer import ModelTrainer
    
    # 生成数据
    print("生成测试数据...")
    df = generate_mock_data(n_samples=5000)
    
    # 处理特征
    print("\n处理特征...")
    processor = FeatureProcessor()
    X, y = processor.fit_transform(df)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # 训练模型
    print("\n训练模型...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, X_test, y_test)
    
    # 生成分析报告
    analyzer = ModelAnalyzer()
    analyzer.generate_full_report(trainer, processor, X_train, y_train, 
                                 X_test, y_test, df)

