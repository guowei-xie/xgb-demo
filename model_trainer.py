"""
模型训练模块
包含XGBoost模型训练、评估等功能
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import callback
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import os
from typing import Dict, Any


class ModelTrainer:
    """XGBoost模型训练器类"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化模型训练器
        
        Args:
            params: XGBoost模型参数
        """
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
        self.params = params
        self.model = None
        self.feature_importance = None
        self.threshold = self.params.get("threshold", 0.5)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              early_stopping_rounds: int = 10) -> 'ModelTrainer':
        """
        训练XGBoost模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            early_stopping_rounds: 早停轮数

        Returns:
            self
        """
        print("开始训练XGBoost模型...")

        self.model = self._build_model(use_valid=X_val is not None and y_val is not None,
                                       early_stopping_rounds=early_stopping_rounds)
        fit_kwargs = self._build_fit_kwargs(X_train, y_train, X_val, y_val)
        self.model.fit(**fit_kwargs)

        self._cache_feature_importance(X_train.columns)

        print("模型训练完成！")
        return self
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征DataFrame
            threshold: 概率阈值，如果为None则使用self.threshold（默认0.5）
            
        Returns:
            预测类别数组
        """
        self._ensure_trained()
        used_threshold = threshold if threshold is not None else self.threshold
        y_proba = self.predict_proba(X)
        return (y_proba >= used_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测正类概率
        
        Args:
            X: 特征DataFrame
            
        Returns:
            正类概率一维数组
        """
        self._ensure_trained()
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征DataFrame
            y: 真实标签
            threshold: 概率阈值，如果为None则使用self.threshold（默认0.5）
            
        Returns:
            评估指标字典
        """
        self._ensure_trained()
        y_pred = self.predict(X, threshold=threshold)
        y_pred_proba = self.predict_proba(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
        }
    
    def print_evaluation_report(self, X: pd.DataFrame, y: pd.Series, 
                                dataset_name: str = "数据集", threshold: float = None):
        """
        打印评估报告
        
        Args:
            X: 特征DataFrame
            y: 真实标签
            dataset_name: 数据集名称
            threshold: 概率阈值，如果为None则使用self.threshold（默认0.5）
        """
        print(f"\n{'='*50}")
        print(f"{dataset_name}评估报告")
        print(f"{'='*50}")
        
        # 显示使用的阈值
        used_threshold = threshold if threshold is not None else self.threshold
        print(f"\n使用的概率阈值: {used_threshold:.4f}")
        
        metrics = self.evaluate(X, y, threshold=threshold)
        
        print(f"\n评估指标:")
        print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"  F1分数:             {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        
        # 混淆矩阵
        y_pred = self.predict(X, threshold=threshold)
        cm = confusion_matrix(y, y_pred)
        print(f"\n混淆矩阵:")
        print(cm)
    
    def set_threshold(self, threshold: float):
        """
        设置概率阈值
        
        Args:
            threshold: 概率阈值（0到1之间）
        """
        if not 0 <= threshold <= 1:
            raise ValueError("阈值必须在0到1之间")
        self.threshold = threshold
        print(f"概率阈值已设置为: {threshold:.4f}")
    
    def get_threshold(self) -> float:
        """
        获取当前概率阈值
        
        Returns:
            当前概率阈值
        """
        return self.threshold
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            top_n: 返回前N个重要特征
            
        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str = 'models/xgb_model.pkl'):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"模型已保存到: {filepath}")
    
    @staticmethod
    def load(filepath: str = 'models/xgb_model.pkl') -> 'ModelTrainer':
        """
        加载模型
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的模型训练器对象
        """
        trainer = ModelTrainer()
        trainer.model = joblib.load(filepath)
        return trainer

    def _ensure_trained(self) -> None:
        """
        确保模型已训练，避免误调用。
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")

    def _build_model(self, use_valid: bool, early_stopping_rounds: int) -> xgb.XGBClassifier:
        """
        创建XGBoost模型实例，自动注入常用配置。
        """
        model_params = self.params.copy()
        model_params.setdefault("enable_categorical", True)
        model_params.setdefault("tree_method", "hist")
        if use_valid:
            model_params["callbacks"] = [
                callback.EarlyStopping(rounds=early_stopping_rounds)
            ]
        return xgb.XGBClassifier(**model_params)

    def _build_fit_kwargs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> Dict[str, Any]:
        """
        根据是否提供验证集构建fit参数。
        """
        fit_kwargs: Dict[str, Any] = {"X": X_train, "y": y_train}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        return fit_kwargs

    def _cache_feature_importance(self, feature_names: pd.Index) -> None:
        """
        快速缓存特征重要性，便于后续分析。
        """
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)