"""
超参数调优模块

提供XGBoost模型的超参数搜索功能，支持随机搜索和网格搜索。
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import make_scorer, roc_auc_score
import joblib
import os
from typing import Dict, Any, List, Optional
import time


class HyperparameterTuner:
    """超参数调优器类"""
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        search_method: str = "random",
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ):
        """
        初始化超参数调优器
        
        Args:
            search_space: 超参数搜索空间字典，例如 {"max_depth": [4, 6, 8], "learning_rate": [0.01, 0.1]}
            search_method: 搜索方法，'random' 或 'grid'
            n_iter: 随机搜索迭代次数（仅random有效）
            cv: 交叉验证折数
            scoring: 评分指标，默认为 'roc_auc'
            n_jobs: 并行任务数，-1表示使用所有CPU
            random_state: 随机种子
            verbose: 详细程度，0-3
        """
        self.search_space = search_space
        self.search_method = search_method.lower()
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        if self.search_method not in ["random", "grid"]:
            raise ValueError("search_method 必须是 'random' 或 'grid'")
        
        self.search_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.cv_results_df_ = None
    
    def search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行超参数搜索
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            base_params: 基础模型参数（不参与搜索的参数）
        
        Returns:
            包含最佳参数和搜索结果的字典
        """
        print("=" * 60)
        print(f"开始超参数调优 - 方法: {self.search_method.upper()}")
        print("=" * 60)
        
        # 构建完整的参数空间
        if base_params is None:
            base_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        
        # 创建交叉验证对象
        cv_fold = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 创建评分器
        scorer = make_scorer(roc_auc_score, response_method='predict_proba')
        
        # 创建基础模型
        base_model = xgb.XGBClassifier(**base_params)
        
        # 执行搜索
        start_time = time.time()
        
        if self.search_method == "random":
            print(f"\n使用随机搜索，迭代次数: {self.n_iter}")
            search_cv = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.search_space,
                n_iter=self.n_iter,
                cv=cv_fold,
                scoring=scorer,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                return_train_score=True,
            )
        else:  # grid search
            print(f"\n使用网格搜索")
            total_combinations = self._count_combinations()
            print(f"总参数组合数: {total_combinations}")
            search_cv = GridSearchCV(
                estimator=base_model,
                param_grid=self.search_space,
                cv=cv_fold,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
            )
        
        print(f"\n开始搜索...")
        print(f"交叉验证折数: {self.cv}")
        print(f"评分指标: {self.scoring}")
        print("-" * 60)
        
        search_cv.fit(X, y)
        
        elapsed_time = time.time() - start_time
        
        # 保存结果
        self.search_results_ = search_cv
        self.best_params_ = search_cv.best_params_
        self.best_score_ = search_cv.best_score_
        self.best_model_ = search_cv.best_estimator_
        
        # 转换为DataFrame便于分析
        self.cv_results_df_ = pd.DataFrame(search_cv.cv_results_)
        
        # 打印结果摘要
        self._print_search_summary(elapsed_time)
        
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "best_model": self.best_model_,
            "cv_results": self.cv_results_df_,
            "search_cv": search_cv,
        }
    
    def _count_combinations(self) -> int:
        """
        计算网格搜索的总组合数
        
        Returns:
            总组合数
        """
        total = 1
        for param_values in self.search_space.values():
            total *= len(param_values)
        return total
    
    def _print_search_summary(self, elapsed_time: float):
        """
        打印搜索摘要
        
        Args:
            elapsed_time: 搜索耗时（秒）
        """
        print("\n" + "=" * 60)
        print("超参数搜索完成")
        print("=" * 60)
        print(f"\n搜索耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        print(f"\n最佳交叉验证得分 ({self.scoring}): {self.best_score_:.6f}")
        print(f"\n最佳参数:")
        for param, value in sorted(self.best_params_.items()):
            print(f"  {param}: {value}")
        
        # 显示Top 5参数组合
        if self.cv_results_df_ is not None and len(self.cv_results_df_) > 1:
            print(f"\nTop 5 参数组合:")
            top_results = self.cv_results_df_.nlargest(
                5, f"mean_test_score"
            )[["params", "mean_test_score", "std_test_score"]]
            
            for idx, row in top_results.iterrows():
                print(f"\n  排名 {len(top_results) - list(top_results.index).index(idx)}:")
                print(f"    得分: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
                print(f"    参数: {row['params']}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳参数
        
        Returns:
            最佳参数字典
        """
        if self.best_params_ is None:
            raise ValueError("尚未执行搜索，请先调用search方法")
        return self.best_params_.copy()
    
    def get_cv_results(self, top_n: int = 10) -> pd.DataFrame:
        """
        获取交叉验证结果
        
        Args:
            top_n: 返回前N个结果
        
        Returns:
            交叉验证结果DataFrame
        """
        if self.cv_results_df_ is None:
            raise ValueError("尚未执行搜索，请先调用search方法")
        
        # 选择相关列
        relevant_cols = [
            col for col in self.cv_results_df_.columns
            if any(x in col for x in ["param_", "mean_test_score", "std_test_score", "rank_test_score"])
        ]
        
        results = self.cv_results_df_[relevant_cols].copy()
        results = results.sort_values("mean_test_score", ascending=False)
        
        return results.head(top_n)
    
    def save_results(
        self,
        best_params_path: str,
        cv_results_path: str,
    ):
        """
        保存搜索结果
        
        Args:
            best_params_path: 最佳参数保存路径
            cv_results_path: 交叉验证结果CSV保存路径
        """
        if self.best_params_ is None:
            raise ValueError("尚未执行搜索，请先调用search方法")
        
        # 保存最佳参数
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        joblib.dump(self.best_params_, best_params_path)
        print(f"\n✓ 最佳参数已保存到: {best_params_path}")
        
        # 保存交叉验证结果
        if self.cv_results_df_ is not None:
            os.makedirs(os.path.dirname(cv_results_path), exist_ok=True)
            self.cv_results_df_.to_csv(cv_results_path, index=False)
            print(f"✓ 交叉验证结果已保存到: {cv_results_path}")
    
    def compare_with_default(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        default_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        比较最佳参数与默认参数的性能
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            default_params: 默认参数字典
        
        Returns:
            包含比较结果的字典
        """
        if self.best_model_ is None:
            raise ValueError("尚未执行搜索，请先调用search方法")
        
        print("\n" + "=" * 60)
        print("最佳参数 vs 默认参数性能比较")
        print("=" * 60)
        
        # 使用最佳参数评估
        best_scores = cross_val_score(
            self.best_model_,
            X,
            y,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            scoring=make_scorer(roc_auc_score, response_method='predict_proba'),
            n_jobs=self.n_jobs,
        )
        
        # 使用默认参数评估
        # 确保默认参数包含必要的配置（如enable_categorical等）
        default_params_complete = default_params.copy()
        default_params_complete.setdefault("enable_categorical", True)
        default_params_complete.setdefault("tree_method", "hist")
        default_model = xgb.XGBClassifier(**default_params_complete)
        default_scores = cross_val_score(
            default_model,
            X,
            y,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            scoring=make_scorer(roc_auc_score, response_method='predict_proba'),
            n_jobs=self.n_jobs,
        )
        
        print(f"\n默认参数交叉验证得分:")
        print(f"  均值: {default_scores.mean():.6f}")
        print(f"  标准差: {default_scores.std():.6f}")
        
        print(f"\n最佳参数交叉验证得分:")
        print(f"  均值: {best_scores.mean():.6f}")
        print(f"  标准差: {best_scores.std():.6f}")
        
        improvement = best_scores.mean() - default_scores.mean()
        improvement_pct = (improvement / default_scores.mean()) * 100
        
        print(f"\n性能提升:")
        print(f"  绝对提升: {improvement:.6f}")
        print(f"  相对提升: {improvement_pct:.2f}%")
        
        return {
            "default_mean": default_scores.mean(),
            "default_std": default_scores.std(),
            "best_mean": best_scores.mean(),
            "best_std": best_scores.std(),
            "improvement": improvement,
            "improvement_pct": improvement_pct,
        }
