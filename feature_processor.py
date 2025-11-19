"""
特征处理模块
包含数据预处理、特征工程等功能
适配实际数据：保留缺失值，XGBoost兼容的特征处理
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import joblib
import os


class FeatureProcessor:
    """特征处理器类"""
    
    def __init__(self):
        """
        初始化特征处理器
        注意：XGBoost可以处理类别特征和缺失值，因此不需要标准化和编码
        """
        self.feature_columns: List[str] = []
        self.is_fitted = False
        # 保存类别特征的CategoricalDtype，用于确保训练和预测时使用相同的类别定义
        self.categorical_dtypes: dict = {}
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建新特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含新特征的DataFrame
        """
        df_featured = df.copy()

        feature_recipes = [
            (['house_price'], 'house_price_log', lambda data: data['house_price'].apply(
                lambda x: np.log1p(x) if pd.notna(x) else np.nan
            )),
            (
                ['city_score', 'house_price'],
                'city_score_house_price_ratio',
                lambda data: data['city_score'] / (data['house_price'] + 1),
            ),
            (
                ['fns_cnt', 'grade'],
                'fns_cnt_per_grade',
                lambda data: data['fns_cnt'] / (data['grade'] + 1),
            ),
            (
                ['is_enable', 'fns_cnt'],
                'is_enable_fns_interaction',
                lambda data: data['is_enable'] * data['fns_cnt'],
            ),
            (
                ['refresh_num', 'fns_cnt'],
                'refresh_fns_interaction',
                lambda data: data['refresh_num'] * data['fns_cnt'],
            ),
        ]

        for required_cols, new_col, builder in feature_recipes:
            if all(col in df_featured.columns for col in required_cols):
                df_featured[new_col] = builder(df_featured)

        return df_featured
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'is_renewal') -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征数据（保留缺失值，不进行编码和标准化）
        将类别特征转换为category类型，以便XGBoost自动识别
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            
        Returns:
            处理后的特征DataFrame和目标Series
        """
        # 创建新特征
        df_processed = self.create_features(df)
        
        # 选择特征列（排除目标列、ID列和其他非特征列）
        exclude_cols = [
            target_col, 
            'user_id', 
            'b2c_term_name',  # 学期名称，非特征
            'l1_term_name',   # 学期名称，非特征
            'l1_term_renewal_end_date'  # 日期字段，非特征
        ]
        
        # 获取所有特征列
        all_feature_cols = [col for col in df_processed.columns 
                           if col not in exclude_cols]
        
        # 如果尚未拟合，保存特征列列表
        if not self.is_fitted:
            self.feature_columns = all_feature_cols
        
        # 选择特征列（确保列顺序一致）
        X = df_processed[self.feature_columns].copy()
        
        X = self._apply_categorical_types(X)
        X = self._sanitize_numeric(X)
        
        # 提取目标变量
        y = df[target_col] if target_col in df.columns else None
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_renewal') -> Tuple[pd.DataFrame, pd.Series]:
        """
        拟合并转换训练数据
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            
        Returns:
            处理后的特征DataFrame和目标Series
        """
        X, y = self.prepare_features(df, target_col)
        
        self.is_fitted = True
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_col: str = 'is_renewal') -> pd.DataFrame:
        """
        转换新数据（使用已拟合的特征列）
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名（如果存在，仅用于排除）
            
        Returns:
            处理后的特征DataFrame
        """
        if not self.is_fitted:
            raise ValueError("特征处理器尚未拟合，请先调用fit_transform")
        
        # 创建新特征
        df_processed = self.create_features(df)
        
        # 选择特征列（使用已保存的特征列列表）
        X = df_processed[self.feature_columns].copy()
        
        # 如果某些特征列在新数据中不存在，添加缺失值列
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = np.nan
        
        # 确保列顺序一致
        X = X[self.feature_columns]
        
        X = self._apply_categorical_types(X)
        X = self._sanitize_numeric(X)
        return X
    
    def save(self, filepath: str = 'models/feature_processor.pkl'):
        """
        保存特征处理器
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"特征处理器已保存到: {filepath}")
    
    @staticmethod
    def load(filepath: str = 'models/feature_processor.pkl') -> 'FeatureProcessor':
        """
        加载特征处理器
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的特征处理器对象
        """
        return joblib.load(filepath)

    def _apply_categorical_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        设置类别特征的dtype，确保训练与预测阶段一致。
        """
        categorical_features = ['city', 'district', 'city_level', 'device']
        for col in categorical_features:
            if col not in X.columns:
                continue
            if X[col].dtype == 'object':
                if not self.is_fitted:
                    unique_cats = X[col].dropna().unique().tolist()
                    cat_dtype = pd.CategoricalDtype(categories=unique_cats, ordered=False)
                    self.categorical_dtypes[col] = cat_dtype
                    X[col] = X[col].astype(cat_dtype)
                else:
                    if col in self.categorical_dtypes:
                        known_dtype = self.categorical_dtypes[col]
                        X[col] = X[col].astype(known_dtype)
                    else:
                        X[col] = X[col].astype('category')
            elif X[col].dtype.name == 'category' and col in self.categorical_dtypes:
                X[col] = X[col].astype(self.categorical_dtypes[col])
        return X

    def _sanitize_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        统一处理数值特征中的正负无穷值。
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(X[col]).any():
                inf_count = np.isinf(X[col]).sum()
                print(f"警告: 特征 {col} 包含 {inf_count} 个inf值，将替换为NaN")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        return X