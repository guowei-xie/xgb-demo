"""
特征处理模块
包含数据预处理、特征工程等功能
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib
import os


class FeatureProcessor:
    """特征处理器类"""
    
    def __init__(self):
        """初始化特征处理器"""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted = False
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """
        对分类特征进行编码
        
        Args:
            df: 输入DataFrame
            categorical_cols: 分类特征列名列表
            
        Returns:
            编码后的DataFrame
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # 处理训练时未见过的类别
                    known_classes = set(self.label_encoders[col].classes_)
                    unknown_mask = ~df[col].isin(known_classes)
                    if unknown_mask.any():
                        # 将未知类别映射为最常见的类别
                        df_encoded.loc[unknown_mask, col] = 0
                    df_encoded[col] = self.label_encoders[col].transform(
                        df[col].where(~unknown_mask, df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].iloc[0])
                    )
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建新特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含新特征的DataFrame
        """
        df_featured = df.copy()
        
        # 房价收入比
        df_featured['price_income_ratio'] = df_featured['house_price'] / (df_featured['income'] + 1)
        
        # 年龄与教育年限的交互
        df_featured['age_education_ratio'] = df_featured['age'] / (df_featured['education_years'] + 1)
        
        # 访问频率（访问次数/距离上次访问天数）
        df_featured['visit_frequency'] = df_featured['visit_count'] / (df_featured['last_visit_days'] + 1)
        
        # 房价对数变换
        df_featured['house_price_log'] = np.log1p(df_featured['house_price'])
        
        # 收入对数变换
        df_featured['income_log'] = np.log1p(df_featured['income'])
        
        # 家庭人均收入
        df_featured['income_per_person'] = df_featured['income'] / df_featured['family_size']
        
        return df_featured
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_enrolled',
                     categorical_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        拟合并转换训练数据
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            categorical_cols: 分类特征列名列表
            
        Returns:
            处理后的特征DataFrame和目标Series
        """
        if categorical_cols is None:
            categorical_cols = ['city', 'grade']
        
        # 创建新特征
        df_processed = self.create_features(df)
        
        # 编码分类特征
        df_encoded = self.encode_categorical_features(df_processed, categorical_cols)
        
        # 选择特征列（排除目标列和ID列）
        exclude_cols = [target_col, 'user_id']
        self.feature_columns = [col for col in df_encoded.columns 
                               if col not in exclude_cols]
        
        X = df_encoded[self.feature_columns].copy()
        y = df[target_col]
        
        # 标准化数值特征
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        self.is_fitted = True
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_col: str = 'is_enrolled') -> pd.DataFrame:
        """
        转换新数据（使用已拟合的转换器）
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名（如果存在）
            
        Returns:
            处理后的特征DataFrame
        """
        if not self.is_fitted:
            raise ValueError("特征处理器尚未拟合，请先调用fit_transform")
        
        # 创建新特征
        df_processed = self.create_features(df)
        
        # 编码分类特征
        categorical_cols = list(self.label_encoders.keys())
        df_encoded = self.encode_categorical_features(df_processed, categorical_cols)
        
        # 选择特征列
        X = df_encoded[self.feature_columns].copy()
        
        # 标准化数值特征
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, random_state: int = 42, 
                  return_indices: bool = False) -> Tuple:
        """
        划分训练集和测试集
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            test_size: 测试集比例
            random_state: 随机种子
            return_indices: 是否返回测试集索引
            
        Returns:
            (X_train, X_test, y_train, y_test) 或 (X_train, X_test, y_train, y_test, test_indices)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if return_indices:
            # 返回测试集的原始索引
            test_indices = X_test.index
            return X_train, X_test, y_train, y_test, test_indices
        else:
            return X_train, X_test, y_train, y_test
    
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


if __name__ == '__main__':
    # 测试特征处理器
    from data_generator import generate_mock_data
    
    print("生成测试数据...")
    df = generate_mock_data(n_samples=1000)
    
    print("\n初始化特征处理器...")
    processor = FeatureProcessor()
    
    print("处理特征...")
    X, y = processor.fit_transform(df)
    
    print(f"\n特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    print(f"\n特征列: {X.columns.tolist()}")
    print(f"\n特征预览:")
    print(X.head())
    
    print("\n划分数据集...")
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

