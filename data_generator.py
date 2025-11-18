"""
数据生成模块
用于生成MOCK用户数据，包含城市、年级、房价等特征，以及是否报名的标签
"""
import pandas as pd
import numpy as np
from typing import Tuple


def generate_mock_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    生成MOCK用户数据
    
    Args:
        n_samples: 生成样本数量，默认10000
        random_state: 随机种子，默认42
        
    Returns:
        包含用户特征和标签的DataFrame
    """
    np.random.seed(random_state)
    
    # 城市列表
    cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '苏州']
    
    # 年级列表
    grades = ['小学', '初中', '高中', '大学', '研究生', '在职']
    
    # 生成基础特征
    data = {
        'user_id': range(1, n_samples + 1),
        'city': np.random.choice(cities, n_samples),
        'grade': np.random.choice(grades, n_samples),
        'age': np.random.randint(18, 50, n_samples),
        'house_price': np.random.lognormal(mean=5.0, sigma=0.8, size=n_samples).round(2),
        'income': np.random.lognormal(mean=4.5, sigma=0.7, size=n_samples).round(2),
        'education_years': np.random.randint(9, 22, n_samples),
        'family_size': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'has_car': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'visit_count': np.random.poisson(lam=3, size=n_samples),
        'last_visit_days': np.random.exponential(scale=30, size=n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # 生成标签：是否报名（基于特征的概率分布）
    # 目标：报名率约15%，强依赖核心特征（收入、房价、年级、访问次数）
    
    # 城市评分：一线城市略高，但影响较小
    city_scores = {'北京': 0.3, '上海': 0.35, '广州': 0.2, '深圳': 0.3, 
                   '杭州': 0.25, '成都': 0.15, '武汉': 0.1, '西安': 0.08, 
                   '南京': 0.18, '苏州': 0.15}
    
    # 年级评分：大学和研究生明显更高（强依赖特征1）
    grade_scores = {'小学': 0.0, '初中': 0.0, '高中': 0.1, '大学': 0.8, 
                    '研究生': 0.9, '在职': 0.2}
    
    # 归一化特征值到0-1范围
    city_score = df['city'].map(city_scores)
    grade_score = df['grade'].map(grade_scores)
    age_norm = (df['age'] - 18) / 32  # 18-50岁归一化
    house_price_norm = np.clip(np.log1p(df['house_price']) / 8, 0, 1)  # 房价对数归一化
    income_norm = np.clip(np.log1p(df['income']) / 7, 0, 1)  # 收入对数归一化
    education_norm = (df['education_years'] - 9) / 13  # 9-22年归一化
    family_norm = (df['family_size'] - 1) / 4  # 1-5归一化
    visit_norm = np.clip(df['visit_count'] / 10, 0, 1)  # 访问次数归一化
    last_visit_norm = np.exp(-df['last_visit_days'] / 20)  # 最近访问天数（越小越好）
    
    # 定义强依赖特征的阈值（只有满足条件才更容易报名）
    # 强依赖特征1：年级 - 必须是大学或研究生
    grade_strong = (grade_score >= 0.8).astype(float)
    
    # 强依赖特征2：收入 - 需要达到中高水平（阈值0.5）
    income_strong = (income_norm >= 0.5).astype(float)
    
    # 强依赖特征3：房价 - 需要达到中高水平（阈值0.5）
    house_price_strong = (house_price_norm >= 0.5).astype(float)
    
    # 强依赖特征4：访问次数 - 需要多次访问（阈值0.3，即至少3次）
    visit_strong = (visit_norm >= 0.3).astype(float)
    
    # 计算强依赖特征组合得分（必须同时满足多个条件）
    # 核心逻辑：只有同时满足多个强依赖条件，报名概率才高
    strong_features_score = (
        0.35 * grade_strong +              # 年级（强依赖1）- 权重最高
        0.25 * income_strong +             # 收入（强依赖2）
        0.20 * house_price_strong +        # 房价（强依赖3）
        0.20 * visit_strong                # 访问次数（强依赖4）
    )
    
    # 计算其他特征的辅助得分（影响较小）
    other_features_score = (
        0.05 * city_score +                # 城市影响（降低）
        0.03 * age_norm +                  # 年龄影响（降低）
        0.02 * education_norm +            # 教育年限影响（降低）
        0.02 * family_norm +               # 家庭规模影响（降低）
        0.02 * df['has_car'] +             # 是否有车（降低）
        0.03 * last_visit_norm             # 最近访问影响（降低）
    )
    
    # 添加强依赖特征的交互效应（同时满足多个条件时效果叠加）
    interaction_score = (
        0.15 * grade_strong * income_strong +                    # 年级×收入
        0.12 * grade_strong * house_price_strong +              # 年级×房价
        0.10 * income_strong * house_price_strong +             # 收入×房价
        0.08 * grade_strong * visit_strong +                    # 年级×访问
        0.05 * income_strong * house_price_strong * visit_strong  # 收入×房价×访问
    )
    
    # 计算总得分（强依赖特征占主导）
    total_score = (
        strong_features_score +           # 强依赖特征得分
        other_features_score +            # 其他特征得分
        interaction_score                 # 交互效应得分
    )
    
    # 使用sigmoid函数转换为概率，并添加负偏移使整体概率降低到15%左右
    # 通过调整偏移量（-1.5）和缩放因子（2.5）来控制整体报名率
    prob = 1 / (1 + np.exp(-(total_score * 2.5 - 1.5)))
    
    # 进一步调整：确保概率范围合理，整体报名率约15%
    prob = np.clip(prob, 0.01, 0.85)  # 限制概率范围
    
    # 微调：如果完全不满足强依赖条件，进一步降低概率
    no_strong_features = (strong_features_score < 0.2)
    prob[no_strong_features] = prob[no_strong_features] * 0.3  # 降低70%的概率
    
    # 添加少量随机噪声（保持特征效应明显）
    prob = prob + np.random.normal(0, 0.02, n_samples)
    prob = np.clip(prob, 0.0, 1.0)
    
    # 生成标签
    df['is_enrolled'] = (np.random.random(n_samples) < prob).astype(int)
    
    return df


def save_data(df: pd.DataFrame, filepath: str = 'data/mock_data.csv'):
    """
    保存数据到CSV文件
    
    Args:
        df: 要保存的DataFrame
        filepath: 保存路径
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据已保存到: {filepath}")


def load_data(filepath: str = 'data/mock_data.csv') -> pd.DataFrame:
    """
    从CSV文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的DataFrame
    """
    return pd.read_csv(filepath, encoding='utf-8-sig')


if __name__ == '__main__':
    # 生成并保存数据
    print("正在生成MOCK数据...")
    df = generate_mock_data(n_samples=10000)
    print(f"数据生成完成，共 {len(df)} 条记录")
    print(f"报名率: {df['is_enrolled'].mean():.2%}")
    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())
    
    save_data(df)

