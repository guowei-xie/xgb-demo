"""
API服务模块

提供RESTful API接口，接收用户特征，返回预测概率和等级标签。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    API_RELOAD,
    API_WORKERS,
    LEVEL_TAG_RULES,
    MODEL_PATH,
    PROCESSOR_PATH,
)
from utils.feature_processor import FeatureProcessor
from utils.level_tagger import assign_level_tags
from utils.model_trainer import ModelTrainer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="用户续报预测API",
    description="基于XGBoost模型的用户续报概率预测服务",
    version="1.0.0",
)

# 全局变量，用于存储加载的模型和特征处理器
processor: Optional[FeatureProcessor] = None
trainer: Optional[ModelTrainer] = None


class UserFeatures(BaseModel):
    """
    用户特征请求模型
    
    支持所有可能的特征字段，缺失的字段将使用None或默认值。
    """
    # 基础特征
    city: Optional[str] = Field(None, description="城市")
    # district: Optional[str] = Field(None, description="区县")
    city_level: Optional[str] = Field(None, description="城市等级")
    city_score: Optional[float] = Field(None, description="城市评分")
    house_price: Optional[float] = Field(None, description="房价")
    grade: Optional[int] = Field(None, description="年级")
    refresh_num: Optional[int] = Field(None, description="复刷次数")
    device: Optional[str] = Field(None, description="设备类型")
    is_enable: Optional[int] = Field(None, description="是否激活课程")
    fns_cnt: Optional[int] = Field(None, description="完课计数")
    
    # 可选字段（用于兼容，但不会被用作特征）
    user_id: Optional[str] = Field(None, description="用户ID（可选）")
    b2c_term_name: Optional[str] = Field(None, description="B2C学期名称（可选）")
    l1_term_name: Optional[str] = Field(None, description="L1学期名称（可选）")
    l1_term_renewal_end_date: Optional[str] = Field(None, description="续报结束日期（可选）")
    
    class Config:
        """Pydantic配置"""
        # 允许使用字段名作为别名
        populate_by_name = True
        # 示例数据
        json_schema_extra = {
            "example": {
                "city": "扬州市",
                "city_level": "三线城市",
                "city_score": 120.0,
                "house_price": 4941.0,
                "grade": 3,
                "refresh_num": 1,
                "device": "电脑",
                "is_enable": 1,
                "fns_cnt": 5,
            }
        }


class PredictionResponse(BaseModel):
    """
    预测响应模型
    """
    probability: float = Field(..., description="续报概率（0-1之间）")
    level_tag: str = Field(..., description="等级标签（S/A/B/C/D）")
    success: bool = Field(True, description="预测是否成功")


class BatchPredictionRequest(BaseModel):
    """
    批量预测请求模型
    """
    users: List[UserFeatures] = Field(..., description="用户特征列表")


class BatchPredictionResponse(BaseModel):
    """
    批量预测响应模型
    """
    predictions: List[PredictionResponse] = Field(..., description="预测结果列表")
    total: int = Field(..., description="总用户数")
    success_count: int = Field(..., description="成功预测数")


def load_models() -> None:
    """
    加载特征处理器和模型
    
    在应用启动时调用，将模型加载到内存中以提高预测速度。
    """
    global processor, trainer
    
    try:
        logger.info("正在加载特征处理器和模型...")
        
        if not PROCESSOR_PATH.exists():
            raise FileNotFoundError(f"未找到特征处理器: {PROCESSOR_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")
        
        processor = FeatureProcessor.load(str(PROCESSOR_PATH))
        trainer = ModelTrainer.load(str(MODEL_PATH))
        
        logger.info("特征处理器和模型加载成功！")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件
    
    在FastAPI应用启动时自动加载模型。
    """
    load_models()


@app.get("/health", tags=["健康检查"])
async def health_check() -> Dict[str, Any]:
    """
    健康检查接口
    
    Returns:
        服务状态信息
    """
    return {
        "status": "healthy",
        "model_loaded": processor is not None and trainer is not None,
        "processor_path": str(PROCESSOR_PATH),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["预测"])
async def predict(user_features: UserFeatures) -> PredictionResponse:
    """
    单个用户预测接口
    
    接收用户特征，返回预测概率和等级标签。
    
    Args:
        user_features: 用户特征数据
        
    Returns:
        预测结果，包含概率和等级标签
        
    Raises:
        HTTPException: 当模型未加载或预测失败时
    """
    if processor is None or trainer is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请检查模型文件是否存在"
        )
    
    try:
        # 将用户特征转换为DataFrame
        user_dict = user_features.model_dump(exclude_none=True)
        user_df = pd.DataFrame([user_dict])
        
        # 特征处理
        features = processor.transform(user_df)
        
        # 预测概率
        probability = float(trainer.predict_proba(features)[0])
        
        # 生成等级标签
        proba_series = pd.Series([probability])
        level_tags = assign_level_tags(proba_series, LEVEL_TAG_RULES)
        level_tag = level_tags.iloc[0]
        
        logger.info(f"预测成功 - 概率: {probability:.4f}, 等级: {level_tag}")
        
        return PredictionResponse(
            probability=probability,
            level_tag=level_tag,
            success=True,
        )
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"预测失败: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["预测"])
async def predict_batch(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    批量用户预测接口
    
    接收多个用户特征，批量返回预测结果。
    
    Args:
        batch_request: 批量预测请求，包含用户特征列表
        
    Returns:
        批量预测结果
        
    Raises:
        HTTPException: 当模型未加载或预测失败时
    """
    if processor is None or trainer is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请检查模型文件是否存在"
        )
    
    if not batch_request.users:
        raise HTTPException(
            status_code=400,
            detail="用户列表不能为空"
        )
    
    try:
        # 将用户特征列表转换为DataFrame
        user_dicts = [user.model_dump(exclude_none=True) for user in batch_request.users]
        users_df = pd.DataFrame(user_dicts)
        
        # 特征处理
        features = processor.transform(users_df)
        
        # 批量预测概率
        probabilities = trainer.predict_proba(features)
        
        # 生成等级标签
        proba_series = pd.Series(probabilities)
        level_tags = assign_level_tags(proba_series, LEVEL_TAG_RULES)
        
        # 构建响应
        predictions = [
            PredictionResponse(
                probability=float(prob),
                level_tag=tag,
                success=True,
            )
            for prob, tag in zip(probabilities, level_tags)
        ]
        
        logger.info(f"批量预测成功 - 总数: {len(predictions)}")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            success_count=len(predictions),
        )
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"批量预测失败: {str(e)}"
        )


@app.get("/", tags=["根路径"])
async def root() -> Dict[str, Any]:
    """
    API根路径
    
    Returns:
        API基本信息
    """
    return {
        "message": "用户续报预测API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    # 从配置文件加载运行参数
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        workers=API_WORKERS if not API_RELOAD else 1,  # reload模式下只能使用单进程
    )

