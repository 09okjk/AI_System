"""
LLM 模型配置相关接口模块
"""

from fastapi import APIRouter, HTTPException
from typing import List
from src.models import LLMConfigCreate, LLMConfigUpdate, LLMConfigResponse
from src.utils import validate_config

router = APIRouter()

# 全局变量引用
def get_managers():
    """获取全局管理器实例"""
    from main import config_manager, llm_manager, logger
    return {
        'config_manager': config_manager,
        'llm_manager': llm_manager,
        'logger': logger
    }

# ==================== LLM 模型配置接口 ====================

@router.get("/api/llm/configs", response_model=List[LLMConfigResponse])
async def get_llm_configs():
    """获取所有 LLM 模型配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info("📋 获取 LLM 配置列表")
    
    try:
        configs = await managers['llm_manager'].get_all_configs()
        if logger:
            logger.info(f"✅ 获取到 {len(configs)} 个 LLM 配置")
        return configs
        
    except Exception as e:
        logger.error(f"❌ 获取 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/llm/configs", response_model=LLMConfigResponse)
async def create_llm_config(config: LLMConfigCreate):
    """创建新的 LLM 模型配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"➕ 创建 LLM 配置: {config.name}")
    
    try:
        # 验证配置
        await validate_config(config.dict(), "llm")
        
        # 创建配置
        result = await managers['llm_manager'].create_config(config)
        
        # 保存到配置文件
        await managers['config_manager'].save_llm_config(result)
        
        logger.info(f"✅ LLM 配置创建成功: {result.name}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ LLM 配置验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 创建 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/llm/configs/{config_id}", response_model=LLMConfigResponse)
async def update_llm_config(config_id: str, config: LLMConfigUpdate):
    """更新 LLM 模型配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"✏️ 更新 LLM 配置: {config_id}")
    
    try:
        # 验证配置存在
        existing_config = await managers['llm_manager'].get_config(config_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"LLM 配置不存在: {config_id}")
        
        # 更新配置
        result = await managers['llm_manager'].update_config(config_id, config)
        
        # 保存到配置文件
        await managers['config_manager'].save_llm_config(result)
        
        logger.info(f"✅ LLM 配置更新成功: {config_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/llm/configs/{config_id}")
async def delete_llm_config(config_id: str):
    """删除 LLM 模型配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"🗑️ 删除 LLM 配置: {config_id}")
    
    try:
        # 删除配置
        await managers['llm_manager'].delete_config(config_id)
        
        # 从配置文件中删除
        await managers['config_manager'].delete_llm_config(config_id)
        
        logger.info(f"✅ LLM 配置删除成功: {config_id}")
        return {"message": f"LLM 配置 {config_id} 已删除"}
        
    except Exception as e:
        logger.error(f"❌ 删除 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/llm/configs/{config_id}/test")
async def test_llm_config(config_id: str):  
    """测试 LLM 模型配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"🧪 测试 LLM 配置: {config_id}")
    
    try:
        result = await managers['llm_manager'].test_config(config_id)
        
        if result["success"]:
            logger.info(f"✅ LLM 配置测试成功: {config_id}")
        else:
            logger.warning(f"⚠️ LLM 配置测试失败: {config_id} - {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 测试 LLM 配置异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))