"""
MCP 配置相关接口模块
"""

from fastapi import APIRouter, HTTPException
from typing import List
from src.models import MCPConfigCreate, MCPConfigUpdate, MCPConfigResponse
from src.utils import validate_config

router = APIRouter()

# 全局变量引用
def get_managers():
    """获取全局管理器实例"""
    from main import config_manager, mcp_manager, logger
    return {
        'config_manager': config_manager,
        'mcp_manager': mcp_manager,
        'logger': logger
    }

# ==================== MCP 配置接口 ====================

@router.get("/api/mcp/configs", response_model=List[MCPConfigResponse])
async def get_mcp_configs():
    """获取所有 MCP 工具配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info("📋 获取 MCP 配置列表")
    
    try:
        configs = await managers['mcp_manager'].get_all_configs()
        if logger:
            logger.info(f"✅ 获取到 {len(configs)} 个 MCP 配置")
        return configs
        
    except Exception as e:
        logger.error(f"❌ 获取 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/mcp/configs", response_model=MCPConfigResponse)
async def create_mcp_config(config: MCPConfigCreate):
    """创建新的 MCP 工具配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"➕ 创建 MCP 配置: {config.name}")
    
    try:
        # 验证配置
        await validate_config(config.dict(), "mcp")
        
        # 创建配置
        result = await managers['mcp_manager'].create_config(config)
        
        # 保存到配置文件
        await managers['config_manager'].save_mcp_config(result)
        
        logger.info(f"✅ MCP 配置创建成功: {result.name}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ MCP 配置验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 创建 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/mcp/configs/{config_id}", response_model=MCPConfigResponse)
async def update_mcp_config(config_id: str, config: MCPConfigUpdate):
    """更新 MCP 工具配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"✏️ 更新 MCP 配置: {config_id}")
    
    try:
        # 验证配置存在
        existing_config = await managers['mcp_manager'].get_config(config_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"MCP 配置不存在: {config_id}")
        
        # 更新配置
        result = await managers['mcp_manager'].update_config(config_id, config)
        
        # 保存到配置文件
        await managers['config_manager'].save_mcp_config(result)
        
        logger.info(f"✅ MCP 配置更新成功: {config_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/mcp/configs/{config_id}")
async def delete_mcp_config(config_id: str):
    """删除 MCP 工具配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"🗑️ 删除 MCP 配置: {config_id}")
    
    try:
        # 停止相关的 MCP 客户端
        await managers['mcp_manager'].stop_client(config_id)
        
        # 删除配置
        await managers['mcp_manager'].delete_config(config_id)
        
        # 从配置文件中删除
        await managers['config_manager'].delete_mcp_config(config_id)
        
        logger.info(f"✅ MCP 配置删除成功: {config_id}")
        return {"message": f"MCP 配置 {config_id} 已删除"}
        
    except Exception as e:
        logger.error(f"❌ 删除 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/mcp/configs/{config_id}/test")
async def test_mcp_config(config_id: str):  
    """测试 MCP 工具配置"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info(f"🧪 测试 MCP 配置: {config_id}")
    
    try:
        result = await managers['mcp_manager'].test_config(config_id)
        
        if result["success"]:
            logger.info(f"✅ MCP 配置测试成功: {config_id}")
        else:
            logger.warning(f"⚠️ MCP 配置测试失败: {config_id} - {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 测试 MCP 配置异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))