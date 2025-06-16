"""
核心接口模块
包含系统状态、LLM对话等核心功能接口
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from src.models import (
    HealthResponse, SystemStatusResponse, 
    ChatRequest, ChatResponse
)
from src.utils import generate_response_id

router = APIRouter()

# 全局变量引用（从 main.py 导入）
def get_managers():
    """获取全局管理器实例"""
    from main import config_manager, mcp_manager, llm_manager, speech_processor, logger, ppt_processor
    return {
        'config_manager': config_manager,
        'mcp_manager': mcp_manager,
        'llm_manager': llm_manager,
        'speech_processor': speech_processor,
        'logger': logger,
        'ppt_processor': ppt_processor
    }

# ==================== 系统状态接口 ====================

@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info("🔍 执行健康检查")
    
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now(datetime.timezone.utc),
            "version": "2.0.0",
            "components": {
                "config_manager": await managers['config_manager'].health_check() if managers['config_manager'] else {"healthy": False},
                "mcp_manager": await managers['mcp_manager'].health_check() if managers['mcp_manager'] else {"healthy": False},
                "llm_manager": await managers['llm_manager'].health_check() if managers['llm_manager'] else {"healthy": False},
                "speech_processor": await managers['speech_processor'].health_check() if managers['speech_processor'] else {"healthy": False},
                "ppt_processor": await managers['ppt_processor'].health_check() if managers['ppt_processor'] else {"healthy": False}
            }
        }
        
        # 检查是否有组件不健康
        unhealthy_components = [
            name for name, health in status["components"].items() 
            if not health.get("healthy", False)
        ]
        
        if unhealthy_components:
            status["status"] = "degraded"
            if logger:
                logger.warning(f"⚠️ 部分组件不健康: {unhealthy_components}")
        
        if logger:
            logger.info("✅ 健康检查完成")
        
        return HealthResponse(**status)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统详细状态"""
    managers = get_managers()
    logger = managers['logger']
    
    if logger:
        logger.info("📊 获取系统状态")
    
    try:
        # 导入辅助函数
        from main import get_active_sessions_count, get_system_metrics
        
        status = {
            "success": True,
            "timestamp": datetime.now(datetime.timezone.utc),
            "uptime": datetime.now(datetime.timezone.utc),  # 实际应用中应该记录启动时间
            "mcp_tools": await managers['mcp_manager'].get_tools_status() if managers['mcp_manager'] else {},
            "llm_models": await managers['llm_manager'].get_models_status() if managers['llm_manager'] else {},
            "active_sessions": await get_active_sessions_count(),
            "system_metrics": await get_system_metrics()
        }
        
        if logger:
            logger.info("✅ 系统状态获取完成")
        
        return SystemStatusResponse(**status)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LLM 对话接口 ====================

@router.post("/api/chat/text", response_model=ChatResponse)
async def text_chat(request: ChatRequest):
    """文本对话接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"💬 开始文本对话 [请求ID: {request_id}] - 模型: {request.model_name}")
    
    try:
        result = await managers['llm_manager'].chat(
            model_name=request.model_name,
            message=request.message,
            system_prompt=request.system_prompt,
            session_id=request.session_id,
            stream=request.stream,
            tools=request.tools,
            request_id=request_id
        )
        
        logger.info(f"✅ 文本对话完成 [请求ID: {request_id}]")
        return result
        
    except Exception as e:
        logger.error(f"❌ 文本对话失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """流式文本对话接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"🌊 开始流式对话 [请求ID: {request_id}] - 模型: {request.model_name}")
    
    try:
        async def generate():
            async for chunk in managers['llm_manager'].stream_chat(
                model_name=request.model_name,
                message=request.message,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                tools=request.tools,
                request_id=request_id
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            logger.info(f"✅ 流式对话完成 [请求ID: {request_id}]")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id
            }
        )
        
    except Exception as e:
        logger.error(f"❌ 流式对话失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))