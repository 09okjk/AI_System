"""
API 路由注册模块
"""

from fastapi import APIRouter
from .mcp_api import router as mcp_router
from .llm_api import router as llm_router
from .speech_api import router as speech_router
from .core_api import router as core_router
from .mongodb_api import router as mongodb_router

def register_routers(app):
    """注册所有 API 路由"""
    
    # 注册各个模块的路由
    app.include_router(core_router, tags=["Core"])
    app.include_router(mcp_router, tags=["MCP"])
    app.include_router(llm_router, tags=["LLM"])
    app.include_router(speech_router, tags=["Speech"])
    app.include_router(mongodb_router, tags=["MongoDB"])