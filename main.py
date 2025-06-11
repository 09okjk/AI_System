#!/usr/bin/env python3
"""
AI Agent 后端服务主入口
提供 API 接口用于 MCP 配置、模型配置和语音处理
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import sys
import uvicorn

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 延迟导入，避免循环依赖
def get_managers():
    """延迟导入管理器类"""
    try:
        from src.config import ConfigManager
        from src.mcp import MCPManager
        from src.llm import LLMManager
        from src.speech import SpeechProcessor
        from src.logger import setup_logger, get_logger
        
        return {
            'ConfigManager': ConfigManager,
            'MCPManager': MCPManager,
            'LLMManager': LLMManager,
            'SpeechProcessor': SpeechProcessor,
            'setup_logger': setup_logger,
            'get_logger': get_logger,
        }
    except ImportError as e:
        # 如果导入失败，返回一个包含错误信息的字典
        return {'error': f"导入失败: {str(e)}"}

# 创建 FastAPI 应用
app = FastAPI(
    title="AI Agent Backend API",
    description="AI智能代理后端服务，支持MCP配置、模型管理和语音处理",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量（在startup时初始化）
managers = None
config_manager = None
mcp_manager = None
llm_manager = None
speech_processor = None
logger = None

# 提前注册路由（确保 start_server.py 能够检测到）
try:
    from api import register_routers
    register_routers(app)
except ImportError as e:
    # 如果导入失败，会在启动时处理
    print(f"⚠️ 路由注册延迟: {e}")

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    global managers, config_manager, mcp_manager, llm_manager, speech_processor, logger
    
    # 获取管理器类
    managers = get_managers()

    # 检查导入是否成功
    if 'error' in managers:
        print(f"❌ 导入模块失败: {managers['error']}")
        raise RuntimeError(f"模块导入失败: {managers['error']}")
    
    # 设置日志
    managers['setup_logger']()
    logger = managers['get_logger'](__name__)
    
    logger.info("🚀 AI Agent Backend 正在启动...")
    
    try:
        # 初始化管理器实例
        config_manager = managers['ConfigManager']()
        mcp_manager = managers['MCPManager']()
        llm_manager = managers['LLMManager']()
        speech_processor = managers['SpeechProcessor']()
        
        # 依次初始化
        await config_manager.initialize()
        logger.info("✅ 配置管理器初始化完成")
        
        await mcp_manager.initialize(config_manager.get_mcp_configs())
        logger.info("✅ MCP 管理器初始化完成")
        
        await llm_manager.initialize(config_manager.get_llm_configs())
        logger.info("✅ LLM 管理器初始化完成")
        
        await speech_processor.initialize()
        logger.info("✅ 语音处理器初始化完成")
        
        # 如果路由还没有注册，再次尝试注册
        if not hasattr(app, '_routes_registered'):
            try:
                from api import register_routers
                register_routers(app)
                app._routes_registered = True
                logger.info("✅ API 路由注册完成")
            except Exception as route_error:
                logger.error(f"❌ 路由注册失败: {route_error}")
                raise
        
        logger.info("🎉 AI Agent Backend 启动成功！")
        
    except Exception as e:
        logger.error(f"❌ 启动失败: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    if logger:
        logger.info("🔄 AI Agent Backend 正在关闭...")
    
    try:
        if speech_processor:
            await speech_processor.cleanup()
        if llm_manager:
            await llm_manager.cleanup()
        if mcp_manager:
            await mcp_manager.cleanup()
        if config_manager:
            await config_manager.cleanup()
        
        if logger:
            logger.info("✅ AI Agent Backend 已安全关闭")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ 关闭时出错: {str(e)}")

# ==================== 辅助函数 ====================

async def get_active_sessions_count() -> int:
    """获取活跃会话数量"""
    try:
        if llm_manager:
            return await llm_manager.get_active_sessions_count()
        return 0
    except:
        return 0

async def get_system_metrics() -> Dict[str, Any]:
    """获取系统指标"""
    try:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except:
        return {}

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听主机")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    print(f"🚀 启动 AI Agent Backend Server")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📖 API 文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )