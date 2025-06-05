#!/usr/bin/env python3
#!/usr/bin/env python3
"""
AI Agent 后端服务主入口
提供 API 接口用于 MCP 配置、模型配置和语音处理
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from src.models import (
            LLMConfigCreate, LLMConfigUpdate, LLMConfigResponse, 
            MCPConfigCreate, MCPConfigUpdate, MCPConfigResponse,
            HealthResponse, SpeechRecognitionResponse,
            SpeechSynthesisResponse, SpeechSynthesisRequest, VoiceChatResponse,
            ChatRequest, ChatResponse, SystemStatusResponse
        )

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
        from src.utils import validate_config, generate_response_id
        
        return {
            'ConfigManager': ConfigManager,
            'MCPManager': MCPManager,
            'LLMManager': LLMManager,
            'SpeechProcessor': SpeechProcessor,
            'setup_logger': setup_logger,
            'get_logger': get_logger,
            'validate_config': validate_config,
            'generate_response_id': generate_response_id,
            # 导出所有响应模型
            'HealthResponse': HealthResponse,
            'SpeechRecognitionResponse': SpeechRecognitionResponse,
            'SpeechSynthesisResponse': SpeechSynthesisResponse,
            'SpeechSynthesisRequest': SpeechSynthesisRequest,
            'VoiceChatResponse': VoiceChatResponse,
            'ChatRequest': ChatRequest,
            'ChatResponse': ChatResponse,
            'MCPConfigCreate': MCPConfigCreate,
            'MCPConfigUpdate': MCPConfigUpdate,
            'MCPConfigResponse': MCPConfigResponse,
            'LLMConfigCreate': LLMConfigCreate,
            'LLMConfigUpdate': LLMConfigUpdate,
            'LLMConfigResponse': LLMConfigResponse,
            'SystemStatusResponse': SystemStatusResponse
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

def get_response_models():
    """获取响应模型类"""
    global managers
    if managers is None or 'error' in managers:
        managers = get_managers()
    return managers

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

# ==================== 系统状态接口 ====================

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    models = get_response_models()

    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    HealthResponse = models['HealthResponse']
    
    if logger:
        logger.info("🔍 执行健康检查")
    
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "2.0.0",
            "components": {
                "config_manager": await config_manager.health_check() if config_manager else {"healthy": False},
                "mcp_manager": await mcp_manager.health_check() if mcp_manager else {"healthy": False},
                "llm_manager": await llm_manager.health_check() if llm_manager else {"healthy": False},
                "speech_processor": await speech_processor.health_check() if speech_processor else {"healthy": False}
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

@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统详细状态"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    SystemStatusResponse = models['SystemStatusResponse']
    
    if logger:
        logger.info("📊 获取系统状态")
    
    try:
        status = {
            "success": True,
            "timestamp": datetime.utcnow(),
            "uptime": datetime.utcnow(),  # 实际应用中应该记录启动时间
            "mcp_tools": await mcp_manager.get_tools_status() if mcp_manager else {},
            "llm_models": await llm_manager.get_models_status() if llm_manager else {},
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

# ==================== MCP 配置接口 ====================

@app.get("/api/mcp/configs", response_model=List[MCPConfigResponse])
async def get_mcp_configs():
    """获取所有 MCP 工具配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    MCPConfigResponse = models['MCPConfigResponse']
    
    if logger:
        logger.info("📋 获取 MCP 配置列表")
    
    try:
        configs = await mcp_manager.get_all_configs()
        if logger:
            logger.info(f"✅ 获取到 {len(configs)} 个 MCP 配置")
        return configs
        
    except Exception as e:
        logger.error(f"❌ 获取 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mcp/configs", response_model=MCPConfigResponse)
async def create_mcp_config(config: MCPConfigCreate):
    """创建新的 MCP 工具配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    MCPConfigResponse = models['MCPConfigResponse']
    
    if logger:
        logger.info(f"➕ 创建 MCP 配置: {config.name}")
    
    try:
        # 验证配置
        await validate_config(config.dict(), "mcp")
        
        # 创建配置
        result = await mcp_manager.create_config(config)
        
        # 保存到配置文件
        await config_manager.save_mcp_config(result)
        
        logger.info(f"✅ MCP 配置创建成功: {result.name}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ MCP 配置验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 创建 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/mcp/configs/{config_id}", response_model=MCPConfigResponse)
async def update_mcp_config(config_id: str, config: MCPConfigUpdate):
    """更新 MCP 工具配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    MCPConfigResponse = models['MCPConfigResponse']
    
    if logger:
        logger.info(f"✏️ 更新 MCP 配置: {config_id}")
    
    try:
        # 验证配置存在
        existing_config = await mcp_manager.get_config(config_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"MCP 配置不存在: {config_id}")
        
        # 更新配置
        result = await mcp_manager.update_config(config_id, config)
        
        # 保存到配置文件
        await config_manager.save_mcp_config(result)
        
        logger.info(f"✅ MCP 配置更新成功: {config_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/mcp/configs/{config_id}")
async def delete_mcp_config(config_id: str):
    """删除 MCP 工具配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    MCPConfigResponse = models['MCPConfigResponse']
    
    if logger:
        logger.info(f"🗑️ 删除 MCP 配置: {config_id}")
    
    try:
        # 停止相关的 MCP 客户端
        await mcp_manager.stop_client(config_id)
        
        # 删除配置
        await mcp_manager.delete_config(config_id)
        
        # 从配置文件中删除
        await config_manager.delete_mcp_config(config_id)
        
        logger.info(f"✅ MCP 配置删除成功: {config_id}")
        return {"message": f"MCP 配置 {config_id} 已删除"}
        
    except Exception as e:
        logger.error(f"❌ 删除 MCP 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mcp/configs/{config_id}/test")
async def test_mcp_config(config_id: str):  
    """测试 MCP 工具配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    MCPConfigResponse = models['MCPConfigResponse']
    
    if logger:
        logger.info(f"🧪 测试 MCP 配置: {config_id}")
    
    try:
        result = await mcp_manager.test_config(config_id)
        
        if result["success"]:
            logger.info(f"✅ MCP 配置测试成功: {config_id}")
        else:
            logger.warning(f"⚠️ MCP 配置测试失败: {config_id} - {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 测试 MCP 配置异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LLM 模型配置接口 ====================

@app.get("/api/llm/configs", response_model=List[LLMConfigResponse])
async def get_llm_configs():
    """获取所有 LLM 模型配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    LLMConfigResponse = models['LLMConfigResponse']
    
    if logger:
        logger.info("📋 获取 LLM 配置列表")
    
    try:
        configs = await llm_manager.get_all_configs()
        if logger:
            logger.info(f"✅ 获取到 {len(configs)} 个 LLM 配置")
        return configs
        
    except Exception as e:
        logger.error(f"❌ 获取 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/configs", response_model=LLMConfigResponse)
async def create_llm_config(config: LLMConfigCreate):
    """创建新的 LLM 模型配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    LLMConfigResponse = models['LLMConfigResponse']
    
    if logger:
        logger.info(f"➕ 创建 LLM 配置: {config.name}")
    
    try:
        # 验证配置
        await validate_config(config.dict(), "llm")
        
        # 创建配置
        result = await llm_manager.create_config(config)
        
        # 保存到配置文件
        await config_manager.save_llm_config(result)
        
        logger.info(f"✅ LLM 配置创建成功: {result.name}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ LLM 配置验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 创建 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/llm/configs/{config_id}", response_model=LLMConfigResponse)
async def update_llm_config(config_id: str, config: LLMConfigUpdate):
    """更新 LLM 模型配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    LLMConfigResponse = models['LLMConfigResponse']
    
    if logger:
        logger.info(f"✏️ 更新 LLM 配置: {config_id}")
    
    try:
        # 验证配置存在
        existing_config = await llm_manager.get_config(config_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"LLM 配置不存在: {config_id}")
        
        # 更新配置
        result = await llm_manager.update_config(config_id, config)
        
        # 保存到配置文件
        await config_manager.save_llm_config(result)
        
        logger.info(f"✅ LLM 配置更新成功: {config_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/llm/configs/{config_id}")
async def delete_llm_config(config_id: str):
    """删除 LLM 模型配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    LLMConfigResponse = models['LLMConfigResponse']
    
    if logger:
        logger.info(f"🗑️ 删除 LLM 配置: {config_id}")
    
    try:
        # 删除配置
        await llm_manager.delete_config(config_id)
        
        # 从配置文件中删除
        await config_manager.delete_llm_config(config_id)
        
        logger.info(f"✅ LLM 配置删除成功: {config_id}")
        return {"message": f"LLM 配置 {config_id} 已删除"}
        
    except Exception as e:
        logger.error(f"❌ 删除 LLM 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/configs/{config_id}/test")
async def test_llm_config(config_id: str):  
    """测试 LLM 模型配置"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    LLMConfigResponse = models['LLMConfigResponse']
    
    if logger:
        logger.info(f"🧪 测试 LLM 配置: {config_id}")
    
    try:
        result = await llm_manager.test_config(config_id)
        
        if result["success"]:
            logger.info(f"✅ LLM 配置测试成功: {config_id}")
        else:
            logger.warning(f"⚠️ LLM 配置测试失败: {config_id} - {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 测试 LLM 配置异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 语音处理接口 ====================

@app.post("/api/speech/recognize", response_model=SpeechRecognitionResponse)
async def recognize_speech(
    audio_file: UploadFile = File(...),
    language: Optional[str] = "zh-CN",
    use_asr_model: Optional[str] = None
):
    """语音识别接口"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    SpeechRecognitionResponse = models['SpeechRecognitionResponse']
    
    request_id = generate_response_id()
    logger.info(f"🎤 开始语音识别 [请求ID: {request_id}] - 文件: {audio_file.filename}")
    
    try:
        # 验证文件类型
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="文件必须是音频格式")
        
        # 读取音频数据
        audio_data = await audio_file.read()
        logger.info(f"📁 音频文件读取完成 [请求ID: {request_id}] - 大小: {len(audio_data)} bytes")
        
        # 执行语音识别
        result = await speech_processor.recognize(
            audio_data=audio_data,
            language=language,
            model_name=use_asr_model,
            request_id=request_id
        )
        
        logger.info(f"✅ 语音识别完成 [请求ID: {request_id}] - 文本长度: {len(result.text)}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 语音识别失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech/synthesize", response_model=SpeechSynthesisResponse)
async def synthesize_speech(request: SpeechSynthesisRequest):
    """语音合成接口"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    SpeechSynthesisResponse = models['SpeechSynthesisResponse']
    
    request_id = generate_response_id()
    logger.info(f"🔊 开始语音合成 [请求ID: {request_id}] - 文本长度: {len(request.text)}")
    
    try:
        # 执行语音合成
        result = await speech_processor.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            pitch=request.pitch,
            tts_model=request.tts_model,
            request_id=request_id
        )
        
        logger.info(f"✅ 语音合成完成 [请求ID: {request_id}] - 音频大小: {len(result.audio_data)} bytes")
        return result
        
    except Exception as e:
        logger.error(f"❌ 语音合成失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/voice", response_model=VoiceChatResponse)
async def voice_chat(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """语音对话接口（语音输入 + 文本和语音输出）"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    VoiceChatResponse = models['VoiceChatResponse']
    
    request_id = generate_response_id()
    logger.info(f"💬 开始语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    try:
        # 1. 语音识别
        audio_data = await audio_file.read()
        logger.info(f"🎤 执行语音识别 [请求ID: {request_id}]")
        
        recognition_result = await speech_processor.recognize(
            audio_data=audio_data,
            request_id=request_id
        )
        
        user_text = recognition_result.text
        logger.info(f"🔤 识别结果 [请求ID: {request_id}]: {user_text}")
        
        # 2. LLM 对话
        logger.info(f"🤖 调用 LLM 模型 [请求ID: {request_id}]")
        
        chat_response = await llm_manager.chat(
            model_name=llm_model,
            message=user_text,
            system_prompt=system_prompt,
            session_id=session_id,
            request_id=request_id
        )
        
        response_text = chat_response.content
        logger.info(f"💭 LLM 响应 [请求ID: {request_id}]: {response_text[:100]}...")
        
        # 3. 语音合成
        logger.info(f"🔊 执行语音合成 [请求ID: {request_id}]")
        
        synthesis_result = await speech_processor.synthesize(
            text=response_text,
            request_id=request_id
        )
        
        logger.info(f"✅ 语音对话完成 [请求ID: {request_id}]")
        
        return VoiceChatResponse(
            request_id=request_id,
            user_text=user_text,
            response_text=response_text,
            response_audio=synthesis_result.audio_data,
            audio_format=synthesis_result.format,
            session_id=session_id or request_id,
            model_used=chat_response.model_name,
            processing_time={
                "recognition": recognition_result.processing_time,
                "llm_chat": chat_response.processing_time,
                "synthesis": synthesis_result.processing_time
            }
        )
        
    except Exception as e:
        logger.error(f"❌ 语音对话失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== LLM 对话接口 ====================

@app.post("/api/chat/text", response_model=ChatResponse)
async def text_chat(request: ChatRequest):
    """文本对话接口"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    ChatResponse = models['ChatResponse']
    
    request_id = generate_response_id()
    logger.info(f"💬 开始文本对话 [请求ID: {request_id}] - 模型: {request.model_name}")
    
    try:
        result = await llm_manager.chat(
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

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """流式文本对话接口"""
    models = get_response_models()
    
    if 'error' in models:
        raise HTTPException(status_code=500, detail=f"模块导入错误: {models['error']}")
    
    ChatResponse = models['ChatResponse']
    
    request_id = generate_response_id()
    logger.info(f"🌊 开始流式对话 [请求ID: {request_id}] - 模型: {request.model_name}")
    
    try:
        async def generate():
            async for chunk in llm_manager.stream_chat(
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
    
    logger.info(f"🚀 启动 AI Agent Backend Server")
    logger.info(f"📍 地址: http://{args.host}:{args.port}")
    logger.info(f"📖 API 文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )