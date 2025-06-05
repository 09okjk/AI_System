"""
AI Agent Backend 核心模块
"""

__version__ = "2.0.0"
__author__ = "AI Agent Team"
__description__ = "AI智能代理后端服务"

from .config import ConfigManager
from .logger import setup_logger, get_logger
from .mcp import MCPManager
from .llm import LLMManager
from .speech import SpeechProcessor
from .utils import (
    generate_response_id,
    generate_session_id,
    validate_config,
    session_manager,
    cache,
    rate_limiter
)
# 添加对 models 模块中类的导入
from .models import (
    HealthResponse, 
    SystemStatusResponse, 
    SpeechRecognitionResponse,
    SpeechSynthesisResponse, 
    SpeechSynthesisRequest, 
    VoiceChatResponse,
    ChatRequest, 
    ChatResponse,
    MCPConfigCreate, 
    MCPConfigUpdate, 
    MCPConfigResponse,
    LLMConfigCreate, 
    LLMConfigUpdate, 
    LLMConfigResponse
)

__all__ = [
    "ConfigManager",
    "MCPManager", 
    "LLMManager",
    "SpeechProcessor",
    "setup_logger",
    "get_logger",
    "generate_response_id",
    "generate_session_id",
    "validate_config",
    "session_manager",
    "cache",
    "rate_limiter",
    # 添加 models 模块中的类
    "HealthResponse",
    "SystemStatusResponse",
    "SpeechRecognitionResponse",
    "SpeechSynthesisResponse",
    "SpeechSynthesisRequest",
    "VoiceChatResponse",
    "ChatRequest",
    "ChatResponse",
    "MCPConfigCreate",
    "MCPConfigUpdate",
    "MCPConfigResponse",
    "LLMConfigCreate",
    "LLMConfigUpdate",
    "LLMConfigResponse"
]