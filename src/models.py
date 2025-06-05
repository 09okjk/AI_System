"""
API 请求和响应模型定义 - 完整版
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# ==================== 枚举定义 ====================

class LLMProvider(str, Enum):
    """LLM 提供商"""
    DASHSCOPE = "dashscope"
    XINFERENCE = "xinference"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

class MCPTransport(str, Enum):
    """MCP 传输方式"""
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    HTTP = "http"

class AudioFormat(str, Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    PCM = "pcm"
    FLAC = "flac"

# ==================== 基础响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class HealthResponse(BaseResponse):
    """健康检查响应"""
    status: str
    version: str
    components: Dict[str, Dict[str, Any]]

class SystemStatusResponse(BaseResponse):
    """系统状态响应"""
    uptime: datetime
    mcp_tools: Dict[str, Any]
    llm_models: Dict[str, Any]
    active_sessions: int
    system_metrics: Dict[str, Any]

# ==================== MCP 相关模型 ====================

class MCPConfigBase(BaseModel):
    """MCP 配置基础模型"""
    name: str = Field(..., description="MCP 工具名称")
    description: Optional[str] = Field(None, description="工具描述")
    command: str = Field(..., description="启动命令")
    args: List[str] = Field(default_factory=list, description="命令参数")
    env: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    transport: MCPTransport = Field(MCPTransport.STDIO, description="传输方式")
    version: Optional[str] = Field(None, description="版本号")
    auto_start: bool = Field(True, description="是否自动启动")
    restart_on_failure: bool = Field(True, description="失败时是否重启")
    timeout: int = Field(30, description="超时时间（秒）")

class MCPConfigCreate(MCPConfigBase):
    """创建 MCP 配置请求"""
    pass

class MCPConfigUpdate(BaseModel):
    """更新 MCP 配置请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    transport: Optional[MCPTransport] = None
    version: Optional[str] = None
    auto_start: Optional[bool] = None
    restart_on_failure: Optional[bool] = None
    timeout: Optional[int] = None

class MCPConfigResponse(MCPConfigBase):
    """MCP 配置响应"""
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_error: Optional[str] = None

# ==================== LLM 相关模型 ====================

class LLMConfigBase(BaseModel):
    """LLM 配置基础模型"""
    name: str = Field(..., description="模型名称")
    provider: LLMProvider = Field(..., description="提供商")
    model_name: str = Field(..., description="实际模型名称")
    api_key: Optional[str] = Field(None, description="API 密钥")
    base_url: Optional[str] = Field(None, description="API 基础 URL")
    max_tokens: Optional[int] = Field(None, description="最大 token 数")
    temperature: float = Field(0.7, description="温度参数")
    top_p: float = Field(0.9, description="top_p 参数")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    is_default: bool = Field(False, description="是否为默认模型")
    enabled: bool = Field(True, description="是否启用")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """验证 API 密钥格式"""
        if v and len(v) < 10:
            raise ValueError('API 密钥长度不能少于10个字符')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """验证温度参数"""
        if v < 0 or v > 2:
            raise ValueError('温度参数必须在0-2之间')
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        """验证 top_p 参数"""
        if v < 0 or v > 1:
            raise ValueError('top_p 参数必须在0-1之间')
        return v

class LLMConfigCreate(LLMConfigBase):
    """创建 LLM 配置请求"""
    pass

class LLMConfigUpdate(BaseModel):
    """更新 LLM 配置请求"""
    name: Optional[str] = None
    provider: Optional[LLMProvider] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    system_prompt: Optional[str] = None
    is_default: Optional[bool] = None
    enabled: Optional[bool] = None

class LLMConfigResponse(LLMConfigBase):
    """LLM 配置响应"""
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None

# ==================== 语音处理相关模型 ====================

class SpeechRecognitionResponse(BaseResponse):
    """语音识别响应"""
    text: str
    language: str
    confidence: float
    processing_time: float
    model_used: str

class SpeechSynthesisRequest(BaseModel):
    """语音合成请求"""
    text: str = Field(..., description="要合成的文本")
    voice: Optional[str] = Field(None, description="声音ID")
    language: str = Field("zh-CN", description="语言")
    speed: float = Field(1.0, description="语速")
    pitch: float = Field(1.0, description="音调")
    tts_model: Optional[str] = Field(None, description="TTS 模型")

class SpeechSynthesisResponse(BaseResponse):
    """语音合成响应"""
    audio_data: str = Field(..., description="音频数据（base64编码）")
    format: AudioFormat
    duration: float
    processing_time: float
    model_used: str

# ==================== 对话相关模型 ====================

class ChatMessage(BaseModel):
    """对话消息"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    model_name: Optional[str] = Field(None, description="使用的模型名称")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    session_id: Optional[str] = Field(None, description="会话ID")
    stream: bool = Field(False, description="是否流式输出")
    tools: Optional[List[str]] = Field(None, description="可用工具列表")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    temperature: Optional[float] = Field(None, description="温度参数")

class ChatResponse(BaseResponse):
    """对话响应"""
    content: str
    model_name: str
    session_id: str
    tools_used: List[str] = Field(default_factory=list)
    processing_time: float
    token_usage: Dict[str, int] = Field(default_factory=dict)

class VoiceChatResponse(BaseResponse):
    """语音对话响应"""
    user_text: str
    response_text: str
    response_audio: str = Field(..., description="音频数据（base64编码）")
    audio_format: AudioFormat
    session_id: str
    model_used: str
    processing_time: Dict[str, float]

# ==================== 验证器 ====================

# @field_validator('api_key')
# def validate_api_key(cls, v):
#     """验证 API 密钥格式"""
#     if v and len(v) < 10:
#         raise ValueError('API 密钥长度不能少于10个字符')
#     return v

# @field_validator('temperature')
# def validate_temperature(cls, v):
#     """验证温度参数"""
#     if v < 0 or v > 2:
#         raise ValueError('温度参数必须在0-2之间')
#     return v

# @field_validator('top_p')
# def validate_top_p(cls, v):
#     """验证 top_p 参数"""
#     if v < 0 or v > 1:
#         raise ValueError('top_p 参数必须在0-1之间')
#     return v

# # 将验证器添加到相应的模型中
# LLMConfigBase.__validators__['validate_api_key'] = validator('api_key', allow_reuse=True)(validate_api_key)
# LLMConfigBase.__validators__['validate_temperature'] = validator('temperature', allow_reuse=True)(validate_temperature)
# LLMConfigBase.__validators__['validate_top_p'] = validator('top_p', allow_reuse=True)(validate_top_p)