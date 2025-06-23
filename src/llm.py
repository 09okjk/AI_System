"""
LLM (Large Language Model) 管理器
负责管理和调用各种 LLM 模型
"""

import asyncio
import uuid
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import openai
from anthropic import Anthropic
import aiohttp

from .logger import get_logger, log_llm_call
from .models import LLMConfigCreate, LLMConfigUpdate, LLMConfigResponse, LLMProvider, ChatMessage
from .utils import generate_response_id, session_manager

logger = get_logger(__name__)

class LLMClient:
    """LLM 客户端基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """初始化客户端"""
        try:
            await self._setup_client()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ LLM 客户端初始化失败 [{self.config['name']}]: {str(e)}")
            return False
    
    async def _setup_client(self):
        """设置客户端（由子类实现）"""
        raise NotImplementedError
    
    async def chat(self, 
                  message: str, 
                  system_prompt: Optional[str] = None,
                  history: Optional[List[ChatMessage]] = None,
                  **kwargs) -> Dict[str, Any]:
        """聊天（由子类实现）"""
        raise NotImplementedError
    
    async def stream_chat(self, 
                         message: str, 
                         system_prompt: Optional[str] = None,
                         history: Optional[List[ChatMessage]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """流式聊天（由子类实现）"""
        raise NotImplementedError
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            start_time = time.time()
            result = await self.chat("Hello", system_prompt="Respond with 'OK'")
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "连接测试成功",
                "response": result.get("content", ""),
                "processing_time": processing_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class DashScopeClient(LLMClient):
    """阿里云 DashScope 客户端"""
    
    async def _setup_client(self):
        """设置 DashScope 客户端"""
        api_key = self.config.get('api_key')
        base_url = self.config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        
        if not api_key:
            raise ValueError("DashScope API key is required")
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def chat(self, 
                  message: str, 
                  system_prompt: Optional[str] = None,
                  history: Optional[List[ChatMessage]] = None,
                  **kwargs) -> Dict[str, Any]:
        """DashScope 聊天"""
        try:
            start_time = time.time()
            
            # 构建消息列表
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            # 添加历史消息
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # 添加用户消息
            messages.append({
                "role": "user",
                "content": message
            })
            
            # 调用 API
            response = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            # 提取响应内容
            content = response.choices[0].message.content
            
            # 计算 token 使用量
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # 记录日志
            log_llm_call(
                logger, self.config['name'], 
                token_usage["prompt_tokens"], 
                token_usage["completion_tokens"],
                processing_time, True
            )
            
            return {
                "content": content,
                "model_name": self.config['name'],
                "processing_time": processing_time,
                "token_usage": token_usage
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            raise e
    
    async def stream_chat(self, 
                         message: str, 
                         system_prompt: Optional[str] = None,
                         history: Optional[List[ChatMessage]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """DashScope 流式聊天"""
        try:
            start_time = time.time()
            
            # 构建消息列表
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            # 流式调用 API
            stream = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                stream=True,
                **kwargs
            )
            
            full_content = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    
                    yield {
                        "type": "content_chunk",
                        "content": content_chunk,
                        "full_content": full_content
                    }
            
            processing_time = time.time() - start_time
            
            # 发送完成信号
            yield {
                "type": "done",
                "content": full_content,
                "model_name": self.config['name'],
                "processing_time": processing_time
            }
            
            log_llm_call(
                logger, self.config['name'], 0, len(full_content), 
                processing_time, True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            yield {
                "type": "error",
                "error": error_msg
            }

class XinferenceClient(LLMClient):
    """Xinference 客户端"""
    
    async def _setup_client(self):
        """设置 Xinference 客户端"""
        base_url = self.config.get('base_url', 'http://localhost:9997/v1')
        
        self.client = openai.AsyncOpenAI(
            api_key="xinference",  # Xinference 不需要真实的 API key
            base_url=base_url
        )
    
    async def chat(self, 
                  message: str, 
                  system_prompt: Optional[str] = None,
                  history: Optional[List[ChatMessage]] = None,
                  **kwargs) -> Dict[str, Any]:
        """Xinference 聊天"""
        try:
            start_time = time.time()
            
            # 构建消息列表
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            logger.info(f"调用 LLM 模型 [请求ID: {request_id}], messages: {messages}")
            # 调用 API
            response = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            content = response.choices[0].message.content
            
            # Xinference 可能不返回详细的 token 使用信息
            token_usage = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            }
            
            log_llm_call(
                logger, self.config['name'], 
                token_usage["prompt_tokens"], 
                token_usage["completion_tokens"],
                processing_time, True
            )
            
            return {
                "content": content,
                "model_name": self.config['name'],
                "processing_time": processing_time,
                "token_usage": token_usage
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            raise e
    
    async def stream_chat(self, 
                         message: str, 
                         system_prompt: Optional[str] = None,
                         history: Optional[List[ChatMessage]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Xinference 流式聊天"""
        # 实现与 DashScope 类似的流式逻辑
        try:
            start_time = time.time()
            
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            stream = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                stream=True,
                **kwargs
            )
            
            full_content = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    
                    yield {
                        "type": "content_chunk",
                        "content": content_chunk,
                        "full_content": full_content
                    }
            
            processing_time = time.time() - start_time
            
            yield {
                "type": "done",
                "content": full_content,
                "model_name": self.config['name'],
                "processing_time": processing_time
            }
            
            log_llm_call(
                logger, self.config['name'], 0, len(full_content), 
                processing_time, True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            yield {
                "type": "error",
                "error": error_msg
            }

class OpenAIClient(LLMClient):
    """OpenAI 客户端"""
    
    async def _setup_client(self):
        """设置 OpenAI 客户端"""
        api_key = self.config.get('api_key')
        base_url = self.config.get('base_url')
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        
        self.client = openai.AsyncOpenAI(**kwargs)
    
    async def chat(self, 
                  message: str, 
                  system_prompt: Optional[str] = None,
                  history: Optional[List[ChatMessage]] = None,
                  **kwargs) -> Dict[str, Any]:
        """OpenAI 聊天"""
        # 实现与 DashScope 类似的逻辑
        try:
            start_time = time.time()
            
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            response = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            content = response.choices[0].message.content
            
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            log_llm_call(
                logger, self.config['name'], 
                token_usage["prompt_tokens"], 
                token_usage["completion_tokens"],
                processing_time, True
            )
            
            return {
                "content": content,
                "model_name": self.config['name'],
                "processing_time": processing_time,
                "token_usage": token_usage
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            raise e
    
    async def stream_chat(self, 
                         message: str, 
                         system_prompt: Optional[str] = None,
                         history: Optional[List[ChatMessage]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """OpenAI 流式聊天"""
        # 与其他客户端类似的实现
        try:
            start_time = time.time()
            
            messages = []
            
            if system_prompt or self.config.get('system_prompt'):
                messages.append({
                    "role": "system",
                    "content": system_prompt or self.config.get('system_prompt')
                })
            
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            stream = await self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                max_tokens=self.config.get('max_tokens', 2000),
                stream=True,
                **kwargs
            )
            
            full_content = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    
                    yield {
                        "type": "content_chunk",
                        "content": content_chunk,
                        "full_content": full_content
                    }
            
            processing_time = time.time() - start_time
            
            yield {
                "type": "done",
                "content": full_content,
                "model_name": self.config['name'],
                "processing_time": processing_time
            }
            
            log_llm_call(
                logger, self.config['name'], 0, len(full_content), 
                processing_time, True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_llm_call(
                logger, self.config['name'], 0, 0, 
                processing_time, False, error_msg
            )
            
            yield {
                "type": "error",
                "error": error_msg
            }

class LLMManager:
    """LLM 管理器"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, LLMClient] = {}
        self.is_initialized = False
        
        # 客户端类型映射
        self.client_classes = {
            LLMProvider.DASHSCOPE: DashScopeClient,
            LLMProvider.XINFERENCE: XinferenceClient,
            LLMProvider.OPENAI: OpenAIClient,
            # 可以添加更多提供商
        }
    
    async def initialize(self, configs: Dict[str, Dict[str, Any]]):
        """初始化 LLM 管理器"""
        logger.info("🔧 初始化 LLM 管理器")
        
        self.configs = configs.copy()
        
        # 初始化启用的 LLM 客户端
        for config_id, config in self.configs.items():
            if config.get('enabled', True):
                await self.initialize_client(config_id)
        
        self.is_initialized = True
        logger.info(f"✅ LLM 管理器初始化完成 - 共 {len(self.configs)} 个配置")
    
    async def initialize_client(self, config_id: str) -> bool:
        """初始化指定的 LLM 客户端"""
        if config_id not in self.configs:
            logger.error(f"❌ LLM 配置不存在: {config_id}")
            return False
        
        config = self.configs[config_id]
        provider = LLMProvider(config['provider'])
        
        if provider not in self.client_classes:
            logger.error(f"❌ 不支持的 LLM 提供商: {provider}")
            return False
        
        try:
            # 创建客户端
            client_class = self.client_classes[provider]
            client = client_class(config)
            
            # 初始化客户端
            success = await client.initialize()
            
            if success:
                self.clients[config_id] = client
                logger.info(f"✅ LLM 客户端初始化成功: {config['name']}")
                
                # 更新配置状态
                self.configs[config_id]['status'] = 'active'
            else:
                logger.error(f"❌ LLM 客户端初始化失败: {config['name']}")
                self.configs[config_id]['status'] = 'error'
            
            return success
            
        except Exception as e:
            logger.error(f"❌ LLM 客户端初始化异常 [{config['name']}]: {str(e)}")
            self.configs[config_id]['status'] = 'error'
            return False
    
    async def get_all_configs(self) -> List[LLMConfigResponse]:
        """获取所有 LLM 配置"""
        configs = []
        
        for config_id, config in self.configs.items():
            config_response = LLMConfigResponse(
                id=config_id,
                name=config['name'],
                provider=LLMProvider(config['provider']),
                model_name=config['model_name'],
                api_key=config.get('api_key', ''),
                base_url=config.get('base_url'),
                max_tokens=config.get('max_tokens'),
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9),
                system_prompt=config.get('system_prompt'),
                is_default=config.get('is_default', False),
                enabled=config.get('enabled', True),
                status=config.get('status', 'inactive'),
                created_at=datetime.fromisoformat(config.get('created_at', datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(config.get('updated_at', datetime.utcnow().isoformat())),
                last_used=datetime.fromisoformat(config['last_used']) if config.get('last_used') else None
            )
            
            configs.append(config_response)
        
        return configs
    
    async def create_config(self, config_create: LLMConfigCreate) -> LLMConfigResponse:
        """创建 LLM 配置"""
        config_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        config_dict = {
            "id": config_id,
            "name": config_create.name,
            "provider": config_create.provider.value,
            "model_name": config_create.model_name,
            "api_key": config_create.api_key,
            "base_url": config_create.base_url,
            "max_tokens": config_create.max_tokens,
            "temperature": config_create.temperature,
            "top_p": config_create.top_p,
            "system_prompt": config_create.system_prompt,
            "is_default": config_create.is_default,
            "enabled": config_create.enabled,
            "status": "inactive",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        self.configs[config_id] = config_dict
        
        # 如果设置为启用，立即初始化
        if config_create.enabled:
            await self.initialize_client(config_id)
        
        return LLMConfigResponse(
            id=config_id,
            status=self.configs[config_id]['status'],
            created_at=now,
            updated_at=now,
            **config_create.dict()
        )
    
    async def update_config(self, config_id: str, config_update: LLMConfigUpdate) -> LLMConfigResponse:
        """更新 LLM 配置"""
        if config_id not in self.configs:
            raise ValueError(f"LLM 配置不存在: {config_id}")
        
        config = self.configs[config_id].copy()
        
        # 更新字段
        update_dict = config_update.dict(exclude_unset=True)
        for key, value in update_dict.items():
            if key == "provider" and value:
                config[key] = value.value
            elif value is not None:
                config[key] = value
        
        config["updated_at"] = datetime.utcnow().isoformat()
        self.configs[config_id] = config
        
        # 如果客户端已初始化，重新初始化以应用新配置
        if config_id in self.clients:
            del self.clients[config_id]
            if config.get('enabled', True):
                await self.initialize_client(config_id)
        
        return LLMConfigResponse(
            id=config_id,
            status=self.configs[config_id]['status'],
            created_at=datetime.fromisoformat(config['created_at']),
            updated_at=datetime.fromisoformat(config['updated_at']),
            name=config['name'],
            provider=LLMProvider(config['provider']),
            model_name=config['model_name'],
            api_key=config.get('api_key', ''),
            base_url=config.get('base_url'),
            max_tokens=config.get('max_tokens'),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9),
            system_prompt=config.get('system_prompt'),
            is_default=config.get('is_default', False),
            enabled=config.get('enabled', True),
            last_used=datetime.fromisoformat(config['last_used']) if config.get('last_used') else None
        )
    
    async def delete_config(self, config_id: str):
        """删除 LLM 配置"""
        if config_id in self.configs:
            # 停止客户端
            if config_id in self.clients:
                del self.clients[config_id]
            
            # 删除配置
            del self.configs[config_id]
            
            logger.info(f"🗑️ 删除 LLM 配置: {config_id}")
    
    async def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取指定 LLM 配置"""
        return self.configs.get(config_id)
    
    async def test_config(self, config_id: str) -> Dict[str, Any]:
        """测试 LLM 配置"""
        if config_id not in self.configs:
            return {"success": False, "error": "配置不存在"}
        
        config = self.configs[config_id]
        provider = LLMProvider(config['provider'])
        
        if provider not in self.client_classes:
            return {"success": False, "error": f"不支持的提供商: {provider}"}
        
        try:
            # 创建临时客户端进行测试
            client_class = self.client_classes[provider]
            test_client = client_class(config)
            
            success = await test_client.initialize()
            if not success:
                return {"success": False, "error": "客户端初始化失败"}
            
            # 测试连接
            test_result = await test_client.test_connection()
            return test_result
        
        except Exception as e:
            return {"success": False, "error": f"测试失败: {str(e)}"}
    
    async def chat(self, 
                  model_name: Optional[str],
                  message: str,
                  system_prompt: Optional[str] = None,
                  session_id: Optional[str] = None,
                  stream: bool = False,
                  tools: Optional[List[str]] = None,
                  request_id: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """聊天"""
        
        # 选择模型
        client = await self._get_client(model_name)
        if not client:
            raise ValueError(f"模型不可用: {model_name}")
        
        # 获取会话历史
        history = []
        if session_id:
            history_msgs = await session_manager.get_messages(session_id, limit=10)
            history = [ChatMessage(role=msg['role'], content=msg['content']) for msg in history_msgs]
        
        try:
            # 执行聊天
            result = await client.chat(
                message=message,
                system_prompt=system_prompt,
                history=history,
                **kwargs
            )
            
            # 保存到会话
            if session_id:
                await session_manager.add_message(session_id, "user", message)
                await session_manager.add_message(session_id, "assistant", result["content"])
            
            # 更新配置使用时间
            config_id = self._get_config_id_by_client(client)
            if config_id:
                self.configs[config_id]['last_used'] = datetime.utcnow().isoformat()
            
            result['session_id'] = session_id or generate_response_id()
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM 聊天失败: {str(e)}")
            raise
    
    async def stream_chat(self, 
                         model_name: Optional[str],
                         message: str,
                         system_prompt: Optional[str] = None,
                         session_id: Optional[str] = None,
                         tools: Optional[List[str]] = None,
                         request_id: Optional[str] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """流式聊天"""
        
        # 选择模型
        client = await self._get_client(model_name)
        if not client:
            yield {"type": "error", "error": f"模型不可用: {model_name}"}
            return
        
        # 获取会话历史
        history = []
        if session_id:
            history_msgs = await session_manager.get_messages(session_id, limit=10)
            history = [ChatMessage(role=msg['role'], content=msg['content']) for msg in history_msgs]
        
        try:
            full_content = ""
            
            async for chunk in client.stream_chat(
                message=message,
                system_prompt=system_prompt,
                history=history,
                **kwargs
            ):
                if chunk.get("type") == "content_chunk":
                    full_content = chunk.get("full_content", "")
                
                chunk['session_id'] = session_id or generate_response_id()
                yield chunk
            
            # 保存到会话
            if session_id and full_content:
                await session_manager.add_message(session_id, "user", message)
                await session_manager.add_message(session_id, "assistant", full_content)
            
            # 更新配置使用时间
            config_id = self._get_config_id_by_client(client)
            if config_id:
                self.configs[config_id]['last_used'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"❌ LLM 流式聊天失败: {str(e)}")
            yield {"type": "error", "error": str(e)}
    
    async def _get_client(self, model_name: Optional[str]) -> Optional[LLMClient]:
        """获取客户端"""
        if model_name:
            # 根据模型名称查找
            for config_id, config in self.configs.items():
                if config['name'] == model_name and config_id in self.clients:
                    return self.clients[config_id]
        
        # 使用默认模型
        for config_id, config in self.configs.items():
            if config.get('is_default') and config_id in self.clients:
                return self.clients[config_id]
        
        # 使用第一个可用模型
        for config_id in self.clients:
            return self.clients[config_id]
        
        return None
    
    def _get_config_id_by_client(self, client: LLMClient) -> Optional[str]:
        """根据客户端获取配置ID"""
        for config_id, c in self.clients.items():
            if c == client:
                return config_id
        return None
    
    async def get_models_status(self) -> Dict[str, Any]:
        """获取所有模型状态"""
        status = {
            "total_configs": len(self.configs),
            "active_clients": len(self.clients),
            "models": {}
        }
        
        for config_id, config in self.configs.items():
            status["models"][config_id] = {
                "name": config['name'],
                "provider": config['provider'],
                "is_active": config_id in self.clients,
                "is_default": config.get('is_default', False),
                "enabled": config.get('enabled', True),
                "last_used": config.get('last_used')
            }
        
        return status
    
    async def get_active_sessions_count(self) -> int:
        """获取活跃会话数量"""
        return await session_manager.get_active_sessions_count()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            enabled_count = len([c for c in self.configs.values() if c.get('enabled', True)])
            active_count = len(self.clients)
            
            return {
                "healthy": active_count > 0,
                "total_configs": len(self.configs),
                "enabled_configs": enabled_count,
                "active_clients": active_count
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理 LLM 管理器资源")
        
        # 清理所有客户端
        self.clients.clear()
        self.configs.clear()
        self.is_initialized = False
        
        logger.info("✅ LLM 管理器资源清理完成")