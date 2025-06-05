"""
通用工具函数模块
"""

import uuid
import hashlib
import time
import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiofiles
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

def generate_response_id() -> str:
    """生成响应ID"""
    return str(uuid.uuid4())

def generate_session_id() -> str:
    """生成会话ID"""
    timestamp = int(time.time())
    random_part = str(uuid.uuid4())[:8]
    return f"session_{timestamp}_{random_part}"

def calculate_md5(data: bytes) -> str:
    """计算MD5哈希值"""
    return hashlib.md5(data).hexdigest()

def calculate_sha256(data: bytes) -> str:
    """计算SHA256哈希值"""
    return hashlib.sha256(data).hexdigest()

def safe_filename(filename: str) -> str:
    """生成安全的文件名"""
    # 移除危险字符
    safe_chars = re.sub(r'[^\w\s-]', '', filename)
    # 替换空格为下划线
    safe_chars = re.sub(r'[-\s]+', '_', safe_chars)
    return safe_chars.strip('_')

def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"

async def validate_config(config_data: Dict[str, Any], config_type: str) -> bool:
    """验证配置数据"""
    logger.debug(f"🔍 验证{config_type}配置")
    
    if config_type == "mcp":
        return await validate_mcp_config(config_data)
    elif config_type == "llm":
        return await validate_llm_config(config_data)
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")

async def validate_mcp_config(config_data: Dict[str, Any]) -> bool:
    """验证MCP配置"""
    required_fields = ["name", "command"]
    
    for field in required_fields:
        if not config_data.get(field):
            raise ValueError(f"缺少必需字段: {field}")
    
    # 验证名称格式
    name = config_data["name"]
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("名称只能包含字母、数字、下划线和连字符")
    
    # 验证传输方式
    transport = config_data.get("transport", "stdio")
    if transport not in ["stdio", "websocket", "http"]:
        raise ValueError(f"不支持的传输方式: {transport}")
    
    # 验证超时时间
    timeout = config_data.get("timeout", 30)
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("超时时间必须是正整数")
    
    return True

async def validate_llm_config(config_data: Dict[str, Any]) -> bool:
    """验证LLM配置"""
    required_fields = ["name", "provider", "model_name"]
    
    for field in required_fields:
        if not config_data.get(field):
            raise ValueError(f"缺少必需字段: {field}")
    
    # 验证提供商
    provider = config_data["provider"]
    supported_providers = ["dashscope", "xinference", "openai", "anthropic", "local"]
    if provider not in supported_providers:
        raise ValueError(f"不支持的提供商: {provider}")
    
    # 验证温度参数
    temperature = config_data.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise ValueError("温度参数必须在0-2之间")
    
    # 验证top_p参数
    top_p = config_data.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        raise ValueError("top_p参数必须在0-1之间")
    
    # 验证API密钥（如果提供）
    api_key = config_data.get("api_key")
    if api_key and len(api_key) < 10:
        raise ValueError("API密钥长度不能少于10个字符")
    
    return True

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    async def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        # 清理过期记录
        self.requests[key] = [
            req_time for req_time in self.requests.get(key, [])
            if now - req_time < self.time_window
        ]
        
        # 检查是否超过限制
        if len(self.requests.get(key, [])) >= self.max_requests:
            return False
        
        # 记录新请求
        if key not in self.requests:
            self.requests[key] = []
        self.requests[key].append(now)
        
        return True

class AsyncCache:
    """异步缓存"""
    
    def __init__(self, default_ttl: int = 300):  # 5分钟默认TTL
        self.cache = {}
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            value, expire_time = self.cache[key]
            if time.time() < expire_time:
                return value
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        if ttl is None:
            ttl = self.default_ttl
        
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)
    
    async def delete(self, key: str) -> None:
        """删除缓存值"""
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    async def cleanup_expired(self) -> None:
        """清理过期缓存"""
        now = time.time()
        expired_keys = [
            key for key, (_, expire_time) in self.cache.items()
            if now >= expire_time
        ]
        
        for key in expired_keys:
            del self.cache[key]

class SessionManager:
    """会话管理器"""
    
    def __init__(self, default_ttl: int = 3600):  # 1小时默认TTL
        self.sessions = {}
        self.default_ttl = default_ttl
    
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新会话"""
        if session_id is None:
            session_id = generate_session_id()
        
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "messages": [],
            "context": {}
        }
        
        logger.debug(f"📝 创建会话: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            # 检查是否过期
            if self.is_session_expired(session):
                await self.delete_session(session_id)
                return None
            
            # 更新最后活动时间
            session["last_activity"] = datetime.utcnow()
        
        return session
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """更新会话"""
        session = await self.get_session(session_id)
        if session:
            session.update(updates)
            session["last_activity"] = datetime.utcnow()
    
    async def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug(f"🗑️ 删除会话: {session_id}")
    
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到会话"""
        session = await self.get_session(session_id)
        if session:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            session["messages"].append(message)
    
    async def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取会话消息"""
        session = await self.get_session(session_id)
        if session:
            messages = session["messages"]
            if limit:
                return messages[-limit:]
            return messages
        return []
    
    async def cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if self.is_session_expired(session)
        ]
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"🧹 清理了 {len(expired_sessions)} 个过期会话")
    
    def is_session_expired(self, session: Dict[str, Any]) -> bool:
        """检查会话是否过期"""
        last_activity = session["last_activity"]
        return datetime.utcnow() - last_activity > timedelta(seconds=self.default_ttl)
    
    async def get_active_sessions_count(self) -> int:
        """获取活跃会话数量"""
        await self.cleanup_expired_sessions()
        return len(self.sessions)

async def save_file_async(file_path: Path, content: bytes) -> None:
    """异步保存文件"""
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

async def load_file_async(file_path: Path) -> bytes:
    """异步加载文件"""
    async with aiofiles.open(file_path, 'rb') as f:
        return await f.read()

def create_error_response(error_message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
    """创建错误响应"""
    return {
        "success": False,
        "error": {
            "message": error_message,
            "code": error_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

def create_success_response(data: Any = None, message: Optional[str] = None) -> Dict[str, Any]:
    """创建成功响应"""
    response = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    if message:
        response["message"] = message
    
    return response

# 全局实例
rate_limiter = RateLimiter()
cache = AsyncCache()
session_manager = SessionManager()