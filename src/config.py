"""
配置管理模块
负责加载、保存和管理各种配置信息
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .logger import get_logger
from .models import MCPConfigResponse, LLMConfigResponse

logger = get_logger(__name__)

# 添加自定义JSON编码器来处理datetime对象
class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理datetime对象"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.mcp_config_file = self.config_dir / "mcp_configs.json"
        self.llm_config_file = self.config_dir / "llm_configs.json"
        self.app_config_file = self.config_dir / "app_config.json"
        
        self.mcp_configs: Dict[str, Dict[str, Any]] = {}
        self.llm_configs: Dict[str, Dict[str, Any]] = {}
        self.app_config: Dict[str, Any] = {}
    
    async def initialize(self):
        """初始化配置管理器"""
        logger.info("🔧 初始化配置管理器")
        
        # 创建配置目录
        self.config_dir.mkdir(exist_ok=True)
        
        # 加载所有配置
        await self.load_all_configs()
        
        # 创建默认配置（如果不存在）
        await self.create_default_configs()
        
        logger.info("✅ 配置管理器初始化完成")
    
    async def load_all_configs(self):
        """加载所有配置文件"""
        logger.info("📖 加载配置文件")
        
        # 加载MCP配置
        if self.mcp_config_file.exists():
            try:
                with open(self.mcp_config_file, 'r', encoding='utf-8') as f:
                    self.mcp_configs = json.load(f)
                logger.info(f"✅ 加载了 {len(self.mcp_configs)} 个MCP配置")
            except Exception as e:
                logger.error(f"❌ 加载MCP配置失败: {e}")
                self.mcp_configs = {}
        
        # 加载LLM配置
        if self.llm_config_file.exists():
            try:
                with open(self.llm_config_file, 'r', encoding='utf-8') as f:
                    self.llm_configs = json.load(f)
                logger.info(f"✅ 加载了 {len(self.llm_configs)} 个LLM配置")
            except Exception as e:
                logger.error(f"❌ 加载LLM配置失败: {e}")
                self.llm_configs = {}
        
        # 加载应用配置
        if self.app_config_file.exists():
            try:
                with open(self.app_config_file, 'r', encoding='utf-8') as f:
                    self.app_config = json.load(f)
                logger.info("✅ 加载应用配置")
            except Exception as e:
                logger.error(f"❌ 加载应用配置失败: {e}")
                self.app_config = {}
    
    async def create_default_configs(self):
        """创建默认配置"""
        
        # 默认应用配置
        if not self.app_config:
            self.app_config = {
                "version": "2.0.0",
                "default_llm_model": None,
                "default_language": "zh-CN",
                "log_level": "INFO",
                "max_session_duration": 3600,  # 1小时
                "created_at": datetime.utcnow().isoformat()
            }
            await self.save_app_config()
        
        # 默认LLM配置（如果环境变量存在）
        if not self.llm_configs and os.getenv("DASHSCOPE_API_KEY"):
            default_llm_config = {
                "id": str(uuid.uuid4()),
                "name": "默认千问模型",
                "provider": "dashscope",
                "model_name": "qwen-plus",
                "api_key": os.getenv("DASHSCOPE_API_KEY"),
                "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000,
                "is_default": True,
                "enabled": True,
                "status": "inactive",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.llm_configs[default_llm_config["id"]] = default_llm_config
            await self.save_llm_configs()
            
            # 设置为默认模型
            self.app_config["default_llm_model"] = default_llm_config["id"]
            await self.save_app_config()
            
            logger.info("✅ 创建默认LLM配置")
    
    async def save_app_config(self):
        """保存应用配置"""
        try:
            with open(self.app_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.app_config, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            logger.debug("💾 应用配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存应用配置失败: {e}")
            raise
    
    async def save_mcp_configs(self):
        """保存MCP配置"""
        try:
            with open(self.mcp_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.mcp_configs, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            logger.debug("💾 MCP配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存MCP配置失败: {e}")
            raise
    
    async def save_llm_configs(self):
        """保存LLM配置"""
        try:
            with open(self.llm_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_configs, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            logger.debug("💾 LLM配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存LLM配置失败: {e}")
            raise
    
    # MCP配置方法
    async def save_mcp_config(self, config: MCPConfigResponse):
        """保存单个MCP配置"""
        config_dict = config.dict()
        config_dict["updated_at"] = datetime.utcnow().isoformat()
        
        self.mcp_configs[config.id] = config_dict
        await self.save_mcp_configs()
        
        logger.info(f"💾 保存MCP配置: {config.name}")
    
    async def delete_mcp_config(self, config_id: str):
        """删除MCP配置"""
        if config_id in self.mcp_configs:
            config_name = self.mcp_configs[config_id].get("name", config_id)
            del self.mcp_configs[config_id]
            await self.save_mcp_configs()
            
            logger.info(f"🗑️ 删除MCP配置: {config_name}")
        else:
            logger.warning(f"⚠️ 尝试删除不存在的MCP配置: {config_id}")
    
    def get_mcp_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有MCP配置"""
        return self.mcp_configs.copy()
    
    def get_mcp_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取指定MCP配置"""
        return self.mcp_configs.get(config_id)
    
    # LLM配置方法
    async def save_llm_config(self, config: LLMConfigResponse):
        """保存单个LLM配置"""
        config_dict = config.dict()
        config_dict["updated_at"] = datetime.utcnow().isoformat()
        
        # 如果设置为默认模型，取消其他模型的默认状态
        if config_dict.get("is_default", False):
            for cfg in self.llm_configs.values():
                cfg["is_default"] = False
            
            # 更新应用配置中的默认模型
            self.app_config["default_llm_model"] = config.id
            await self.save_app_config()
        
        self.llm_configs[config.id] = config_dict
        await self.save_llm_configs()
        
        logger.info(f"💾 保存LLM配置: {config.name}")
    
    async def delete_llm_config(self, config_id: str):
        """删除LLM配置"""
        if config_id in self.llm_configs:
            config_name = self.llm_configs[config_id].get("name", config_id)
            
            # 如果删除的是默认模型，清除默认设置
            if self.app_config.get("default_llm_model") == config_id:
                self.app_config["default_llm_model"] = None
                await self.save_app_config()
            
            del self.llm_configs[config_id]
            await self.save_llm_configs()
            
            logger.info(f"🗑️ 删除LLM配置: {config_name}")
        else:
            logger.warning(f"⚠️ 尝试删除不存在的LLM配置: {config_id}")
    
    def get_llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有LLM配置"""
        return self.llm_configs.copy()
    
    def get_llm_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取指定LLM配置"""
        return self.llm_configs.get(config_id)
    
    def get_default_llm_config(self) -> Optional[Dict[str, Any]]:
        """获取默认LLM配置"""
        default_id = self.app_config.get("default_llm_model")
        if default_id:
            return self.get_llm_config(default_id)
        
        # 如果没有设置默认模型，返回第一个启用的模型
        for config in self.llm_configs.values():
            if config.get("enabled", True):
                return config
        
        return None
    
    # 应用配置方法
    def get_app_config(self) -> Dict[str, Any]:
        """获取应用配置"""
        return self.app_config.copy()
    
    async def update_app_config(self, updates: Dict[str, Any]):
        """更新应用配置"""
        self.app_config.update(updates)
        await self.save_app_config()
        
        logger.info("🔄 应用配置已更新")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查配置文件是否可读写
            test_file = self.config_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            return {
                "healthy": True,
                "mcp_configs_count": len(self.mcp_configs),
                "llm_configs_count": len(self.llm_configs),
                "config_dir": str(self.config_dir)
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 配置管理器清理完成")