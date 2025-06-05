"""
MCP (Model Context Protocol) 管理器
负责管理和运行 MCP 工具
"""

import asyncio
import json
import subprocess
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .logger import get_logger, log_mcp_operation
from .models import MCPConfigCreate, MCPConfigUpdate, MCPConfigResponse, MCPTransport
from .utils import generate_response_id

logger = get_logger(__name__)

class MCPClient:
    """MCP 客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.last_error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        
    async def start(self) -> bool:
        """启动 MCP 客户端"""
        if self.is_running:
            return True
            
        try:
            logger.info(f"🚀 启动 MCP 客户端: {self.config['name']}")
            
            # 构建命令
            cmd = [self.config['command']] + self.config.get('args', [])
            env = self.config.get('env', {})
            
            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**env} if env else None,
                cwd=self.config.get('working_dir')
            )
            
            self.is_running = True
            self.start_time = datetime.utcnow()
            self.last_error = None
            
            # 测试连接
            test_result = await self.test_connection()
            if not test_result:
                await self.stop()
                return False
            
            log_mcp_operation(
                logger, "start", self.config['name'], "success",
                {"command": ' '.join(cmd)}
            )
            
            return True
            
        except Exception as e:
            error_msg = f"启动失败: {str(e)}"
            self.last_error = error_msg
            logger.error(f"❌ MCP 客户端启动失败 [{self.config['name']}]: {error_msg}")
            
            log_mcp_operation(
                logger, "start", self.config['name'], "failed",
                {"error": error_msg}
            )
            
            return False
    
    async def stop(self):
        """停止 MCP 客户端"""
        if not self.is_running or not self.process:
            return
        
        try:
            logger.info(f"🛑 停止 MCP 客户端: {self.config['name']}")
            
            # 优雅关闭
            if self.process.stdin:
                self.process.stdin.close()
            
            # 等待进程结束
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制终止
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            
            self.is_running = False
            self.process = None
            
            log_mcp_operation(
                logger, "stop", self.config['name'], "success"
            )
            
        except Exception as e:
            logger.error(f"❌ 停止 MCP 客户端失败 [{self.config['name']}]: {str(e)}")
    
    async def test_connection(self, timeout: int = 10) -> bool:
        """测试连接"""
        if not self.is_running or not self.process:
            return False
        
        try:
            # 发送测试请求
            test_request = {
                "action": "list_functions"
            }
            
            request_json = json.dumps(test_request) + '\n'
            
            # 发送请求
            if self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()
            
            # 等待响应
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.process.stdout and self.process.stdout.readable():
                    try:
                        # 非阻塞读取
                        import select
                        ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                        if ready:
                            response_line = self.process.stdout.readline()
                            if response_line.strip():
                                response = json.loads(response_line.strip())
                                if "functions" in response:
                                    return True
                    except (json.JSONDecodeError, OSError):
                        pass
                
                await asyncio.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ MCP 连接测试失败 [{self.config['name']}]: {str(e)}")
            return False
    
    async def call_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 MCP 函数"""
        if not self.is_running or not self.process:
            return {"error": "MCP 客户端未运行"}
        
        try:
            request = {
                "action": "call_function",
                "name": function_name,
                "arguments": arguments
            }
            
            request_json = json.dumps(request) + '\n'
            
            # 发送请求
            if self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()
            
            # 读取响应
            if self.process.stdout:
                response_line = self.process.stdout.readline()
                if response_line.strip():
                    response = json.loads(response_line.strip())
                    
                    log_mcp_operation(
                        logger, "function_call", self.config['name'], "success",
                        {"function": function_name, "args": arguments}
                    )
                    
                    return response
            
            return {"error": "无法读取响应"}
            
        except Exception as e:
            error_msg = f"函数调用失败: {str(e)}"
            
            log_mcp_operation(
                logger, "function_call", self.config['name'], "failed",
                {"function": function_name, "error": error_msg}
            )
            
            return {"error": error_msg}
    
    async def list_functions(self) -> Dict[str, Any]:
        """列出可用函数"""
        if not self.is_running or not self.process:
            return {"error": "MCP 客户端未运行"}
        
        try:
            request = {"action": "list_functions"}
            request_json = json.dumps(request) + '\n'
            
            # 发送请求
            if self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()
            
            # 读取响应
            if self.process.stdout:
                response_line = self.process.stdout.readline()
                if response_line.strip():
                    return json.loads(response_line.strip())
            
            return {"error": "无法读取响应"}
            
        except Exception as e:
            return {"error": f"列出函数失败: {str(e)}"}
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "name": self.config['name'],
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_error": self.last_error,
            "process_id": self.process.pid if self.process else None
        }

class MCPManager:
    """MCP 管理器"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.is_initialized = False
    
    async def initialize(self, configs: Dict[str, Dict[str, Any]]):
        """初始化 MCP 管理器"""
        logger.info("🔧 初始化 MCP 管理器")
        
        self.configs = configs.copy()
        
        # 启动自动启动的 MCP 客户端
        for config_id, config in self.configs.items():
            if config.get('auto_start', True):
                await self.start_client(config_id)
        
        self.is_initialized = True
        logger.info(f"✅ MCP 管理器初始化完成 - 共 {len(self.configs)} 个配置")
    
    async def start_client(self, config_id: str) -> bool:
        """启动指定的 MCP 客户端"""
        if config_id not in self.configs:
            logger.error(f"❌ MCP 配置不存在: {config_id}")
            return False
        
        config = self.configs[config_id]
        
        # 如果客户端已存在，先停止
        if config_id in self.clients:
            await self.clients[config_id].stop()
        
        # 创建新客户端
        client = MCPClient(config)
        success = await client.start()
        
        if success:
            self.clients[config_id] = client
            logger.info(f"✅ MCP 客户端启动成功: {config['name']}")
        else:
            logger.error(f"❌ MCP 客户端启动失败: {config['name']}")
        
        return success
    
    async def stop_client(self, config_id: str):
        """停止指定的 MCP 客户端"""
        if config_id in self.clients:
            await self.clients[config_id].stop()
            del self.clients[config_id]
            logger.info(f"🛑 MCP 客户端已停止: {config_id}")
    
    async def restart_client(self, config_id: str) -> bool:
        """重启指定的 MCP 客户端"""
        logger.info(f"🔄 重启 MCP 客户端: {config_id}")
        
        await self.stop_client(config_id)
        return await self.start_client(config_id)
    
    async def get_all_configs(self) -> List[MCPConfigResponse]:
        """获取所有 MCP 配置"""
        configs = []
        
        for config_id, config in self.configs.items():
            client_status = self.clients[config_id].get_status() if config_id in self.clients else None
            
            config_response = MCPConfigResponse(
                id=config_id,
                name=config['name'],
                description=config.get('description', ''),
                command=config['command'],
                args=config.get('args', []),
                env=config.get('env', {}),
                transport=MCPTransport(config.get('transport', 'stdio')),
                version=config.get('version'),
                auto_start=config.get('auto_start', True),
                restart_on_failure=config.get('restart_on_failure', True),
                timeout=config.get('timeout', 30),
                status="running" if client_status and client_status['is_running'] else "stopped",
                created_at=datetime.fromisoformat(config.get('created_at', datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(config.get('updated_at', datetime.utcnow().isoformat())),
                last_error=client_status.get('last_error') if client_status else None
            )
            
            configs.append(config_response)
        
        return configs
    
    async def create_config(self, config_create: MCPConfigCreate) -> MCPConfigResponse:
        """创建 MCP 配置"""
        config_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        config_dict = {
            "id": config_id,
            "name": config_create.name,
            "description": config_create.description or "",
            "command": config_create.command,
            "args": config_create.args,
            "env": config_create.env,
            "transport": config_create.transport.value,
            "version": config_create.version,
            "auto_start": config_create.auto_start,
            "restart_on_failure": config_create.restart_on_failure,
            "timeout": config_create.timeout,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        self.configs[config_id] = config_dict
        
        # 如果设置为自动启动，立即启动
        if config_create.auto_start:
            await self.start_client(config_id)
        
        return MCPConfigResponse(
            id=config_id,
            status="running" if config_create.auto_start else "stopped",
            created_at=now,
            updated_at=now,
            **config_create.dict()
        )
    
    async def update_config(self, config_id: str, config_update: MCPConfigUpdate) -> MCPConfigResponse:
        """更新 MCP 配置"""
        if config_id not in self.configs:
            raise ValueError(f"MCP 配置不存在: {config_id}")
        
        config = self.configs[config_id].copy()
        
        # 更新字段
        update_dict = config_update.dict(exclude_unset=True)
        for key, value in update_dict.items():
            if key == "transport" and value:
                config[key] = value.value
            elif value is not None:
                config[key] = value
        
        config["updated_at"] = datetime.utcnow().isoformat()
        self.configs[config_id] = config
        
        # 如果客户端正在运行，重启以应用新配置
        if config_id in self.clients:
            await self.restart_client(config_id)
        
        return MCPConfigResponse(
            id=config_id,
            status="running" if config_id in self.clients else "stopped",
            created_at=datetime.fromisoformat(config['created_at']),
            updated_at=datetime.fromisoformat(config['updated_at']),
            name=config['name'],
            description=config.get('description', ''),
            command=config['command'],
            args=config.get('args', []),
            env=config.get('env', {}),
            transport=MCPTransport(config.get('transport', 'stdio')),
            version=config.get('version'),
            auto_start=config.get('auto_start', True),
            restart_on_failure=config.get('restart_on_failure', True),
            timeout=config.get('timeout', 30)
        )
    
    async def delete_config(self, config_id: str):
        """删除 MCP 配置"""
        if config_id in self.configs:
            # 停止客户端
            await self.stop_client(config_id)
            
            # 删除配置
            del self.configs[config_id]
            
            logger.info(f"🗑️ 删除 MCP 配置: {config_id}")
    
    async def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取指定 MCP 配置"""
        return self.configs.get(config_id)
    
    async def test_config(self, config_id: str) -> Dict[str, Any]:
        """测试 MCP 配置"""
        if config_id not in self.configs:
            return {"success": False, "error": "配置不存在"}
        
        config = self.configs[config_id]
        
        try:
            # 创建临时客户端进行测试
            test_client = MCPClient(config)
            success = await test_client.start()
            
            if success:
                # 测试列出函数
                functions_result = await test_client.list_functions()
                await test_client.stop()
                
                return {
                    "success": True,
                    "message": "连接测试成功",
                    "functions": functions_result.get("functions", [])
                }
            else:
                return {
                    "success": False,
                    "error": test_client.last_error or "启动失败"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"测试失败: {str(e)}"
            }
    
    async def call_tool_function(self, config_id: str, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具函数"""
        if config_id not in self.clients:
            return {"error": f"MCP 客户端未运行: {config_id}"}
        
        client = self.clients[config_id]
        return await client.call_function(function_name, arguments)
    
    async def get_tool_functions(self, config_id: str) -> Dict[str, Any]:
        """获取工具函数列表"""
        if config_id not in self.clients:
            return {"error": f"MCP 客户端未运行: {config_id}"}
        
        client = self.clients[config_id]
        return await client.list_functions()
    
    async def get_tools_status(self) -> Dict[str, Any]:
        """获取所有工具状态"""
        status = {
            "total_configs": len(self.configs),
            "running_clients": len(self.clients),
            "tools": {}
        }
        
        for config_id, config in self.configs.items():
            client_status = None
            if config_id in self.clients:
                client_status = self.clients[config_id].get_status()
            
            status["tools"][config_id] = {
                "name": config['name'],
                "is_running": client_status is not None and client_status['is_running'],
                "last_error": client_status.get('last_error') if client_status else None
            }
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            running_count = len(self.clients)
            total_count = len(self.configs)
            
            # 检查是否有失败的客户端
            failed_clients = []
            for config_id in self.configs:
                if config_id not in self.clients and self.configs[config_id].get('auto_start', True):
                    failed_clients.append(config_id)
            
            return {
                "healthy": len(failed_clients) == 0,
                "total_configs": total_count,
                "running_clients": running_count,
                "failed_clients": failed_clients
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理 MCP 管理器资源")
        
        # 停止所有客户端
        for config_id in list(self.clients.keys()):
            await self.stop_client(config_id)
        
        self.clients.clear()
        self.configs.clear()
        self.is_initialized = False
        
        logger.info("✅ MCP 管理器资源清理完成")