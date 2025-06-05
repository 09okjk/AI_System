#!/usr/bin/env python3
"""
AI Agent 后端测试运行器
提供完整的测试功能，包括单元测试、集成测试和性能测试
"""

import asyncio
import pytest
import sys
import os
import time
import json
import requests
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import base64

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger, get_logger
from src.config import ConfigManager
from src.mcp import MCPManager
from src.llm import LLMManager
from src.speech import SpeechProcessor

# 设置测试日志
setup_logger(log_level="DEBUG", log_dir="test_logs", app_name="ai_agent_test")
logger = get_logger(__name__)

class TestRunner:
    """测试运行器"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = None
        self.test_results = []
    
    async def setup(self):
        """初始化测试环境"""
        logger.info("🔧 初始化测试环境")
        
        self.session = aiohttp.ClientSession()
        
        # 等待服务器启动
        await self.wait_for_server()
        
        logger.info("✅ 测试环境初始化完成")
    
    async def cleanup(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境")
        
        if self.session:
            await self.session.close()
        
        logger.info("✅ 测试环境清理完成")
    
    async def wait_for_server(self, timeout: int = 30):
        """等待服务器启动"""
        logger.info("⏳ 等待服务器启动...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{self.server_url}/api/health") as response:
                    if response.status == 200:
                        logger.info("✅ 服务器已启动")
                        return
            except:
                pass
            
            await asyncio.sleep(1)
        
        raise TimeoutError("服务器启动超时")
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🧪 开始运行所有测试")
        
        test_suites = [
            ("系统健康检查", self.test_system_health),
            ("MCP配置管理", self.test_mcp_management),
            ("LLM配置管理", self.test_llm_management),
            ("语音识别功能", self.test_speech_recognition),
            ("语音合成功能", self.test_speech_synthesis),
            ("文本对话功能", self.test_text_chat),
            ("语音对话功能", self.test_voice_chat),
            ("并发性能测试", self.test_concurrent_requests),
            ("错误处理测试", self.test_error_handling)
        ]
        
        total_tests = len(test_suites)
        passed_tests = 0
        
        for i, (test_name, test_func) in enumerate(test_suites, 1):
            logger.info(f"🔍 运行测试 [{i}/{total_tests}]: {test_name}")
            
            try:
                start_time = time.time()
                await test_func()
                duration = time.time() - start_time
                
                self.test_results.append({
                    "name": test_name,
                    "status": "PASSED",
                    "duration": duration,
                    "error": None
                })
                
                passed_tests += 1
                logger.info(f"✅ 测试通过: {test_name} ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                
                self.test_results.append({
                    "name": test_name,
                    "status": "FAILED",
                    "duration": duration,
                    "error": str(e)
                })
                
                logger.error(f"❌ 测试失败: {test_name} - {str(e)}")
        
        # 输出测试总结
        logger.info(f"📊 测试完成: {passed_tests}/{total_tests} 通过")
        
        return self.test_results
    
    async def test_system_health(self):
        """测试系统健康检查"""
        
        # 健康检查
        async with self.session.get(f"{self.server_url}/api/health") as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] in ["healthy", "degraded"]
        
        # 系统状态
        async with self.session.get(f"{self.server_url}/api/status") as response:
            assert response.status == 200
            data = await response.json()
            assert "uptime" in data
            assert "mcp_tools" in data
            assert "llm_models" in data
    
    async def test_mcp_management(self):
        """测试MCP配置管理"""
        
        # 获取MCP配置列表
        async with self.session.get(f"{self.server_url}/api/mcp/configs") as response:
            assert response.status == 200
            configs = await response.json()
            assert isinstance(configs, list)
        
        # 创建测试MCP配置
        test_config = {
            "name": "test_mcp_tool",
            "description": "测试MCP工具",
            "command": "python",
            "args": ["test_mcp_server.py"],
            "transport": "stdio"
        }
        
        async with self.session.post(
            f"{self.server_url}/api/mcp/configs",
            json=test_config
        ) as response:
            assert response.status == 200
            created_config = await response.json()
            config_id = created_config["id"]
        
        # 测试MCP配置
        async with self.session.post(
            f"{self.server_url}/api/mcp/configs/{config_id}/test"
        ) as response:
            assert response.status == 200
        
        # 更新MCP配置
        update_data = {"description": "更新后的测试MCP工具"}
        async with self.session.put(
            f"{self.server_url}/api/mcp/configs/{config_id}",
            json=update_data
        ) as response:
            assert response.status == 200
        
        # 删除MCP配置
        async with self.session.delete(
            f"{self.server_url}/api/mcp/configs/{config_id}"
        ) as response:
            assert response.status == 200
    
    async def test_llm_management(self):
        """测试LLM配置管理"""
        
        # 获取LLM配置列表
        async with self.session.get(f"{self.server_url}/api/llm/configs") as response:
            assert response.status == 200
            configs = await response.json()
            assert isinstance(configs, list)
        
        # 创建测试LLM配置
        test_config = {
            "name": "test_llm_model",
            "provider": "dashscope",
            "model_name": "qwen-test",
            "api_key": "test_api_key_1234567890",
            "temperature": 0.7
        }
        
        async with self.session.post(
            f"{self.server_url}/api/llm/configs",
            json=test_config
        ) as response:
            assert response.status == 200
            created_config = await response.json()
            config_id = created_config["id"]
        
        # 测试LLM配置
        async with self.session.post(
            f"{self.server_url}/api/llm/configs/{config_id}/test"
        ) as response:
            assert response.status == 200
        
        # 删除LLM配置
        async with self.session.delete(
            f"{self.server_url}/api/llm/configs/{config_id}"
        ) as response:
            assert response.status == 200
    
    async def test_speech_recognition(self):
        """测试语音识别功能"""
        
        # 创建测试音频文件
        test_audio_data = self.create_test_audio()
        
        # 准备multipart数据
        data = aiohttp.FormData()
        data.add_field(
            'audio_file',
            test_audio_data,
            filename='test.wav',
            content_type='audio/wav'
        )
        data.add_field('language', 'zh-CN')
        
        async with self.session.post(
            f"{self.server_url}/api/speech/recognize",
            data=data
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert "text" in result
            assert "confidence" in result
    
    async def test_speech_synthesis(self):
        """测试语音合成功能"""
        
        test_request = {
            "text": "这是一个语音合成测试",
            "language": "zh-CN",
            "speed": 1.0,
            "pitch": 1.0
        }
        
        async with self.session.post(
            f"{self.server_url}/api/speech/synthesize",
            json=test_request
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert "audio_data" in result
            assert "format" in result
    
    async def test_text_chat(self):
        """测试文本对话功能"""
        
        test_request = {
            "message": "你好，这是一个测试消息",
            "model_name": None,  # 使用默认模型
            "session_id": "test_session_123"
        }
        
        async with self.session.post(
            f"{self.server_url}/api/chat/text",
            json=test_request
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert "content" in result
            assert "model_name" in result
            assert "session_id" in result
    
    async def test_voice_chat(self):
        """测试语音对话功能"""
        
        # 创建测试音频文件
        test_audio_data = self.create_test_audio()
        
        # 准备multipart数据
        data = aiohttp.FormData()
        data.add_field(
            'audio_file',
            test_audio_data,
            filename='test.wav',
            content_type='audio/wav'
        )
        data.add_field('session_id', 'test_voice_session_123')
        
        async with self.session.post(
            f"{self.server_url}/api/chat/voice",
            data=data
        ) as response:
            assert response.status == 200
            result = await response.json()
            assert "user_text" in result
            assert "response_text" in result
            assert "response_audio" in result
    
    async def test_concurrent_requests(self):
        """测试并发请求性能"""
        
        # 并发健康检查
        tasks = []
        for i in range(10):
            task = self.session.get(f"{self.server_url}/api/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            assert response.status == 200
            response.close()
    
    async def test_error_handling(self):
        """测试错误处理"""
        
        # 测试不存在的端点
        async with self.session.get(f"{self.server_url}/api/nonexistent") as response:
            assert response.status == 404
            
        # 测试无效的MCP配置
        invalid_mcp_config = {
            "name": "",  # 无效的名称
            "command": "invalid_command"
        }
        
        async with self.session.post(
            f"{self.server_url}/api/mcp/configs",
            json=invalid_mcp_config
        ) as response:
            assert response.status == 400
        
        # 测试无效的LLM配置
        invalid_llm_config = {
            "name": "test",
            "provider": "invalid_provider",
            "model_name": "test"
        }
        
        async with self.session.post(
            f"{self.server_url}/api/llm/configs",
            json=invalid_llm_config
        ) as response:
            assert response.status == 400
    
    def create_test_audio(self) -> bytes:
        """创建测试音频数据"""
        # 创建一个简单的WAV文件头和静音数据
        import struct
        
        # WAV文件头
        sample_rate = 16000
        duration = 1  # 1秒
        channels = 1
        bits_per_sample = 16
        
        # 计算参数
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = sample_rate * duration * channels * bits_per_sample // 8
        
        # 构建WAV头
        wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk size
            1,   # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        
        # 创建静音数据
        audio_data = b'\x00' * data_size
        
        return wav_header + audio_data
    
    def generate_test_report(self) -> str:
        """生成测试报告"""
        report = {
            "timestamp": time.time(),
            "total_tests": len(self.test_results),
            "passed": len([t for t in self.test_results if t["status"] == "PASSED"]),
            "failed": len([t for t in self.test_results if t["status"] == "FAILED"]),
            "total_duration": sum(t["duration"] for t in self.test_results),
            "tests": self.test_results
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)

class PerformanceTestRunner:
    """性能测试运行器"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = None
    
    async def setup(self):
        """初始化性能测试环境"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """清理性能测试环境"""
        if self.session:
            await self.session.close()
    
    async def run_load_test(self, concurrent_users: int = 10, requests_per_user: int = 10):
        """运行负载测试"""
        logger.info(f"🚀 开始负载测试: {concurrent_users} 并发用户, 每用户 {requests_per_user} 请求")
        
        async def user_simulation(user_id: int):
            """模拟单个用户的请求"""
            user_results = []
            
            for i in range(requests_per_user):
                start_time = time.time()
                try:
                    async with self.session.get(f"{self.server_url}/api/health") as response:
                        duration = time.time() - start_time
                        user_results.append({
                            "user_id": user_id,
                            "request_id": i,
                            "status_code": response.status,
                            "duration": duration,
                            "success": response.status == 200
                        })
                except Exception as e:
                    duration = time.time() - start_time
                    user_results.append({
                        "user_id": user_id,
                        "request_id": i,
                        "status_code": None,
                        "duration": duration,
                        "success": False,
                        "error": str(e)
                    })
            
            return user_results
        
        # 启动并发用户
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        
        # 整理结果
        flat_results = [result for user_results in all_results for result in user_results]
        
        # 计算统计信息
        successful_requests = [r for r in flat_results if r["success"]]
        failed_requests = [r for r in flat_results if not r["success"]]
        
        stats = {
            "total_requests": len(flat_results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(flat_results) * 100,
            "avg_response_time": sum(r["duration"] for r in successful_requests) / len(successful_requests) if successful_requests else 0,
            "min_response_time": min(r["duration"] for r in successful_requests) if successful_requests else 0,
            "max_response_time": max(r["duration"] for r in successful_requests) if successful_requests else 0
        }
        
        logger.info(f"📊 负载测试完成: 成功率 {stats['success_rate']:.1f}%, 平均响应时间 {stats['avg_response_time']:.3f}s")
        
        return stats, flat_results

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent 后端测试运行器")
    parser.add_argument("--server", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--test-type", choices=["all", "unit", "integration", "performance"], 
                       default="all", help="测试类型")
    parser.add_argument("--output", help="测试报告输出文件")
    parser.add_argument("--concurrent-users", type=int, default=10, help="并发用户数（性能测试）")
    parser.add_argument("--requests-per-user", type=int, default=10, help="每用户请求数（性能测试）")
    
    args = parser.parse_args()
    
    if args.test_type in ["all", "unit", "integration"]:
        # 功能测试
        test_runner = TestRunner(args.server)
        
        try:
            await test_runner.setup()
            results = await test_runner.run_all_tests()
            
            # 生成报告
            report = test_runner.generate_test_report()
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"📄 测试报告已保存到: {args.output}")
            else:
                print(report)
        
        finally:
            await test_runner.cleanup()
    
    if args.test_type in ["all", "performance"]:
        # 性能测试
        perf_runner = PerformanceTestRunner(args.server)
        
        try:
            await perf_runner.setup()
            stats, results = await perf_runner.run_load_test(
                args.concurrent_users, 
                args.requests_per_user
            )
            
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        finally:
            await perf_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())