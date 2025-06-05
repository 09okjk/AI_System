#!/usr/bin/env python3
"""
SensVoice MCP客户端示例
演示如何与SensVoice MCP服务器进行交互
"""

import asyncio
import json
import websockets
import base64
import argparse
from pathlib import Path

class SensVoiceMCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
    
    async def connect(self):
        """连接到MCP服务器"""
        self.websocket = await websockets.connect(self.server_url)
        print(f"已连接到SensVoice MCP服务器: {self.server_url}")
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            print("已断开连接")
    
    async def send_request(self, action: str, name: str = None, arguments: dict = None):
        """发送MCP请求"""
        request = {"action": action}
        if name:
            request["name"] = name
        if arguments:
            request["arguments"] = arguments
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def list_functions(self):
        """列出可用函数"""
        response = await self.send_request("list_functions")
        return response.get("functions", [])
    
    async def initialize_model(self, model_name: str, device: str = "cpu", **kwargs):
        """初始化模型"""
        arguments = {"model_name": model_name, "device": device}
        arguments.update(kwargs)
        response = await self.send_request("call_function", "initialize_model", arguments)
        return response
    
    async def list_models(self):
        """列出已加载的模型"""
        response = await self.send_request("call_function", "list_models")
        return response
    
    async def load_audio_file(self, filepath: str):
        """加载音频文件"""
        arguments = {"filepath": filepath}
        response = await self.send_request("call_function", "load_audio", arguments)
        return response
    
    async def audio_recognize(self, model_key: str, audio_base64: str, **kwargs):
        """语音识别"""
        arguments = {
            "model_key": model_key,
            "audio_input": audio_base64,
            "input_type": "base64"
        }
        arguments.update(kwargs)
        response = await self.send_request("call_function", "audio_recognize", arguments)
        return response
    
    async def vad_detect(self, model_key: str, audio_base64: str):
        """语音端点检测"""
        arguments = {
            "model_key": model_key,
            "audio_input": audio_base64,
            "input_type": "base64"
        }
        response = await self.send_request("call_function", "vad_detect", arguments)
        return response
    
    async def punctuation_restore(self, model_key: str, text: str):
        """标点恢复"""
        arguments = {
            "model_key": model_key,
            "text": text
        }
        response = await self.send_request("call_function", "punctuation_restore", arguments)
        return response
    
    async def save_audio(self, audio_base64: str, filepath: str):
        """保存音频文件"""
        arguments = {
            "audio_base64": audio_base64,
            "filepath": filepath
        }
        response = await self.send_request("call_function", "save_audio", arguments)
        return response

async def demo_usage():
    """演示使用方法"""
    # 连接到服务器
    client = SensVoiceMCPClient("ws://localhost:8081")
    await client.connect()
    
    try:
        # 1. 列出可用函数
        print("=== 可用函数 ===")
        functions = await client.list_functions()
        for func in functions:
            print(f"- {func['name']}: {func['description']}")
        
        # 2. 初始化ASR模型（包含VAD和标点）
        print("\n=== 初始化ASR模型 ===")
        asr_response = await client.initialize_model(
            "paraformer-zh", 
            device="cpu",
            vad_model="fsmn-vad",
            punc_model="ct-punc"
        )
        print(f"ASR模型初始化结果: {asr_response}")
        
        if asr_response.get("result", {}).get("success"):
            asr_model_key = asr_response["result"]["model_key"]
            
            # 3. 初始化VAD模型
            print("\n=== 初始化VAD模型 ===")
            vad_response = await client.initialize_model("fsmn-vad", device="cpu")
            print(f"VAD模型初始化结果: {vad_response}")
            
            # 4. 初始化标点模型
            print("\n=== 初始化标点模型 ===")
            punc_response = await client.initialize_model("ct-punc", device="cpu")
            print(f"标点模型初始化结果: {punc_response}")
            
            # 5. 列出已加载的模型
            print("\n=== 已加载的模型 ===")
            models_response = await client.list_models()
            print(f"已加载模型: {models_response}")
            
            # 6. 测试音频文件（需要提供实际的音频文件路径）
            audio_file_path = "test_audio.wav"  # 替换为实际路径
            if Path(audio_file_path).exists():
                print(f"\n=== 加载测试音频: {audio_file_path} ===")
                audio_response = await client.load_audio_file(audio_file_path)
                
                if audio_response.get("result", {}).get("success"):
                    audio_base64 = audio_response["result"]["audio_base64"]
                    
                    # 7. 语音识别
                    print("\n=== 语音识别 ===")
                    asr_result = await client.audio_recognize(
                        asr_model_key,
                        audio_base64,
                        hotword="SensVoice"
                    )
                    print(f"识别结果: {asr_result}")
                    
                    # 8. VAD检测
                    if vad_response.get("result", {}).get("success"):
                        vad_model_key = vad_response["result"]["model_key"]
                        print("\n=== VAD检测 ===")
                        vad_result = await client.vad_detect(vad_model_key, audio_base64)
                        print(f"VAD结果: {vad_result}")
                    
                    # 9. 标点恢复
                    if punc_response.get("result", {}).get("success"):
                        punc_model_key = punc_response["result"]["model_key"]
                        test_text = "那今天的会就到这里吧 happy new year 明年见"
                        print(f"\n=== 标点恢复 ===")
                        print(f"原始文本: {test_text}")
                        punc_result = await client.punctuation_restore(punc_model_key, test_text)
                        print(f"标点恢复结果: {punc_result}")
                else:
                    print(f"加载音频失败: {audio_response.get('result', {}).get('error')}")
            else:
                print(f"测试音频文件不存在: {audio_file_path}")
                print("请提供有效的WAV音频文件路径来测试语音识别功能")
        else:
            print(f"模型初始化失败: {asr_response}")
    
    finally:
        await client.disconnect()

async def interactive_mode():
    """交互模式"""
    client = SensVoiceMCPClient("ws://localhost:8081")
    await client.connect()
    
    try:
        print("SensVoice MCP客户端 - 交互模式")
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("可用命令:")
                print("  list - 列出可用函数")
                print("  models - 列出已加载的模型")
                print("  init <model_name> - 初始化模型")
                print("  asr <model_key> <audio_path> - 语音识别")
                print("  vad <model_key> <audio_path> - VAD检测")
                print("  punc <model_key> <text> - 标点恢复")
                print("  quit - 退出")
            elif command == "list":
                functions = await client.list_functions()
                for func in functions:
                    print(f"- {func['name']}: {func['description']}")
            elif command == "models":
                response = await client.list_models()
                print(f"已加载模型: {response}")
            elif command.startswith("init "):
                model_name = command[5:].strip()
                response = await client.initialize_model(model_name)
                print(f"初始化结果: {response}")
            elif command.startswith("punc "):
                parts = command[5:].split(" ", 1)
                if len(parts) == 2:
                    model_key, text = parts
                    response = await client.punctuation_restore(model_key, text)
                    print(f"标点恢复结果: {response}")
                else:
                    print("用法: punc <model_key> <text>")
            else:
                print("未知命令，输入 'help' 查看可用命令")
    
    finally:
        await client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="SensVoice MCP客户端")
    parser.add_argument("--server", default="ws://localhost:8081", help="MCP服务器地址")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_usage())
    elif args.interactive:
        asyncio.run(interactive_mode())
    else:
        print("请指定 --demo 或 --interactive 模式")
        print("使用 --help 查看更多选项")

if __name__ == "__main__":
    main()