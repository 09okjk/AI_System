#!/usr/bin/env python3
"""
CosyVoice MCP客户端示例
演示如何与CosyVoice MCP服务器进行交互
"""

import asyncio
import json
import websockets
import base64
import argparse
from pathlib import Path

class CosyVoiceMCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
    
    async def connect(self):
        """连接到MCP服务器"""
        self.websocket = await websockets.connect(self.server_url)
        print(f"已连接到CosyVoice MCP服务器: {self.server_url}")
    
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
    
    async def initialize_model(self, model_dir: str):
        """初始化模型"""
        arguments = {"model_dir": model_dir}
        response = await self.send_request("call_function", "initialize_model", arguments)
        return response
    
    async def get_model_info(self):
        """获取模型信息"""
        response = await self.send_request("call_function", "get_model_info")
        return response
    
    async def load_audio_file(self, filepath: str):
        """加载音频文件"""
        arguments = {"filepath": filepath}
        response = await self.send_request("call_function", "load_audio", arguments)
        return response
    
    async def zero_shot_tts(self, text: str, prompt_text: str, prompt_audio_base64: str):
        """零样本语音合成"""
        arguments = {
            "text": text,
            "prompt_text": prompt_text,
            "prompt_audio_base64": prompt_audio_base64
        }
        response = await self.send_request("call_function", "zero_shot_tts", arguments)
        return response
    
    async def cross_lingual_tts(self, text: str, prompt_audio_base64: str):
        """跨语言语音合成"""
        arguments = {
            "text": text,
            "prompt_audio_base64": prompt_audio_base64
        }
        response = await self.send_request("call_function", "cross_lingual_tts", arguments)
        return response
    
    async def instruct_tts(self, text: str, instruction: str, prompt_audio_base64: str):
        """指令式语音合成"""
        arguments = {
            "text": text,
            "instruction": instruction,
            "prompt_audio_base64": prompt_audio_base64
        }
        response = await self.send_request("call_function", "instruct_tts", arguments)
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
    client = CosyVoiceMCPClient("ws://localhost:8082")
    await client.connect()
    
    try:
        # 1. 列出可用函数
        print("=== 可用函数 ===")
        functions = await client.list_functions()
        for func in functions:
            print(f"- {func['name']}: {func['description']}")
        
        # 2. 初始化模型（如果服务器启动时未初始化）
        print("\n=== 初始化模型 ===")
        model_response = await client.initialize_model("pretrained_models/CosyVoice2-0.5B")
        print(f"初始化结果: {model_response}")
        
        # 3. 获取模型信息
        print("\n=== 模型信息 ===")
        info_response = await client.get_model_info()
        print(f"模型信息: {info_response}")
        
        # 4. 加载参考音频（需要提供实际的音频文件路径）
        prompt_audio_path = "zero_shot_prompt.wav"  # 替换为实际路径
        if Path(prompt_audio_path).exists():
            print(f"\n=== 加载参考音频: {prompt_audio_path} ===")
            audio_response = await client.load_audio_file(prompt_audio_path)
            if audio_response.get("success"):
                prompt_audio_base64 = audio_response["audio_base64"]
                
                # 5. 零样本语音合成
                print("\n=== 零样本语音合成 ===")
                tts_response = await client.zero_shot_tts(
                    text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
                    prompt_text="希望你以后能够做的比我还好呦。",
                    prompt_audio_base64=prompt_audio_base64
                )
                
                if tts_response.get("result", {}).get("success"):
                    results = tts_response["result"]["results"]
                    for i, result in enumerate(results):
                        # 保存生成的音频
                        output_path = f"output_zero_shot_{i}.wav"
                        save_response = await client.save_audio(result["audio_base64"], output_path)
                        print(f"音频 {i} 保存结果: {save_response}")
                
                # 6. 指令式语音合成
                print("\n=== 指令式语音合成 ===")
                instruct_response = await client.instruct_tts(
                    text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
                    instruction="用四川话说这句话",
                    prompt_audio_base64=prompt_audio_base64
                )
                
                if instruct_response.get("result", {}).get("success"):
                    results = instruct_response["result"]["results"]
                    for i, result in enumerate(results):
                        output_path = f"output_instruct_{i}.wav"
                        save_response = await client.save_audio(result["audio_base64"], output_path)
                        print(f"指令式音频 {i} 保存结果: {save_response}")
            else:
                print(f"加载音频失败: {audio_response.get('error')}")
        else:
            print(f"参考音频文件不存在: {prompt_audio_path}")
            print("请提供有效的WAV音频文件路径来测试语音合成功能")
    
    finally:
        await client.disconnect()

async def interactive_mode():
    """交互模式"""
    client = CosyVoiceMCPClient("ws://localhost:8082")
    await client.connect()
    
    try:
        print("CosyVoice MCP客户端 - 交互模式")
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("可用命令:")
                print("  list - 列出可用函数")
                print("  init <model_dir> - 初始化模型")
                print("  info - 获取模型信息")
                print("  load <audio_path> - 加载音频文件")
                print("  tts <text> <prompt_text> <audio_path> - 零样本语音合成")
                print("  quit - 退出")
            elif command == "list":
                functions = await client.list_functions()
                for func in functions:
                    print(f"- {func['name']}: {func['description']}")
            elif command == "info":
                response = await client.get_model_info()
                print(f"模型信息: {response}")
            elif command.startswith("init "):
                model_dir = command[5:].strip()
                response = await client.initialize_model(model_dir)
                print(f"初始化结果: {response}")
            else:
                print("未知命令，输入 'help' 查看可用命令")
    
    finally:
        await client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="CosyVoice MCP客户端")
    parser.add_argument("--server", default="ws://localhost:8082", help="MCP服务器地址")
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