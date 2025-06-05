#!/usr/bin/env python3
"""
CosyVoice2-0.5B MCP服务器
基于WebSocket协议的远程MCP服务器，提供CosyVoice2语音合成功能
"""

import asyncio
import json
import logging
import argparse
import websockets
import sys
import os
import base64
import io
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CosyVoiceMCPServer")

# 全局变量存储CosyVoice实例
cosyvoice_instance = None

def initialize_cosyvoice(model_dir: str, load_jit: bool = False, load_trt: bool = False, fp16: bool = False):
    """初始化CosyVoice模型"""
    global cosyvoice_instance
    try:
        # 添加第三方库路径
        sys.path.append('third_party/Matcha-TTS')
        
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        
        cosyvoice_instance = {
            'model': CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16),
            'load_wav': load_wav,
            'torchaudio': torchaudio
        }
        logger.info(f"CosyVoice2模型已成功加载: {model_dir}")
        return True
    except Exception as e:
        logger.error(f"初始化CosyVoice模型失败: {str(e)}")
        return False

def audio_to_base64(audio_tensor, sample_rate: int) -> str:
    """将音频张量转换为base64编码的WAV数据"""
    try:
        import torchaudio
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
        buffer.seek(0)
        audio_data = buffer.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"音频转换失败: {str(e)}")
        raise

def base64_to_audio_file(base64_data: str, temp_dir: str = None) -> str:
    """将base64编码的音频数据保存为临时文件"""
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        audio_data = base64.b64decode(base64_data)
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=temp_dir, delete=False)
        temp_file.write(audio_data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"base64音频文件保存失败: {str(e)}")
        raise

def text_to_speech_zero_shot(text: str, prompt_text: str, prompt_audio_base64: str, stream: bool = False) -> Dict[str, Any]:
    """零样本语音合成"""
    if cosyvoice_instance is None:
        raise RuntimeError("CosyVoice模型未初始化")
    
    try:
        # 保存prompt音频到临时文件
        prompt_audio_path = base64_to_audio_file(prompt_audio_base64)
        
        # 加载prompt音频
        prompt_speech_16k = cosyvoice_instance['load_wav'](prompt_audio_path, 16000)
        
        # 进行语音合成
        results = []
        for i, result in enumerate(cosyvoice_instance['model'].inference_zero_shot(
            text, prompt_text, prompt_speech_16k, stream=stream
        )):
            audio_base64 = audio_to_base64(result['tts_speech'], cosyvoice_instance['model'].sample_rate)
            results.append({
                'index': i,
                'audio_base64': audio_base64,
                'sample_rate': cosyvoice_instance['model'].sample_rate
            })
        
        # 清理临时文件
        os.unlink(prompt_audio_path)
        
        return {
            'success': True,
            'results': results,
            'method': 'zero_shot'
        }
    except Exception as e:
        logger.error(f"零样本语音合成失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def text_to_speech_cross_lingual(text: str, prompt_audio_base64: str, stream: bool = False) -> Dict[str, Any]:
    """跨语言语音合成（支持细粒度控制）"""
    if cosyvoice_instance is None:
        raise RuntimeError("CosyVoice模型未初始化")
    
    try:
        # 保存prompt音频到临时文件
        prompt_audio_path = base64_to_audio_file(prompt_audio_base64)
        
        # 加载prompt音频
        prompt_speech_16k = cosyvoice_instance['load_wav'](prompt_audio_path, 16000)
        
        # 进行跨语言语音合成
        results = []
        for i, result in enumerate(cosyvoice_instance['model'].inference_cross_lingual(
            text, prompt_speech_16k, stream=stream
        )):
            audio_base64 = audio_to_base64(result['tts_speech'], cosyvoice_instance['model'].sample_rate)
            results.append({
                'index': i,
                'audio_base64': audio_base64,
                'sample_rate': cosyvoice_instance['model'].sample_rate
            })
        
        # 清理临时文件
        os.unlink(prompt_audio_path)
        
        return {
            'success': True,
            'results': results,
            'method': 'cross_lingual'
        }
    except Exception as e:
        logger.error(f"跨语言语音合成失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def text_to_speech_instruct(text: str, instruction: str, prompt_audio_base64: str, stream: bool = False) -> Dict[str, Any]:
    """指令式语音合成"""
    if cosyvoice_instance is None:
        raise RuntimeError("CosyVoice模型未初始化")
    
    try:
        # 保存prompt音频到临时文件
        prompt_audio_path = base64_to_audio_file(prompt_audio_base64)
        
        # 加载prompt音频
        prompt_speech_16k = cosyvoice_instance['load_wav'](prompt_audio_path, 16000)
        
        # 进行指令式语音合成
        results = []
        for i, result in enumerate(cosyvoice_instance['model'].inference_instruct2(
            text, instruction, prompt_speech_16k, stream=stream
        )):
            audio_base64 = audio_to_base64(result['tts_speech'], cosyvoice_instance['model'].sample_rate)
            results.append({
                'index': i,
                'audio_base64': audio_base64,
                'sample_rate': cosyvoice_instance['model'].sample_rate
            })
        
        # 清理临时文件
        os.unlink(prompt_audio_path)
        
        return {
            'success': True,
            'results': results,
            'method': 'instruct'
        }
    except Exception as e:
        logger.error(f"指令式语音合成失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def get_model_info() -> Dict[str, Any]:
    """获取模型信息"""
    if cosyvoice_instance is None:
        return {'error': 'CosyVoice模型未初始化'}
    
    try:
        return {
            'success': True,
            'sample_rate': cosyvoice_instance['model'].sample_rate,
            'model_type': 'CosyVoice2-0.5B',
            'available_methods': ['zero_shot', 'cross_lingual', 'instruct']
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_audio_file(audio_base64: str, filepath: str) -> Dict[str, Any]:
    """保存音频文件到指定路径"""
    try:
        audio_data = base64.b64decode(audio_base64)
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        return {
            'success': True,
            'message': f'音频文件已保存到: {filepath}'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def load_audio_file(filepath: str) -> Dict[str, Any]:
    """加载音频文件为base64格式"""
    try:
        if not os.path.exists(filepath):
            return {'success': False, 'error': f'文件不存在: {filepath}'}
        
        with open(filepath, 'rb') as f:
            audio_data = f.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return {
            'success': True,
            'audio_base64': audio_base64,
            'filepath': filepath
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# MCP工具定义
TOOLS = {
    "initialize_model": {
        "function": initialize_cosyvoice,
        "description": "初始化CosyVoice2-0.5B模型",
        "parameters": {
            "model_dir": {"type": "string", "description": "模型目录路径，如：pretrained_models/CosyVoice2-0.5B"},
            "load_jit": {"type": "boolean", "description": "是否加载JIT模式（默认false）", "default": False},
            "load_trt": {"type": "boolean", "description": "是否加载TensorRT模式（默认false）", "default": False},
            "fp16": {"type": "boolean", "description": "是否使用FP16精度（默认false）", "default": False}
        }
    },
    
    "zero_shot_tts": {
        "function": text_to_speech_zero_shot,
        "description": "零样本语音合成 - 根据参考音频生成指定文本的语音",
        "parameters": {
            "text": {"type": "string", "description": "要合成的文本"},
            "prompt_text": {"type": "string", "description": "参考音频对应的文本"},
            "prompt_audio_base64": {"type": "string", "description": "参考音频的base64编码（WAV格式）"},
            "stream": {"type": "boolean", "description": "是否使用流式生成（默认false）", "default": False}
        }
    },
    
    "cross_lingual_tts": {
        "function": text_to_speech_cross_lingual,
        "description": "跨语言语音合成 - 支持细粒度控制，如[laughter]等标记",
        "parameters": {
            "text": {"type": "string", "description": "要合成的文本，支持细粒度控制标记如[laughter]"},
            "prompt_audio_base64": {"type": "string", "description": "参考音频的base64编码（WAV格式）"},
            "stream": {"type": "boolean", "description": "是否使用流式生成（默认false）", "default": False}
        }
    },
    
    "instruct_tts": {
        "function": text_to_speech_instruct,
        "description": "指令式语音合成 - 根据指令控制语音风格",
        "parameters": {
            "text": {"type": "string", "description": "要合成的文本"},
            "instruction": {"type": "string", "description": "语音风格指令，如：用四川话说这句话"},
            "prompt_audio_base64": {"type": "string", "description": "参考音频的base64编码（WAV格式）"},
            "stream": {"type": "boolean", "description": "是否使用流式生成（默认false）", "default": False}
        }
    },
    
    "get_model_info": {
        "function": get_model_info,
        "description": "获取当前加载的模型信息",
        "parameters": {}
    },
    
    "save_audio": {
        "function": save_audio_file,
        "description": "保存base64编码的音频文件到指定路径",
        "parameters": {
            "audio_base64": {"type": "string", "description": "base64编码的音频数据"},
            "filepath": {"type": "string", "description": "保存的文件路径"}
        }
    },
    
    "load_audio": {
        "function": load_audio_file,
        "description": "加载音频文件并返回base64编码",
        "parameters": {
            "filepath": {"type": "string", "description": "音频文件路径"}
        }
    }
}

def handle_mcp_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """处理MCP协议请求"""
    try:
        action = request_data.get("action")
        
        # 列出可用函数
        if action == "list_functions":
            functions = []
            for name, info in TOOLS.items():
                function_def = {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"]
                }
                functions.append(function_def)
            return {"functions": functions}
        
        # 调用函数
        elif action == "call_function":
            function_name = request_data.get("name")
            arguments = request_data.get("arguments", {})
            
            if function_name not in TOOLS:
                return {"error": f"Function {function_name} not found"}
            
            # 处理默认参数
            parameters = TOOLS[function_name]["parameters"]
            for param_name, param_info in parameters.items():
                if param_name not in arguments and "default" in param_info:
                    arguments[param_name] = param_info["default"]
            
            # 调用函数
            function = TOOLS[function_name]["function"]
            result = function(**arguments)
            
            # 返回结果
            return {"result": result}
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.exception("Error handling MCP request")
        return {"error": str(e)}

async def handle_websocket(websocket, path):
    """处理WebSocket连接"""
    client_ip = websocket.remote_address[0]
    logger.info(f"客户端连接: {client_ip}")
    
    try:
        async for message in websocket:
            try:
                # 解析JSON消息
                request_data = json.loads(message)
                logger.info(f"收到请求: {request_data.get('action', 'unknown')} - {request_data.get('name', '')}")
                
                # 处理MCP请求
                response = handle_mcp_request(request_data)
                
                # 发送响应
                await websocket.send(json.dumps(response))
            
            except json.JSONDecodeError:
                logger.error(f"无效的JSON: {message}")
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            
            except Exception as e:
                logger.exception("处理消息时出错")
                await websocket.send(json.dumps({"error": str(e)}))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"客户端断开连接: {client_ip}")
    
    except Exception as e:
        logger.exception(f"WebSocket处理异常: {e}")

async def start_server(host: str, port: int):
    """启动WebSocket服务器"""
    server = await websockets.serve(handle_websocket, host, port)
    logger.info(f"CosyVoice MCP服务器已启动: ws://{host}:{port}")
    logger.info("可用工具:")
    for tool_name, tool_info in TOOLS.items():
        logger.info(f"  - {tool_name}: {tool_info['description']}")
    await server.wait_closed()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CosyVoice2-0.5B MCP服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听的主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="监听的端口 (默认: 8080)")
    parser.add_argument("--model-dir", help="CosyVoice模型目录路径（可选，也可通过MCP调用初始化）")
    args = parser.parse_args()
    
    # 如果提供了模型目录，则预先初始化模型
    if args.model_dir:
        logger.info(f"预初始化模型: {args.model_dir}")
        if initialize_cosyvoice(args.model_dir):
            logger.info("模型初始化成功")
        else:
            logger.error("模型初始化失败，服务器仍将启动，可稍后通过MCP调用初始化")
    
    # 启动服务器
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.exception(f"服务器错误: {e}")

if __name__ == "__main__":
    main()