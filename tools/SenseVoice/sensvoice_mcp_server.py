#!/usr/bin/env python3
"""
SensVoice/FunASR MCP服务器
基于WebSocket协议的远程MCP服务器，提供SensVoice语音识别功能
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
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SensVoiceMCPServer")

# 全局变量存储模型实例
model_instances = {}

def initialize_model(model_name: str, 
                    device: str = "cpu", 
                    ncpu: int = 4, 
                    batch_size: int = 1,
                    hub: str = "ms",
                    vad_model: str = None,
                    punc_model: str = None,
                    spk_model: str = None,
                    **kwargs) -> Dict[str, Any]:
    """初始化SensVoice/FunASR模型"""
    global model_instances
    try:
        from funasr import AutoModel
        
        # 构建模型参数
        model_kwargs = {
            "model": model_name,
            "device": device,
            "ncpu": ncpu,
            "batch_size": batch_size,
            "hub": hub
        }
        
        # 添加VAD模型
        if vad_model:
            model_kwargs["vad_model"] = vad_model
            
        # 添加标点模型
        if punc_model:
            model_kwargs["punc_model"] = punc_model
            
        # 添加说话人识别模型
        if spk_model:
            model_kwargs["spk_model"] = spk_model
            
        # 添加其他参数
        model_kwargs.update(kwargs)
        
        # 创建模型实例
        model = AutoModel(**model_kwargs)
        
        # 存储模型实例
        model_key = f"{model_name}_{device}"
        model_instances[model_key] = {
            'model': model,
            'config': model_kwargs
        }
        
        logger.info(f"SensVoice模型已成功加载: {model_name} on {device}")
        return {
            'success': True, 
            'model_key': model_key,
            'model_name': model_name,
            'device': device,
            'config': model_kwargs
        }
    except Exception as e:
        logger.error(f"初始化SensVoice模型失败: {str(e)}")
        logger.exception("详细错误信息:")
        return {'success': False, 'error': str(e)}

def list_models() -> Dict[str, Any]:
    """列出已加载的模型"""
    try:
        models = []
        for model_key, model_info in model_instances.items():
            models.append({
                'model_key': model_key,
                'config': model_info['config']
            })
        return {
            'success': True,
            'models': models,
            'count': len(models)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def base64_to_audio_file(base64_data: str, temp_dir: str = None, suffix: str = '.wav') -> str:
    """将base64编码的音频数据保存为临时文件"""
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        audio_data = base64.b64decode(base64_data)
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, dir=temp_dir, delete=False)
        temp_file.write(audio_data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"base64音频文件保存失败: {str(e)}")
        raise

def audio_recognize(model_key: str, 
                   audio_input: Union[str, bytes], 
                   input_type: str = "base64",
                   output_dir: str = None,
                   batch_size_s: int = 300,
                   batch_size_threshold_s: int = 60,
                   hotword: str = None,
                   **kwargs) -> Dict[str, Any]:
    """语音识别"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 处理输入
        if input_type == "base64":
            # base64编码的音频数据
            audio_file = base64_to_audio_file(audio_input)
            input_data = audio_file
        elif input_type == "file_path":
            # 文件路径
            input_data = audio_input
        elif input_type == "bytes":
            # 音频字节流
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(audio_input)
            temp_file.close()
            input_data = temp_file.name
        elif input_type == "wav_scp":
            # wav.scp格式
            input_data = audio_input
        else:
            return {'success': False, 'error': f'不支持的输入类型: {input_type}'}
        
        # 构建推理参数
        generate_kwargs = {}
        if output_dir:
            generate_kwargs['output_dir'] = output_dir
        if batch_size_s:
            generate_kwargs['batch_size_s'] = batch_size_s
        if batch_size_threshold_s:
            generate_kwargs['batch_size_threshold_s'] = batch_size_threshold_s
        if hotword:
            generate_kwargs['hotword'] = hotword
        
        # 添加其他参数
        generate_kwargs.update(kwargs)
        
        # 执行识别
        results = model.generate(input=input_data, **generate_kwargs)
        
        # 清理临时文件
        if input_type in ["base64", "bytes"]:
            try:
                os.unlink(input_data)
            except:
                pass
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key,
            'input_type': input_type
        }
    except Exception as e:
        logger.error(f"语音识别失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def streaming_asr(model_key: str,
                 audio_chunk_base64: str,
                 cache: Dict = None,
                 is_final: bool = False,
                 chunk_size: List[int] = [0, 10, 5],
                 encoder_chunk_look_back: int = 4,
                 decoder_chunk_look_back: int = 1) -> Dict[str, Any]:
    """实时语音识别"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 解码音频数据
        audio_data = base64.b64decode(audio_chunk_base64)
        
        # 转换为numpy数组（假设是16位PCM，16kHz）
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 执行流式识别
        results = model.generate(
            input=audio_np,
            cache=cache or {},
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back
        )
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key,
            'is_final': is_final,
            'cache': cache
        }
    except Exception as e:
        logger.error(f"实时语音识别失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def vad_detect(model_key: str,
               audio_input: Union[str, bytes],
               input_type: str = "base64",
               **kwargs) -> Dict[str, Any]:
    """语音端点检测"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 处理输入
        if input_type == "base64":
            audio_file = base64_to_audio_file(audio_input)
            input_data = audio_file
        elif input_type == "file_path":
            input_data = audio_input
        else:
            return {'success': False, 'error': f'不支持的输入类型: {input_type}'}
        
        # 执行VAD检测
        results = model.generate(input=input_data, **kwargs)
        
        # 清理临时文件
        if input_type == "base64":
            try:
                os.unlink(input_data)
            except:
                pass
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key,
            'segments': results[0]['value'] if results and 'value' in results[0] else []
        }
    except Exception as e:
        logger.error(f"VAD检测失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def streaming_vad(model_key: str,
                 audio_chunk_base64: str,
                 cache: Dict = None,
                 is_final: bool = False,
                 chunk_size: int = 200) -> Dict[str, Any]:
    """实时语音端点检测"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 解码音频数据
        audio_data = base64.b64decode(audio_chunk_base64)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 执行流式VAD
        results = model.generate(
            input=audio_np,
            cache=cache or {},
            is_final=is_final,
            chunk_size=chunk_size
        )
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key,
            'is_final': is_final,
            'cache': cache,
            'segments': results[0]['value'] if results and 'value' in results[0] else []
        }
    except Exception as e:
        logger.error(f"实时VAD检测失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def punctuation_restore(model_key: str, text: str) -> Dict[str, Any]:
    """标点恢复"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 执行标点恢复
        results = model.generate(input=text)
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key,
            'original_text': text,
            'restored_text': results[0]['value'] if results and 'value' in results[0] else text
        }
    except Exception as e:
        logger.error(f"标点恢复失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def timestamp_prediction(model_key: str,
                        audio_file_base64: str,
                        text_content: str) -> Dict[str, Any]:
    """时间戳预测"""
    if model_key not in model_instances:
        return {'success': False, 'error': f'模型未找到: {model_key}'}
    
    try:
        model = model_instances[model_key]['model']
        
        # 保存音频文件
        audio_file = base64_to_audio_file(audio_file_base64)
        
        # 保存文本文件
        text_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        text_file.write(text_content)
        text_file.close()
        
        # 执行时间戳预测
        results = model.generate(input=(audio_file, text_file.name), data_type=("sound", "text"))
        
        # 清理临时文件
        os.unlink(audio_file)
        os.unlink(text_file.name)
        
        return {
            'success': True,
            'results': results,
            'model_key': model_key
        }
    except Exception as e:
        logger.error(f"时间戳预测失败: {str(e)}")
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
        "function": initialize_model,
        "description": "初始化SensVoice/FunASR模型",
        "parameters": {
            "model_name": {"type": "string", "description": "模型名称，如：paraformer-zh、fsmn-vad、ct-punc等"},
            "device": {"type": "string", "description": "设备类型：cpu、cuda:0、mps、xpu等", "default": "cpu"},
            "ncpu": {"type": "integer", "description": "CPU线程数", "default": 4},
            "batch_size": {"type": "integer", "description": "批处理大小", "default": 1},
            "hub": {"type": "string", "description": "模型下载源：ms(modelscope)或hf(huggingface)", "default": "ms"},
            "vad_model": {"type": "string", "description": "VAD模型名称，可选", "default": None},
            "punc_model": {"type": "string", "description": "标点模型名称，可选", "default": None},
            "spk_model": {"type": "string", "description": "说话人识别模型名称，可选", "default": None}
        }
    },
    
    "list_models": {
        "function": list_models,
        "description": "列出已加载的模型",
        "parameters": {}
    },
    
    "audio_recognize": {
        "function": audio_recognize,
        "description": "语音识别（非实时）",
        "parameters": {
            "model_key": {"type": "string", "description": "模型键值"},
            "audio_input": {"type": "string", "description": "音频输入（base64编码或文件路径）"},
            "input_type": {"type": "string", "description": "输入类型：base64、file_path、bytes、wav_scp", "default": "base64"},
            "output_dir": {"type": "string", "description": "输出目录（可选）", "default": None},
            "batch_size_s": {"type": "integer", "description": "动态batch总音频时长（秒）", "default": 300},
            "batch_size_threshold_s": {"type": "integer", "description": "batch阈值（秒）", "default": 60},
            "hotword": {"type": "string", "description": "热词（可选）", "default": None}
        }
    },
    
    "streaming_asr": {
        "function": streaming_asr,
        "description": "实时语音识别",
        "parameters": {
            "model_key": {"type": "string", "description": "模型键值"},
            "audio_chunk_base64": {"type": "string", "description": "音频片段的base64编码"},
            "cache": {"type": "object", "description": "缓存对象", "default": None},
            "is_final": {"type": "boolean", "description": "是否为最后一个片段", "default": False},
            "chunk_size": {"type": "array", "description": "流式延时配置 [0,10,5]", "default": [0, 10, 5]},
            "encoder_chunk_look_back": {"type": "integer", "description": "编码器回看chunks数", "default": 4},
            "decoder_chunk_look_back": {"type": "integer", "description": "解码器回看chunks数", "default": 1}
        }
    },
    
    "vad_detect": {
        "function": vad_detect,
        "description": "语音端点检测（非实时）",
        "parameters": {
            "model_key": {"type": "string", "description": "VAD模型键值"},
            "audio_input": {"type": "string", "description": "音频输入（base64编码或文件路径）"},
            "input_type": {"type": "string", "description": "输入类型：base64、file_path", "default": "base64"}
        }
    },
    
    "streaming_vad": {
        "function": streaming_vad,
        "description": "实时语音端点检测",
        "parameters": {
            "model_key": {"type": "string", "description": "VAD模型键值"},
            "audio_chunk_base64": {"type": "string", "description": "音频片段的base64编码"},
            "cache": {"type": "object", "description": "缓存对象", "default": None},
            "is_final": {"type": "boolean", "description": "是否为最后一个片段", "default": False},
            "chunk_size": {"type": "integer", "description": "chunk大小（毫秒）", "default": 200}
        }
    },
    
    "punctuation_restore": {
        "function": punctuation_restore,
        "description": "标点恢复",
        "parameters": {
            "model_key": {"type": "string", "description": "标点模型键值"},
            "text": {"type": "string", "description": "需要恢复标点的文本"}
        }
    },
    
    "timestamp_prediction": {
        "function": timestamp_prediction,
        "description": "时间戳预测",
        "parameters": {
            "model_key": {"type": "string", "description": "时间戳模型键值"},
            "audio_file_base64": {"type": "string", "description": "音频文件的base64编码"},
            "text_content": {"type": "string", "description": "对应的文本内容"}
        }
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
    logger.info(f"SensVoice MCP服务器已启动: ws://{host}:{port}")
    logger.info("可用工具:")
    for tool_name, tool_info in TOOLS.items():
        logger.info(f"  - {tool_name}: {tool_info['description']}")
    await server.wait_closed()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SensVoice/FunASR MCP服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听的主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="监听的端口 (默认: 8081)")
    parser.add_argument("--auto-init", action="store_true", help="自动初始化常用模型")
    args = parser.parse_args()
    
    # 如果启用自动初始化
    if args.auto_init:
        logger.info("自动初始化常用模型...")
        try:
            # 初始化ASR模型
            asr_result = initialize_model("paraformer-zh", device="cpu", vad_model="fsmn-vad", punc_model="ct-punc")
            if asr_result['success']:
                logger.info(f"ASR模型初始化成功: {asr_result['model_key']}")
            
            # 初始化VAD模型
            vad_result = initialize_model("fsmn-vad", device="cpu")
            if vad_result['success']:
                logger.info(f"VAD模型初始化成功: {vad_result['model_key']}")
                
            # 初始化标点模型
            punc_result = initialize_model("ct-punc", device="cpu")
            if punc_result['success']:
                logger.info(f"标点模型初始化成功: {punc_result['model_key']}")
                
        except Exception as e:
            logger.error(f"自动初始化模型失败: {e}")
    
    # 启动服务器
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.exception(f"服务器错误: {e}")

if __name__ == "__main__":
    main()