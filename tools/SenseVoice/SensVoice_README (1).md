# SensVoice/FunASR MCP服务器

基于Model Context Protocol (MCP) 的SensVoice/FunASR语音识别服务器，提供全面的语音处理功能。

## 功能特性

- **语音识别** (ASR): 支持中文、英文等多语言语音识别
- **语音端点检测** (VAD): 检测语音的起始和结束时间点
- **标点恢复**: 为识别文本添加标点符号
- **实时处理**: 支持流式语音识别和VAD检测
- **时间戳预测**: 预测文本与音频的对齐时间戳
- **多模型支持**: 可同时加载和使用多个模型
- **WebSocket协议**: 支持远程访问和实时通信

## 安装要求

### 1. 安装FunASR

```bash
# 安装FunASR
pip install funasr
# 或从源码安装
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR
pip install -e .
```

### 2. 安装MCP服务器依赖

```bash
pip install websockets asyncio numpy
```

## 使用方法

### 1. 启动MCP服务器

```bash
# 基本启动
python sensvoice_mcp_server.py

# 指定端口
python sensvoice_mcp_server.py --port 8081

# 自动初始化常用模型
python sensvoice_mcp_server.py --auto-init
```

### 2. 使用客户端

```bash
# 运行演示
python sensvoice_mcp_client.py --demo

# 交互模式
python sensvoice_mcp_client.py --interactive
```

## MCP工具说明

### initialize_model
初始化SensVoice/FunASR模型
- `model_name`: 模型名称 (如: paraformer-zh, fsmn-vad, ct-punc)
- `device`: 设备类型 (cpu, cuda:0, mps, xpu)
- `vad_model`: VAD模型名称 (可选)
- `punc_model`: 标点模型名称 (可选)
- `spk_model`: 说话人识别模型名称 (可选)

### audio_recognize
语音识别（非实时）
- `model_key`: 模型键值
- `audio_input`: 音频输入（base64编码或文件路径）
- `input_type`: 输入类型 (base64, file_path, bytes, wav_scp)
- `hotword`: 热词（可选）

### streaming_asr
实时语音识别
- `model_key`: 模型键值
- `audio_chunk_base64`: 音频片段的base64编码
- `cache`: 缓存对象
- `is_final`: 是否为最后一个片段
- `chunk_size`: 流式延时配置

### vad_detect
语音端点检测（非实时）
- `model_key`: VAD模型键值
- `audio_input`: 音频输入（base64编码或文件路径）

### streaming_vad
实时语音端点检测
- `model_key`: VAD模型键值
- `audio_chunk_base64`: 音频片段的base64编码
- `chunk_size`: chunk大小（毫秒）

### punctuation_restore
标点恢复
- `model_key`: 标点模型键值
- `text`: 需要恢复标点的文本

### timestamp_prediction
时间戳预测
- `model_key`: 时间戳模型键值
- `audio_file_base64`: 音频文件的base64编码
- `text_content`: 对应的文本内容

## 支持的模型

### ASR模型
- `paraformer-zh`: 中文语音识别
- `paraformer-en`: 英文语音识别
- `paraformer-zh-streaming`: 中文实时语音识别

### VAD模型
- `fsmn-vad`: 语音端点检测

### 标点模型
- `ct-punc`: 中文标点恢复

### 时间戳模型
- `fa-zh`: 中文时间戳预测

## API使用示例

### Python客户端示例

```python
import asyncio
import json
import websockets

async def example_usage():
    # 连接到服务器
    websocket = await websockets.connect("ws://localhost:8081")
    
    # 初始化ASR模型（带VAD和标点）
    request = {
        "action": "call_function",
        "name": "initialize_model",
        "arguments": {
            "model_name": "paraformer-zh",
            "device": "cpu",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc"
        }
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    result = json.loads(response)
    model_key = result["result"]["model_key"]
    
    # 加载音频文件
    request = {
        "action": "call_function",
        "name": "load_audio",
        "arguments": {"filepath": "test_audio.wav"}
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    audio_data = json.loads(response)
    
    if audio_data["result"]["success"]:
        # 语音识别
        request = {
            "action": "call_function",
            "name": "audio_recognize",
            "arguments": {
                "model_key": model_key,
                "audio_input": audio_data["result"]["audio_base64"],
                "input_type": "base64",
                "hotword": "SensVoice"
            }
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        asr_result = json.loads(response)
        print("识别结果:", asr_result)
    
    await websocket.close()

# 运行示例
asyncio.run(example_usage())
```

### 标点恢复示例

```python
async def punctuation_example():
    websocket = await websockets.connect("ws://localhost:8081")
    
    # 初始化标点模型
    request = {
        "action": "call_function",
        "name": "initialize_model",
        "arguments": {"model_name": "ct-punc", "device": "cpu"}
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    result = json.loads(response)
    model_key = result["result"]["model_key"]
    
    # 标点恢复
    request = {
        "action": "call_function",
        "name": "punctuation_restore",
        "arguments": {
            "model_key": model_key,
            "text": "那今天的会就到这里吧 happy new year 明年见"
        }
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    punc_result = json.loads(response)
    print("标点恢复结果:", punc_result)
    
    await websocket.close()
```

### 实时语音识别示例

```python
async def streaming_asr_example():
    websocket = await websockets.connect("ws://localhost:8081")
    
    # 初始化流式ASR模型
    request = {
        "action": "call_function",
        "name": "initialize_model",
        "arguments": {
            "model_name": "paraformer-zh-streaming",
            "device": "cpu"
        }
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    result = json.loads(response)
    model_key = result["result"]["model_key"]
    
    # 模拟流式音频数据（实际应用中从麦克风获取）
    cache = {}
    for i in range(10):  # 模拟10个音频块
        # 这里应该是实际的音频数据
        audio_chunk_base64 = "..."  # 实际的base64编码音频数据
        is_final = (i == 9)  # 最后一个块
        
        request = {
            "action": "call_function",
            "name": "streaming_asr",
            "arguments": {
                "model_key": model_key,
                "audio_chunk_base64": audio_chunk_base64,
                "cache": cache,
                "is_final": is_final
            }
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        result = json.loads(response)
        
        # 更新缓存
        cache = result["result"]["cache"]
        print(f"块 {i} 识别结果:", result)
    
    await websocket.close()
```

## 注意事项

1. **音频格式**: 输入音频建议使用16kHz的WAV格式
2. **模型下载**: 首次使用时会自动下载模型，请确保网络连接正常
3. **设备选择**: GPU推理速度更快，但需要CUDA环境
4. **内存使用**: 长音频处理可能需要较大内存

## 故障排除

### 模型下载失败
- 检查网络连接
- 尝试更换hub参数（ms或hf）
- 手动下载模型到本地

### 音频识别错误
- 确保音频文件存在且格式正确
- 检查base64编码是否有效
- 验证模型是否正确初始化

### 实时处理问题
- 检查音频块大小是否合适
- 确认缓存对象传递正确
- 验证is_final参数设置

## 扩展开发

服务器采用模块化设计，可以轻松添加新功能：

1. 在`TOOLS`字典中添加新工具定义
2. 实现对应的处理函数
3. 更新客户端以支持新功能

## 许可证

请遵循FunASR项目的许可证要求。