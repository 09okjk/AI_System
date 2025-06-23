"""
语音处理相关接口模块
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from starlette.responses import StreamingResponse
from typing import Optional
import re
from src.models import (
    SpeechRecognitionResponse, SpeechSynthesisResponse, 
    SpeechSynthesisRequest, VoiceChatResponse
)
from src.utils import generate_response_id
import json
import time
import base64

router = APIRouter()

# 全局变量引用
def get_managers():
    """获取全局管理器实例"""
    from main import speech_processor, llm_manager, logger
    return {
        'speech_processor': speech_processor,
        'llm_manager': llm_manager,
        'logger': logger
    }

# ==================== 语音处理接口 ====================

@router.post("/api/speech/recognize", response_model=SpeechRecognitionResponse)
async def recognize_speech(
    audio_file: UploadFile = File(...),
    language: Optional[str] = "zh-CN",
    use_asr_model: Optional[str] = None
):
    """语音识别接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始语音识别 [请求ID: {request_id}] - 文件: {audio_file.filename}")
    
    try:
        # 验证文件类型
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="文件必须是音频格式")
        
        # 读取音频数据
        audio_data = await audio_file.read()
        logger.info(f"音频文件读取完成 [请求ID: {request_id}] - 大小: {len(audio_data)} bytes")
        
        # 执行语音识别
        result = await managers['speech_processor'].recognize(
            audio_data=audio_data,
            language=language,
            model_name=use_asr_model,
            request_id=request_id
        )
        
        logger.info(f"语音识别完成 [请求ID: {request_id}] - 文本长度: {len(result.text)}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音识别失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/speech/synthesize", response_model=SpeechSynthesisResponse)
async def synthesize_speech(request: SpeechSynthesisRequest):
    """语音合成接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始语音合成 [请求ID: {request_id}] - 文本长度: {len(request.text)}")
    
    try:
        # 执行语音合成
        result = await managers['speech_processor'].synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            pitch=request.pitch,
            tts_model=request.tts_model,
            request_id=request_id
        )
        
        logger.info(f"语音合成完成 [请求ID: {request_id}] - 音频大小: {len(result.audio_data)} bytes")
        return result
        
    except Exception as e:
        logger.error(f"语音合成失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat/voice", response_model=VoiceChatResponse)
async def voice_chat(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """语音对话接口（语音输入 + 文本和语音输出）"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    try:
        # 1. 语音识别
        audio_data = await audio_file.read()
        logger.info(f"执行语音识别 [请求ID: {request_id}]")
        
        recognition_result = await managers['speech_processor'].recognize(
            audio_data=audio_data,
            request_id=request_id
        )
        
        user_text = recognition_result.text
        user_text = re.sub(r'<\s*\|\s*[^|]+\s*\|\s*>', '', user_text).strip()
        logger.info(f"识别结果 [请求ID: {request_id}]: {user_text}")
        
        # 2. LLM 对话
        logger.info(f"调用 LLM 模型 [请求ID: {request_id}], 请求内容: {user_text}, 系统提示: {system_prompt}")
        
        chat_response = await managers['llm_manager'].chat(
            model_name=llm_model,
            message=user_text,
            system_prompt=system_prompt,
            session_id=session_id,
            request_id=request_id
        )
        
        response_text = chat_response["content"]
        logger.info(f"LLM 响应 [请求ID: {request_id}]: {response_text[:100]}...")
        
        # 3. 语音合成
        logger.info(f"执行语音合成 [请求ID: {request_id}]")
        
        synthesis_result = await managers['speech_processor'].synthesize(
            text=response_text,
            request_id=request_id
        )
        
        logger.info(f"语音对话完成 [请求ID: {request_id}]")
        
        return VoiceChatResponse(
            request_id=request_id,
            user_text=user_text,
            response_text=response_text,
            response_audio=synthesis_result.audio_data,
            audio_format=synthesis_result.format,
            session_id=session_id or request_id,
            model_used=chat_response["model_name"],
            processing_time={
                "recognition": recognition_result.processing_time,
                "llm_chat": chat_response["processing_time"],
                "synthesis": synthesis_result.processing_time
            }
        )
        
    except Exception as e:
        logger.error(f"语音对话失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat/voice/stream")
async def voice_chat_stream(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """流式语音对话接口（语音输入 + 流式文本和语音输出）"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始流式语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    # 为流式响应创建生成器函数
    async def stream_generator():
        try:
            # 1. 语音识别
            audio_data = await audio_file.read()
            logger.info(f"执行语音识别 [请求ID: {request_id}]")
            
            recognition_result = await managers['speech_processor'].recognize(
                audio_data=audio_data,
                request_id=request_id
            )
            
            user_text = recognition_result.text
            user_text = re.sub(r'<\s*\|\s*[^|]+\s*\|\s*>', '', user_text).strip()
            logger.info(f"识别结果 [请求ID: {request_id}]: {user_text}")
            
            # 发送识别结果
            yield json.dumps({
                "type": "recognition",
                "request_id": request_id,
                "text": user_text
            }) + "\n"
            
            # 2. LLM 流式对话
            logger.info(f"开始 LLM 流式对话 [请求ID: {request_id}], 请求内容: {user_text}, 系统提示: {system_prompt}")
            
            # 文本缓冲区，用于分段处理
            text_buffer = ""
            # 文本分段的最小长度（中文约30-50字为宜）
            min_segment_length = 40
            # 定义分段标记（可以是标点符号或自然段落）
            segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
            
            # 调用流式LLM对话
            start_time = time.time()
            async for chunk in managers['llm_manager'].stream_chat(
                model_name=llm_model,
                message=user_text,
                system_prompt=system_prompt,
                session_id=session_id,
                request_id=request_id
            ):
                # 获取文本块
                if isinstance(chunk, dict):
                    text_chunk = chunk.get("content", "")
                else:
                    text_chunk = chunk
                    
                if not text_chunk:
                    continue
                
                logger.info(f"LLM 流式对话响应 [请求ID: {request_id}], 响应内容: {text_chunk}")

                # 检查是否是JSON格式，如果是则提取content字段
                try:
                    # 尝试解析为JSON
                    if isinstance(text_chunk, str) and text_chunk.strip().startswith('{'):
                        json_data = json.loads(text_chunk)
                        if 'content' in json_data:
                            text_chunk = json_data.get('content', '')
                            logger.info(f"提取到JSON中的content: {text_chunk}")
                except Exception as json_err:
                    # 如果解析失败，使用原始文本
                    pass
                
                # 累积到缓冲区
                text_buffer += text_chunk
                
                # 检查是否可以分段处理
                should_process = False
                process_index = len(text_buffer)
                
                # 如果文本长度超过最小分段长度，检查是否有合适的分段点
                if len(text_buffer) >= min_segment_length:
                    for marker in segment_markers:
                        last_marker = text_buffer.rfind(marker)
                        if last_marker > 0:  # 找到标记
                            process_index = last_marker + len(marker)
                            should_process = True
                            break
                
                # 如果找到分段点，处理此段
                if should_process:
                    segment_text = text_buffer[:process_index]
                    text_buffer = text_buffer[process_index:]
                    
                    # 发送文本分段
                    yield json.dumps({
                        "type": "text",
                        "segment_id": f"{request_id}_{int(time.time())}",
                        "text": segment_text
                    }) + "\n"
                    
                    # 3. 生成此段的语音
                    logger.info(f"为分段文本合成语音 [长度: {len(segment_text)}], 内容: {segment_text}")
                    
                    try:
                        synthesis_result = await managers['speech_processor'].synthesize(
                            text=segment_text,
                            request_id=f"{request_id}_seg_{int(time.time())}"
                        )
                        
                        def is_already_base64(data):
                            try:
                                # 尝试解码，如果成功则可能已经是Base64
                                base64.b64decode(data)
                                return True
                            except:
                                return False
                        
                        # 确保音频数据是字节类型，然后转换为Base64
                        audio_data = synthesis_result.audio_data
                        # 如果已经是Base64，直接使用
                        if isinstance(audio_data, str) and is_already_base64(audio_data):
                            audio_base64 = audio_data
                        # 否则进行Base64编码
                        elif isinstance(audio_data, bytes):
                            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        
                        # 发送音频段
                        yield json.dumps({
                            "type": "audio",
                            "segment_id": f"{request_id}_{int(time.time())}",
                            "text": segment_text,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }) + "\n"
                    except Exception as e:
                        logger.error(f"音频合成失败: {str(e)}")
                        yield json.dumps({
                            "type": "error",
                            "message": f"音频合成失败: {str(e)}"
                        }) + "\n"
            
            # 处理剩余的文本缓冲区
            if text_buffer:
                # 发送最后一段文本
                yield json.dumps({
                    "type": "text",
                    "segment_id": f"{request_id}_final",
                    "text": text_buffer
                }) + "\n"
                
                # 为最后一段文本合成语音
                try:
                    synthesis_result = await managers['speech_processor'].synthesize(
                        text=text_buffer,
                        request_id=f"{request_id}_final"
                    )
                    
                    # 确保音频数据是字节类型，然后转换为Base64
                    audio_data = synthesis_result.audio_data
                    if isinstance(audio_data, str):
                        # 如果是字符串，需要先编码为字节
                        audio_data = audio_data.encode('utf-8')
                        
                    # 将二进制音频数据转换为Base64编码的字符串
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # 发送最后一段音频
                    yield json.dumps({
                        "type": "audio",
                        "segment_id": f"{request_id}_final",
                        "text": text_buffer,
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }) + "\n"
                except Exception as e:
                    logger.error(f"最终音频合成失败: {str(e)}")
                    yield json.dumps({
                        "type": "error",
                        "message": f"最终音频合成失败: {str(e)}"
                    }) + "\n"
            
            # 发送完成信号
            processing_time = time.time() - start_time
            yield json.dumps({
                "type": "done",
                "request_id": request_id,
                "processing_time": processing_time
            }) + "\n"
            
            logger.info(f"流式语音对话完成 [请求ID: {request_id}] - 总时长: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"
    
    # 返回流式响应
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson"
    )