"""
语音处理相关接口模块 - 修复版本
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
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
import asyncio

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

# ==================== 优化的辅助函数 ====================

async def process_stream_chunk(text_chunk, cache_state, logger, request_id):
    """处理单个流式数据块，提取page和content信息"""
    cache_state["raw_buffer"] += text_chunk
    
    if not cache_state["content_started"]:
        # 尝试提取page属性
        if '"page":' in cache_state["raw_buffer"]:
            try:
                # 更稳健的JSON解析方式
                page_pattern = r'"page"\s*:\s*([^,}]+)'
                match = re.search(page_pattern, cache_state["raw_buffer"])
                if match:
                    page_value = match.group(1).strip().strip('"\'')
                    cache_state["current_page"] = page_value
                    logger.info(f"提取到页码: {cache_state['current_page']}")
            except Exception as e:
                logger.warning(f"页码提取失败: {str(e)}")
        
        # 检查是否开始接收content
        if '"content":' in cache_state["raw_buffer"]:
            content_index = cache_state["raw_buffer"].find('"content":') + len('"content":')
            processed_text = cache_state["raw_buffer"][content_index:].lstrip(' "')
            cache_state["content_started"] = True
            cache_state["raw_buffer"] = ""  # 清空原始缓存
            logger.info(f"开始接收content内容: {processed_text[:50]}...")
            return processed_text
    else:
        # 已经开始接收content，直接返回文本
        # 检查是否有新的JSON响应开始，并提取新的页码
        if '"page":' in text_chunk:
            try:
                page_pattern = r'"page"\s*:\s*([^,}]+)'
                match = re.search(page_pattern, text_chunk)
                if match:
                    page_value = match.group(1).strip().strip('"\'')
                    cache_state["current_page"] = page_value
                    logger.info(f"更新页码: {cache_state['current_page']}")
            except Exception as e:
                logger.warning(f"页码更新失败: {str(e)}")
        
        # 清理可能的JSON结构标记
        cleaned_text = text_chunk
        if '"content":' in cleaned_text:
            content_index = cleaned_text.find('"content":') + len('"content":')
            cleaned_text = cleaned_text[content_index:].lstrip(' "')
        
        # 移除可能的JSON结束标记
        if '}' in cleaned_text:
            json_end = cleaned_text.find('}')
            cleaned_text = cleaned_text[:json_end]
        
        return cleaned_text
    
    return ""

async def check_and_process_segment(cache_state, min_segment_length, segment_markers):
    """检查是否需要分段，如果需要则返回分段文本"""
    text_buffer = cache_state["text_buffer"]
    
    if len(text_buffer) >= min_segment_length:
        for marker in segment_markers:
            last_marker = text_buffer.rfind(marker)
            if last_marker > 0:
                process_index = last_marker + len(marker)
                segment_text = text_buffer[:process_index]
                cache_state["text_buffer"] = text_buffer[process_index:]
                cache_state["segment_counter"] += 1
                return segment_text
    
    return None

async def process_audio_data(audio_data):
    """统一处理音频数据转Base64"""
    def is_already_base64(data):
        try:
            base64.b64decode(data)
            return True
        except:
            return False
    
    if isinstance(audio_data, str) and is_already_base64(audio_data):
        return audio_data
    elif isinstance(audio_data, bytes):
        return base64.b64encode(audio_data).decode('utf-8')
    else:
        # 如果是其他类型，尝试转换为bytes再编码
        return base64.b64encode(str(audio_data).encode('utf-8')).decode('utf-8')

# ==================== 修复的流式语音对话接口 ====================

@router.post("/api/chat/voice/stream")
async def voice_chat_stream(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """流式语音对话接口（语音输入 + 流式文本和语音输出）- 修复版本"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始流式语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    async def stream_generator():
        try:
            # 发送开始标记 - 立即发送以测试连接
            yield "data: " + json.dumps({
                "type": "start",
                "request_id": request_id,
                "timestamp": time.time()
            }) + "\n\n"
            
            # 强制刷新缓冲区
            await asyncio.sleep(0.01)
            
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
            recognition_data = json.dumps({
                "type": "recognition",
                "request_id": request_id,
                "text": user_text
            })
            yield "data: " + recognition_data + "\n\n"
            logger.info(f"发送识别结果: {recognition_data}")
            
            # 强制刷新
            await asyncio.sleep(0.01)
            
            # 2. LLM 流式对话 - 优化缓存处理
            logger.info(f"开始 LLM 流式对话 [请求ID: {request_id}]")
            
            # 优化的缓存状态管理
            cache_state = {
                "raw_buffer": "",           # 原始JSON数据缓存
                "text_buffer": "",          # 提取的文本内容缓存
                "current_page": None,       # 当前页码
                "content_started": False,   # 是否开始接收content内容
                "segment_counter": 0        # 分段计数器
            }
            
            # 分段配置
            min_segment_length = 40
            segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
            
            start_time = time.time()
            async for chunk in managers['llm_manager'].stream_chat(
                model_name=llm_model,
                message=user_text,
                system_prompt=system_prompt,
                session_id=session_id,
                request_id=request_id
            ):
                # 获取文本块
                text_chunk = chunk.get("content", "") if isinstance(chunk, dict) else chunk
                if not text_chunk:
                    continue
                
                # 处理流式数据缓存
                processed_text = await process_stream_chunk(
                    text_chunk, cache_state, logger, request_id
                )
                
                if processed_text:
                    cache_state["text_buffer"] += processed_text
                    
                    # 检查是否需要分段处理
                    segment_text = await check_and_process_segment(
                        cache_state, min_segment_length, segment_markers
                    )
                    
                    if segment_text:
                        # 发送文本段和音频 - 修复yield问题
                        segment_id = f"{request_id}_seg_{cache_state['segment_counter']}"
                        
                        # 发送文本分段
                        text_response = {
                            "type": "text",
                            "segment_id": segment_id,
                            "text": segment_text
                        }
                        
                        if cache_state["current_page"]:
                            text_response["page"] = cache_state["current_page"]
                        
                        text_data = json.dumps(text_response)
                        yield "data: " + text_data + "\n\n"
                        logger.info(f"发送文本分段 [ID: {segment_id}]: {segment_text[:50]}...")
                        
                        # 强制刷新
                        await asyncio.sleep(0.01)
                        
                        # 生成并发送音频
                        try:
                            logger.info(f"为分段文本合成语音 [长度: {len(segment_text)}]")
                            
                            synthesis_result = await managers['speech_processor'].synthesize(
                                text=segment_text,
                                request_id=segment_id
                            )
                            
                            # 处理音频数据
                            audio_base64 = await process_audio_data(synthesis_result.audio_data)
                            
                            audio_response = {
                                "type": "audio",
                                "segment_id": segment_id,
                                "text": segment_text,
                                "audio": audio_base64,
                                "format": synthesis_result.format
                            }
                            
                            if cache_state["current_page"]:
                                audio_response["page"] = cache_state["current_page"]
                            
                            audio_data = json.dumps(audio_response)
                            yield "data: " + audio_data + "\n\n"
                            logger.info(f"发送音频分段 [ID: {segment_id}]: 音频长度 {len(audio_base64)} 字符")
                            
                            # 强制刷新
                            await asyncio.sleep(0.01)
                            
                        except Exception as e:
                            logger.error(f"音频合成失败 [分段: {segment_id}]: {str(e)}")
                            error_data = json.dumps({
                                "type": "error",
                                "segment_id": segment_id,
                                "message": f"音频合成失败: {str(e)}"
                            })
                            yield "data: " + error_data + "\n\n"
            
            # 处理剩余缓存内容
            if cache_state["text_buffer"]:
                final_text = cache_state["text_buffer"]
                segment_id = f"{request_id}_final"
                
                # 发送最终文本
                text_response = {
                    "type": "text",
                    "segment_id": segment_id,
                    "text": final_text
                }
                
                if cache_state["current_page"]:
                    text_response["page"] = cache_state["current_page"]
                
                text_data = json.dumps(text_response)
                yield "data: " + text_data + "\n\n"
                logger.info(f"发送最终文本 [ID: {segment_id}]: {final_text[:50]}...")
                
                # 强制刷新
                await asyncio.sleep(0.01)
                
                # 生成最终音频
                try:
                    logger.info(f"为最终文本合成语音 [长度: {len(final_text)}]")
                    
                    synthesis_result = await managers['speech_processor'].synthesize(
                        text=final_text,
                        request_id=segment_id
                    )
                    
                    audio_base64 = await process_audio_data(synthesis_result.audio_data)
                    
                    audio_response = {
                        "type": "audio",
                        "segment_id": segment_id,
                        "text": final_text,
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }
                    
                    if cache_state["current_page"]:
                        audio_response["page"] = cache_state["current_page"]
                    
                    audio_data = json.dumps(audio_response)
                    yield "data: " + audio_data + "\n\n"
                    logger.info(f"发送最终音频 [ID: {segment_id}]: 音频长度 {len(audio_base64)} 字符")
                    
                    # 强制刷新
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"最终音频合成失败: {str(e)}")
                    error_data = json.dumps({
                        "type": "error",
                        "segment_id": segment_id,
                        "message": f"最终音频合成失败: {str(e)}"
                    })
                    yield "data: " + error_data + "\n\n"
            
            # 发送完成信号
            processing_time = time.time() - start_time
            done_data = json.dumps({
                "type": "done",
                "request_id": request_id,
                "processing_time": processing_time,
                "page": cache_state["current_page"]  # 在完成信号中也包含页码
            })
            yield "data: " + done_data + "\n\n"
            
            logger.info(f"流式语音对话完成 [请求ID: {request_id}] - 总时长: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            error_data = json.dumps({
                "type": "error",
                "message": str(e)
            })
            yield "data: " + error_data + "\n\n"
    
    # 使用Server-Sent Events格式
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用nginx缓冲
        }
    )