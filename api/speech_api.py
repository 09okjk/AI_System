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

# ==================== 优化的辅助函数 ====================

async def process_stream_chunk(text_chunk, cache_state, logger, request_id):
    """处理单个流式数据块，提取page和content信息"""
    # 首次接收内容，进行处理
    if not cache_state["content_started"]:
        # 先添加到原始缓冲区
        cache_state["raw_buffer"] += text_chunk
        
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
            content_marker = '"content":'
            content_index = cache_state["raw_buffer"].find(content_marker) + len(content_marker)
            raw_content = cache_state["raw_buffer"][content_index:].lstrip(' "')
            
            # 清理JSON结构
            json_end = raw_content.find('}')
            if json_end >= 0:
                processed_text = raw_content[:json_end].rstrip('"')
            else:
                processed_text = raw_content.rstrip('"')
                
            # 清理嵌套JSON
            json_start = processed_text.find('{"')
            if json_start >= 0:
                processed_text = processed_text[:json_start]
            
            cache_state["content_started"] = True
            cache_state["raw_buffer"] = ""  # 清空原始缓存
            logger.info(f"开始接收content内容: {processed_text[:50]}...")
            
            # 重要：返回处理后的文本，而不是累积
            return processed_text
        else:
            # 尚未找到content标记，继续等待
            return ""
    else:
        # 已经开始接收content
        
        # 检查是否有新的JSON响应开始
        if '"content":' in text_chunk:
            logger.info("检测到新的JSON响应")
            
            # 尝试提取新的页码
            try:
                page_pattern = r'"page"\s*:\s*([^,}]+)'
                match = re.search(page_pattern, text_chunk)
                if match:
                    page_value = match.group(1).strip().strip('"\'')
                    cache_state["current_page"] = page_value
                    logger.info(f"更新页码: {cache_state['current_page']}")
            except Exception as e:
                logger.warning(f"页码更新失败: {str(e)}")
            
            # 提取新的content内容
            content_marker = '"content":'
            content_index = text_chunk.find(content_marker) + len(content_marker)
            raw_content = text_chunk[content_index:].lstrip(' "')
            
            # 清理JSON结构
            json_end = raw_content.find('}')
            if json_end >= 0:
                processed_text = raw_content[:json_end].rstrip('"')
            else:
                processed_text = raw_content.rstrip('"')
                
            # 清理嵌套JSON
            json_start = processed_text.find('{"')
            if json_start >= 0:
                processed_text = processed_text[:json_start]
                
            # 重要：发现新的JSON响应时，重置文本缓冲区
            cache_state["text_buffer"] = ""
            logger.info("检测到新响应，重置文本缓冲区")
            
            return processed_text
        else:
            # 普通文本块，清理后返回
            processed_text = text_chunk
            
            # 清理可能的JSON字符串
            json_start = processed_text.find('{"')
            if json_start >= 0:
                processed_text = processed_text[:json_start]
                
            # 清理结束括号和引号
            json_end = processed_text.find('}')
            if json_end >= 0:
                processed_text = processed_text[:json_end]
                
            processed_text = processed_text.rstrip('",')
            
            return processed_text

async def check_and_process_segment(cache_state, min_segment_length, segment_markers):
    """检查是否需要分段，如果需要则返回分段文本"""
    text_buffer = cache_state["text_buffer"]
    
    # 如果缓冲区为空，不处理
    if not text_buffer:
        return None
    
    # 如果文本长度够长，检查分段点
    if len(text_buffer) >= min_segment_length:
        last_marker_pos = -1
        last_marker_len = 0
        
        # 查找最合适的分段点（最靠近末尾的标点符号）
        for marker in segment_markers:
            pos = text_buffer.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos
                last_marker_len = len(marker)
        
        # 如果找到了分段点
        if last_marker_pos > 0:
            process_index = last_marker_pos + last_marker_len
            segment_text = text_buffer[:process_index]
            
            # 重要：更新缓冲区，移除已处理的文本
            cache_state["text_buffer"] = text_buffer[process_index:]
            cache_state["segment_counter"] += 1
            
            # 记录分段边界，帮助调试
            logger.info(f"分段点: '{text_buffer[last_marker_pos:process_index]}' 位置: {last_marker_pos}")
            logger.info(f"分段文本: '{segment_text}'")
            logger.info(f"剩余缓冲: '{cache_state['text_buffer'][:20]}...'")
            
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

# ==================== 优化的流式语音对话接口 ====================

@router.post("/api/chat/voice/stream")
async def voice_chat_stream(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """流式语音对话接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始流式语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
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
            recognition_response = {
                "type": "recognition",
                "request_id": request_id,
                "text": user_text
            }
            yield json.dumps(recognition_response, ensure_ascii=False) + "\n"
            
            # 2. LLM 流式对话
            logger.info(f"开始 LLM 流式对话 [请求ID: {request_id}]")
            
            text_buffer = ""
            min_segment_length = 40
            segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
            segment_counter = 0
            
            start_time = time.time()
            async for chunk in managers['llm_manager'].stream_chat(
                model_name=llm_model,
                message=user_text,
                system_prompt=system_prompt,
                session_id=session_id,
                request_id=request_id
            ):
                # 简化文本块处理
                if isinstance(chunk, dict):
                    text_chunk = chunk.get("content", "")
                    current_page = chunk.get("page", None)
                else:
                    text_chunk = str(chunk)
                    current_page = None
                
                if not text_chunk:
                    continue
                
                # 清理文本内容（移除JSON格式字符）
                text_chunk = text_chunk.replace('"content":', '').replace('"', '').strip()
                if text_chunk.startswith(':'):
                    text_chunk = text_chunk[1:].strip()
                
                text_buffer += text_chunk
                
                # 检查是否可以分段
                should_process = False
                process_index = len(text_buffer)
                
                if len(text_buffer) >= min_segment_length:
                    for marker in segment_markers:
                        last_marker = text_buffer.rfind(marker)
                        if last_marker > 0:
                            process_index = last_marker + len(marker)
                            should_process = True
                            break
                
                if should_process:
                    segment_text = text_buffer[:process_index]
                    text_buffer = text_buffer[process_index:]
                    segment_counter += 1
                    
                    # 发送文本段
                    text_response = {
                        "type": "text",
                        "segment_id": f"{request_id}_seg_{segment_counter}",
                        "text": segment_text
                    }
                    if current_page:
                        text_response["page"] = current_page
                    
                    yield json.dumps(text_response, ensure_ascii=False) + "\n"
                    
                    # 生成音频
                    try:
                        synthesis_result = await managers['speech_processor'].synthesize(
                            text=segment_text,
                            request_id=f"{request_id}_seg_{segment_counter}"
                        )
                        
                        # 确保音频数据是Base64编码
                        audio_data = synthesis_result.audio_data
                        if isinstance(audio_data, bytes):
                            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        else:
                            audio_base64 = str(audio_data)
                        
                        audio_response = {
                            "type": "audio",
                            "segment_id": f"{request_id}_seg_{segment_counter}",
                            "text": segment_text,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }
                        if current_page:
                            audio_response["page"] = current_page
                            
                        yield json.dumps(audio_response, ensure_ascii=False) + "\n"
                        
                    except Exception as e:
                        logger.error(f"音频合成失败: {str(e)}")
                        error_response = {
                            "type": "error",
                            "message": f"音频合成失败: {str(e)}"
                        }
                        yield json.dumps(error_response, ensure_ascii=False) + "\n"
            
            # 处理最后的文本缓冲区
            if text_buffer.strip():
                segment_counter += 1
                
                # 发送最后的文本
                final_text_response = {
                    "type": "text",
                    "segment_id": f"{request_id}_final",
                    "text": text_buffer.strip()
                }
                yield json.dumps(final_text_response, ensure_ascii=False) + "\n"
                
                # 生成最后的音频
                try:
                    synthesis_result = await managers['speech_processor'].synthesize(
                        text=text_buffer.strip(),
                        request_id=f"{request_id}_final"
                    )
                    
                    audio_data = synthesis_result.audio_data
                    if isinstance(audio_data, bytes):
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    else:
                        audio_base64 = str(audio_data)
                    
                    final_audio_response = {
                        "type": "audio",
                        "segment_id": f"{request_id}_final",
                        "text": text_buffer.strip(),
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }
                    yield json.dumps(final_audio_response, ensure_ascii=False) + "\n"
                    
                except Exception as e:
                    logger.error(f"最终音频合成失败: {str(e)}")
            
            # 发送完成信号
            processing_time = time.time() - start_time
            done_response = {
                "type": "done",
                "request_id": request_id,
                "processing_time": processing_time
            }
            yield json.dumps(done_response, ensure_ascii=False) + "\n"
            
            logger.info(f"流式语音对话完成 [请求ID: {request_id}]")
            
        except Exception as e:
            logger.error(f"流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            error_response = {
                "type": "error",
                "message": str(e)
            }
            yield json.dumps(error_response, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
        }
    )