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
            yield f"data: {json.dumps({'type': 'recognition', 'request_id': request_id, 'text': user_text})}\n\n"
            
            # 2. LLM 流式对话
            logger.info(f"开始 LLM 流式对话 [请求ID: {request_id}], 请求内容: {user_text}")
            
            # 初始化状态变量
            json_buffer = ""  # 用于累积JSON数据
            content_buffer = ""  # 用于累积content内容
            current_page = None  # 当前页码
            found_content = False  # 是否找到content字段
            segment_counter = 0  # 分段计数器
            
            # 文本分段参数
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
                try:
                    # 获取文本块
                    if isinstance(chunk, dict):
                        text_chunk = chunk.get("content", "")
                    else:
                        text_chunk = str(chunk)
                    
                    if not text_chunk:
                        continue
                    
                    # 累积到JSON缓冲区
                    json_buffer += text_chunk
                    
                    # 如果还没有找到content字段，尝试解析JSON
                    if not found_content:
                        try:
                            # 尝试解析完整的JSON
                            if json_buffer.strip().endswith('}'):
                                parsed = json.loads(json_buffer.strip())
                                if 'content' in parsed:
                                    content_buffer = parsed['content']
                                    current_page = parsed.get('page')
                                    found_content = True
                                    logger.info(f"解析到完整JSON，页码: {current_page}")
                            else:
                                # 尝试部分解析
                                partial_match = re.search(r'"page":\s*([^,}]+)', json_buffer)
                                if partial_match:
                                    try:
                                        current_page = json.loads(partial_match.group(1).strip())
                                        logger.info(f"提取到页码: {current_page}")
                                    except:
                                        current_page = partial_match.group(1).strip(' "\'')
                                
                                content_match = re.search(r'"content":\s*"([^"]*(?:\\.[^"]*)*)"', json_buffer)
                                if content_match:
                                    content_buffer = content_match.group(1)
                                    found_content = True
                                    logger.info(f"部分解析到content: {content_buffer[:50]}...")
                        except json.JSONDecodeError:
                            # 继续累积，等待更多数据
                            pass
                        except Exception as e:
                            logger.warning(f"JSON解析警告: {e}")
                    else:
                        # 已经找到content，直接处理新的文本块
                        # 检查是否是新的JSON响应开始
                        if text_chunk.strip().startswith('{') or '"content":' in text_chunk:
                            try:
                                # 重新解析新的JSON
                                json_buffer = text_chunk
                                parsed = json.loads(json_buffer.strip()) if json_buffer.strip().endswith('}') else None
                                if parsed and 'content' in parsed:
                                    content_buffer += parsed['content']
                                    if 'page' in parsed:
                                        current_page = parsed['page']
                                        logger.info(f"更新页码: {current_page}")
                                else:
                                    # 部分JSON，添加到content_buffer
                                    content_match = re.search(r'"content":\s*"([^"]*)', text_chunk)
                                    if content_match:
                                        content_buffer += content_match.group(1)
                            except:
                                # 如果解析失败，直接添加到content_buffer
                                clean_chunk = re.sub(r'[{}",]', '', text_chunk)
                                if 'content' not in clean_chunk and 'page' not in clean_chunk:
                                    content_buffer += clean_chunk
                        else:
                            # 清理并添加到content_buffer
                            clean_chunk = text_chunk.replace('\\n', '\n').replace('\\"', '"')
                            content_buffer += clean_chunk
                    
                    # 检查是否可以分段
                    if found_content and len(content_buffer) >= min_segment_length:
                        best_split_pos = -1
                        
                        # 查找最佳分割点
                        for marker in segment_markers:
                            pos = content_buffer.rfind(marker)
                            if pos > best_split_pos:
                                best_split_pos = pos
                        
                        if best_split_pos > 0:
                            # 分割文本
                            segment_text = content_buffer[:best_split_pos + 1].strip()
                            content_buffer = content_buffer[best_split_pos + 1:]
                            
                            if segment_text:
                                segment_counter += 1
                                segment_id = f"{request_id}_seg_{segment_counter}"
                                
                                # 发送文本分段
                                text_data = {
                                    "type": "text",
                                    "segment_id": segment_id,
                                    "text": segment_text
                                }
                                if current_page is not None:
                                    text_data["page"] = current_page
                                
                                yield f"data: {json.dumps(text_data)}\n\n"
                                
                                # 合成语音
                                try:
                                    logger.info(f"合成语音分段 [{segment_id}]: {segment_text[:50]}...")
                                    synthesis_result = await managers['speech_processor'].synthesize(
                                        text=segment_text,
                                        request_id=segment_id
                                    )
                                    
                                    # 处理音频数据
                                    audio_data = synthesis_result.audio_data
                                    if isinstance(audio_data, str):
                                        # 如果已经是base64字符串
                                        try:
                                            base64.b64decode(audio_data)
                                            audio_base64 = audio_data
                                        except:
                                            # 如果不是有效的base64，按字符串处理
                                            audio_base64 = base64.b64encode(audio_data.encode('utf-8')).decode('utf-8')
                                    else:
                                        # 如果是字节数据
                                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                                    
                                    # 发送音频分段
                                    audio_data = {
                                        "type": "audio",
                                        "segment_id": segment_id,
                                        "text": segment_text,
                                        "audio": audio_base64,
                                        "format": synthesis_result.format
                                    }
                                    if current_page is not None:
                                        audio_data["page"] = current_page
                                    
                                    yield f"data: {json.dumps(audio_data)}\n\n"
                                    
                                except Exception as e:
                                    logger.error(f"音频合成失败 [{segment_id}]: {e}")
                                    yield f"data: {json.dumps({'type': 'error', 'message': f'音频合成失败: {str(e)}'})}\n\n"
                
                except Exception as e:
                    logger.error(f"处理文本块失败: {e}")
                    continue
            
            # 处理剩余的content_buffer
            if content_buffer.strip():
                segment_counter += 1
                segment_id = f"{request_id}_final"
                
                # 发送最后一段文本
                text_data = {
                    "type": "text",
                    "segment_id": segment_id,
                    "text": content_buffer.strip()
                }
                if current_page is not None:
                    text_data["page"] = current_page
                
                yield f"data: {json.dumps(text_data)}\n\n"
                
                # 合成最后一段语音
                try:
                    logger.info(f"合成最终语音分段: {content_buffer.strip()[:50]}...")
                    synthesis_result = await managers['speech_processor'].synthesize(
                        text=content_buffer.strip(),
                        request_id=segment_id
                    )
                    
                    # 处理音频数据
                    audio_data = synthesis_result.audio_data
                    if isinstance(audio_data, str):
                        try:
                            base64.b64decode(audio_data)
                            audio_base64 = audio_data
                        except:
                            audio_base64 = base64.b64encode(audio_data.encode('utf-8')).decode('utf-8')
                    else:
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # 发送最终音频分段
                    audio_data = {
                        "type": "audio",
                        "segment_id": segment_id,
                        "text": content_buffer.strip(),
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }
                    if current_page is not None:
                        audio_data["page"] = current_page
                    
                    yield f"data: {json.dumps(audio_data)}\n\n"
                    
                except Exception as e:
                    logger.error(f"最终音频合成失败: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'最终音频合成失败: {str(e)}'})}\n\n"
            
            # 发送完成信号
            processing_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'done', 'request_id': request_id, 'processing_time': processing_time})}\n\n"
            
            logger.info(f"流式语音对话完成 [请求ID: {request_id}] - 总时长: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    # 返回SSE流式响应
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )