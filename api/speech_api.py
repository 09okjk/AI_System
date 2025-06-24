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
    """流式语音对话接口（语音输入 + 流式文本和语音输出）- 优化版本"""
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
            yield json.dumps({
                "type": "recognition",
                "request_id": request_id,
                "text": user_text
            }) + "\n"
            
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
                        # 发送文本段和音频
                        await send_text_and_audio_segment(
                            segment_text, cache_state, request_id, managers, logger
                        )
            
            # 处理剩余缓存内容
            if cache_state["text_buffer"]:
                await send_final_segment(
                    cache_state, request_id, managers, logger
                )
            
            # 发送完成信号
            processing_time = time.time() - start_time
            yield json.dumps({
                "type": "done",
                "request_id": request_id,
                "processing_time": processing_time,
                "page": cache_state["current_page"]  # 在完成信号中也包含页码
            }) + "\n"
            
            logger.info(f"流式语音对话完成 [请求ID: {request_id}] - 总时长: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson"
    )

# 辅助函数 - 处理流式数据块
async def process_stream_chunk(text_chunk, cache_state, logger, request_id):
    """处理单个流式数据块，提取page和content信息"""
    cache_state["raw_buffer"] += text_chunk
    
    if not cache_state["content_started"]:
        # 尝试提取page属性
        if '"page":' in cache_state["raw_buffer"]:
            try:
                # 更稳健的JSON解析方式
                import re
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
        return text_chunk
    
    return ""

# 辅助函数 - 检查并处理分段
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

# 辅助函数 - 发送文本和音频分段
async def send_text_and_audio_segment(segment_text, cache_state, request_id, managers, logger):
    """发送文本分段和对应的音频"""
    segment_id = f"{request_id}_seg_{cache_state['segment_counter']}"
    
    # 发送文本分段
    text_response = {
        "type": "text",
        "segment_id": segment_id,
        "text": segment_text
    }
    
    if cache_state["current_page"]:
        text_response["page"] = cache_state["current_page"]
    
    yield json.dumps(text_response) + "\n"
    
    # 生成并发送音频
    try:
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
        
        yield json.dumps(audio_response) + "\n"
        
    except Exception as e:
        logger.error(f"音频合成失败 [分段: {segment_id}]: {str(e)}")
        yield json.dumps({
            "type": "error",
            "segment_id": segment_id,
            "message": f"音频合成失败: {str(e)}"
        }) + "\n"

# 辅助函数 - 处理音频数据
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

# 辅助函数 - 发送最终分段
async def send_final_segment(cache_state, request_id, managers, logger):
    """处理并发送最后剩余的文本内容"""
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
    
    yield json.dumps(text_response) + "\n"
    
    # 生成最终音频
    try:
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
        
        yield json.dumps(audio_response) + "\n"
        
    except Exception as e:
        logger.error(f"最终音频合成失败: {str(e)}")
        yield json.dumps({
            "type": "error",
            "segment_id": segment_id,
            "message": f"最终音频合成失败: {str(e)}"
        }) + "\n"
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
                
                # logger.info(f"LLM 流式对话响应 [请求ID: {request_id}], 响应内容: {text_chunk}")

                # 初始化状态跟踪变量
                if not hasattr(stream_generator, 'found_content_marker'):
                    stream_generator.found_content_marker = False
                    stream_generator.raw_buffer = ""
                    # 添加计数器，避免无限等待
                    stream_generator.chunks_processed = 0
                    # 添加页码记录
                    stream_generator.current_page = None

                # 检查是否超过一定数量的块还未找到标记，如果是则放弃等待
                if not stream_generator.found_content_marker:
                    stream_generator.chunks_processed += 1
                    # 如果处理了10个块还没找到标记，放弃等待直接处理
                    if stream_generator.chunks_processed > 10:
                        logger.warning(f"已处理{stream_generator.chunks_processed}个文本块但未找到content标记，将直接处理文本")
                        stream_generator.found_content_marker = True
                
                processed_chunk = ""
                # 处理文本块
                if not stream_generator.found_content_marker:
                    # 在发现content标记前，先累积到原始缓冲区
                    stream_generator.raw_buffer += text_chunk
                    
                    # 检查是否包含了content标记
                    content_marker = '"content":'
                    if content_marker in stream_generator.raw_buffer:
                        # 尝试提取页码
                        try:
                            page_marker = '"page":'
                            if page_marker in stream_generator.raw_buffer:
                                page_start = stream_generator.raw_buffer.find(page_marker) + len(page_marker)
                                page_end = stream_generator.raw_buffer.find(',', page_start)
                                if page_end == -1:
                                    page_end = stream_generator.raw_buffer.find('}', page_start)
                                if page_end > page_start:
                                    page_str = stream_generator.raw_buffer[page_start:page_end].strip()
                                    stream_generator.current_page = page_str.strip(' "\'')
                                    logger.info(f"提取到页码: {stream_generator.current_page}")
                        except:
                            # 如果提取失败也不影响主流程
                            pass
                            
                        # 找到content标记
                        content_index = stream_generator.raw_buffer.find(content_marker) + len(content_marker)
                        # 只保留标记后面的文本
                        processed_chunk = stream_generator.raw_buffer[content_index:].lstrip(' "')
                        logger.info(f"检测到content标记，开始提取内容: {processed_chunk}")
                        # 设置标志，之后的文本直接进入主缓冲区
                        stream_generator.found_content_marker = True
                        # 不再需要原始缓冲区
                        stream_generator.raw_buffer = ""
                        
                        # 清空之前可能已经添加的JSON开头部分
                        text_buffer = ""
                        logger.info("清空之前的文本缓冲区，避免JSON结构重复")
                    else:
                        # 未找到content标记，跳过当前块
                        logger.info("等待content标记...")
                        # 但不影响已经缓存的处理
                        if text_buffer:
                            # 如果有待处理的缓冲，继续处理它
                            processed_chunk = ""
                        else:
                            # 否则跳过处理
                            continue
                else:
                    # 已经找到content标记，直接处理文本
                    processed_chunk = text_chunk
                    
                    # 检查是否有新的JSON响应开始
                    if '"content":' in processed_chunk:
                        logger.info("检测到新的content标记")
                        
                        # 尝试提取新的页码
                        try:
                            page_marker = '"page":'
                            if page_marker in processed_chunk:
                                page_start = processed_chunk.find(page_marker) + len(page_marker)
                                page_end = processed_chunk.find(',', page_start)
                                if page_end == -1:
                                    page_end = processed_chunk.find('}', page_start)
                                if page_end > page_start:
                                    page_str = processed_chunk[page_start:page_end].strip()
                                    stream_generator.current_page = page_str.strip(' "\'')
                                    logger.info(f"提取到新页码: {stream_generator.current_page}")
                        except:
                            # 如果提取失败也不影响主流程
                            pass
                    
                    # 尝试移除JSON结束标记
                    try:
                        json_end = processed_chunk.find('}')
                        if json_end >= 0:
                            processed_chunk = processed_chunk[:json_end]
                            logger.info("移除JSON结束标记")
                    except:
                        # 如果移除失败也不影响主流程
                        pass
                
                # 添加到文本缓冲区
                text_buffer += processed_chunk
                
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
                    response_data = {
                        "type": "text", 
                        "segment_id": f"{request_id}_{int(time.time())}", 
                        "text": segment_text
                    }
                    
                    # 添加页码信息
                    if hasattr(stream_generator, 'current_page') and stream_generator.current_page is not None:
                        response_data["page"] = stream_generator.current_page
                    
                    yield json.dumps(response_data) + "\n"
                    
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
                        response_data = {
                            "type": "audio",
                            "segment_id": f"{request_id}_{int(time.time())}",
                            "text": segment_text,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }
                        
                        # 添加页码信息
                        if hasattr(stream_generator, 'current_page') and stream_generator.current_page is not None:
                            response_data["page"] = stream_generator.current_page
                            
                        yield json.dumps(response_data) + "\n"
                    except Exception as e:
                        logger.error(f"音频合成失败: {str(e)}")
                        yield json.dumps({
                            "type": "error",
                            "message": f"音频合成失败: {str(e)}"
                        }) + "\n"
            
            # 处理剩余的文本缓冲区
            if text_buffer:
                # 发送最后一段文本
                response_data = {
                    "type": "text",
                    "segment_id": f"{request_id}_final",
                    "text": text_buffer
                }
                
                # 添加页码信息
                if hasattr(stream_generator, 'current_page') and stream_generator.current_page is not None:
                    response_data["page"] = stream_generator.current_page
                    
                yield json.dumps(response_data) + "\n"
                
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
                    response_data = {
                        "type": "audio",
                        "segment_id": f"{request_id}_final",
                        "text": text_buffer,
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }
                    
                    # 添加页码信息
                    if hasattr(stream_generator, 'current_page') and stream_generator.current_page is not None:
                        response_data["page"] = stream_generator.current_page
                        
                    yield json.dumps(response_data) + "\n"
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