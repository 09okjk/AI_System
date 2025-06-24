"""
语音处理相关接口模块 - 修复文本重复问题
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

# ==================== 优化的流式处理类 ====================
class StreamProcessor:
    """流式文本处理器 - 防止重复问题的改进版本"""
    
    def __init__(self, request_id: str, logger):
        self.request_id = request_id
        self.logger = logger
        
        # 状态管理
        self.raw_buffer = ""
        self.text_buffer = ""
        self.content_started = False
        self.current_page = None
        self.segment_counter = 0
        
        # 去重机制
        self.processed_chunks: Set[str] = set()  # 已处理的文本块哈希
        self.last_processed_content = ""  # 上一次处理的内容
        
        # 配置
        self.min_segment_length = 40
        self.segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
        self.max_buffer_size = 10000  # 最大缓冲区大小
    
    def _get_content_hash(self, content: str) -> str:
        """生成内容哈希用于去重"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def process_chunk(self, text_chunk: str) -> str:
        """
        处理单个文本块，返回可用于合成的文本 - 改进版本
        """
        if not text_chunk or not text_chunk.strip():
            return ""
        
        # 生成当前块的哈希
        chunk_hash = self._get_content_hash(text_chunk)
        
        # 检查是否已经处理过这个块
        if chunk_hash in self.processed_chunks:
            self.logger.debug(f"跳过重复的文本块: {text_chunk[:30]}...")
            return ""
        
        # 添加到已处理集合
        self.processed_chunks.add(chunk_hash)
        
        # 限制已处理集合的大小，防止内存泄漏
        if len(self.processed_chunks) > 1000:
            # 保留最近的500个
            recent_chunks = list(self.processed_chunks)[-500:]
            self.processed_chunks = set(recent_chunks)
        
        # 添加到原始缓冲区
        self.raw_buffer += text_chunk
        
        # 限制缓冲区大小
        if len(self.raw_buffer) > self.max_buffer_size:
            self.raw_buffer = self.raw_buffer[-self.max_buffer_size//2:]
            self.logger.warning("原始缓冲区过大，已截断")
        
        # 如果还没开始接收内容，检查是否包含content标记
        if not self.content_started:
            return self._check_content_start()
        else:
            # 已经开始接收内容，直接处理
            return self._process_content_chunk(text_chunk)
    
    def _check_content_start(self) -> str:
        """检查是否开始接收内容 - 改进版本"""
        content_marker = '"content":'
        
        if content_marker not in self.raw_buffer:
            return ""
        
        # 提取页码信息
        self._extract_page_info()
        
        # 找到内容标记，提取内容
        content_index = self.raw_buffer.find(content_marker) + len(content_marker)
        content_text = self.raw_buffer[content_index:].lstrip(' "')
        
        # 更彻底的JSON清理
        content_text = self._clean_json_artifacts(content_text)
        
        # 设置状态
        self.content_started = True
        self.last_processed_content = content_text
        self.raw_buffer = ""  # 清空原始缓冲区
        
        self.logger.info(f"开始接收内容: {content_text[:50]}...")
        return content_text
    
    def _process_content_chunk(self, text_chunk: str) -> str:
        """处理内容块 - 改进版本"""
        # 检查是否有新的JSON响应开始
        if '"page":' in text_chunk:
            self._extract_page_from_chunk(text_chunk)
        
        # 清理JSON标记
        cleaned_text = self._clean_json_artifacts(text_chunk)
        
        # 移除可能重复的content标记和其他JSON结构
        if '"content":' in cleaned_text:
            content_index = cleaned_text.find('"content":') + len('"content":')
            cleaned_text = cleaned_text[content_index:].lstrip(' "')
            self.logger.debug("移除重复的content标记")
        
        # 检查与上次处理内容的重复
        if cleaned_text and cleaned_text == self.last_processed_content:
            self.logger.debug(f"跳过与上次相同的内容: {cleaned_text[:30]}...")
            return ""
        
        # 检查是否是前一内容的子串（部分重复）
        if cleaned_text and self.last_processed_content:
            if cleaned_text in self.last_processed_content or self.last_processed_content in cleaned_text:
                # 如果是扩展内容，只返回新增部分
                if len(cleaned_text) > len(self.last_processed_content):
                    new_content = cleaned_text[len(self.last_processed_content):].lstrip()
                    self.last_processed_content = cleaned_text
                    return new_content
                else:
                    return ""
        
        self.last_processed_content = cleaned_text
        return cleaned_text
    
    def _clean_json_artifacts(self, text: str) -> str:
        """更彻底的JSON结构标记清理"""
        if not text:
            return ""
        
        # 移除JSON结束标记
        text = re.sub(r'\s*["}]+\s*$', '', text)
        
        # 移除其他JSON标记
        text = text.replace(',"', ' ').replace('"}', '').replace('\\"', '"')
        
        # 处理换行符
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        
        # 移除多余的引号
        text = re.sub(r'^"+(.*?)"*$', r'\1', text)
        
        # 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_page_info(self):
        """从原始缓冲区提取页码信息"""
        try:
            page_pattern = r'"page"\s*:\s*([^,}]+)'
            match = re.search(page_pattern, self.raw_buffer)
            if match:
                page_value = match.group(1).strip().strip('"\'')
                if page_value != self.current_page:
                    self.current_page = page_value
                    self.logger.info(f"提取到页码: {self.current_page}")
        except Exception as e:
            self.logger.warning(f"页码提取失败: {str(e)}")
    
    def _extract_page_from_chunk(self, text_chunk: str):
        """从文本块中提取页码"""
        try:
            page_pattern = r'"page"\s*:\s*([^,}]+)'
            match = re.search(page_pattern, text_chunk)
            if match:
                page_value = match.group(1).strip().strip('"\'')
                if page_value != self.current_page:
                    self.current_page = page_value
                    self.logger.info(f"更新页码: {self.current_page}")
        except Exception as e:
            self.logger.warning(f"页码更新失败: {str(e)}")
    
    def add_to_buffer(self, text: str):
        """添加文本到缓冲区 - 改进版本"""
        if text and text.strip():
            # 避免添加重复内容
            if text != self.text_buffer[-len(text):] if len(self.text_buffer) >= len(text) else True:
                self.text_buffer += text
                
                # 限制缓冲区大小
                if len(self.text_buffer) > self.max_buffer_size:
                    self.text_buffer = self.text_buffer[-self.max_buffer_size//2:]
                    self.logger.warning("文本缓冲区过大，已截断")
    
    def check_segment(self) -> Optional[str]:
        """检查是否可以分段 - 改进版本"""
        if len(self.text_buffer) < self.min_segment_length:
            return None
        
        # 查找最适合的分段点
        best_split_point = -1
        for marker in self.segment_markers:
            marker_pos = self.text_buffer.rfind(marker)
            if marker_pos > best_split_point:
                best_split_point = marker_pos
        
        if best_split_point > 0:
            # 分段点包含标点符号
            split_point = best_split_point + 1
            segment = self.text_buffer[:split_point].strip()
            self.text_buffer = self.text_buffer[split_point:].strip()
            self.segment_counter += 1
            
            # 验证分段不为空且不重复
            if segment and len(segment) > 5:
                self.logger.info(f"文本分段 #{self.segment_counter}: {segment[:50]}...")
                return segment
        
        return None
    
    def get_final_segment(self) -> Optional[str]:
        """获取最终剩余的文本段 - 改进版本"""
        if self.text_buffer.strip() and len(self.text_buffer.strip()) > 5:
            final_text = self.text_buffer.strip()
            self.text_buffer = ""
            self.logger.info(f"最终文本段: {final_text[:50]}...")
            return final_text
        return None
    
    def cleanup(self):
        """清理资源"""
        self.processed_chunks.clear()
        self.raw_buffer = ""
        self.text_buffer = ""
        self.last_processed_content = ""

# ==================== 修复的流式语音对话接口 ====================

@router.post("/api/chat/voice/stream")
async def voice_chat_stream(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """流式语音对话接口（语音输入 + 流式文本和语音输出）- 修复重复问题"""
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
            
            # 2. 初始化流式处理器
            processor = StreamProcessor(request_id, logger)
            
            # 3. LLM 流式对话
            logger.info(f"开始 LLM 流式对话 [请求ID: {request_id}]")
            
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
                
                # 处理文本块
                processed_text = processor.process_chunk(text_chunk)
                if not processed_text:
                    continue
                
                # 添加到缓冲区
                processor.add_to_buffer(processed_text)
                
                # 检查是否可以分段
                segment = processor.check_segment()
                if segment:
                    # 发送文本段
                    segment_id = f"{request_id}_seg_{processor.segment_counter}"
                    text_response = {
                        "type": "text",
                        "segment_id": segment_id,
                        "text": segment
                    }
                    
                    if processor.current_page:
                        text_response["page"] = processor.current_page
                    
                    yield json.dumps(text_response) + "\n"
                    logger.info(f"发送文本分段 [ID: {segment_id}]: {segment[:50]}...")
                    
                    # 生成语音
                    try:
                        logger.info(f"为分段文本合成语音 [长度: {len(segment)}], 内容: {segment[:30]}...")
                        
                        synthesis_result = await managers['speech_processor'].synthesize(
                            text=segment,  # 直接使用处理后的segment，避免重复
                            request_id=segment_id
                        )
                        
                        # 处理音频数据
                        audio_data = synthesis_result.audio_data
                        if isinstance(audio_data, str):
                            # 检查是否已经是Base64
                            try:
                                base64.b64decode(audio_data)
                                audio_base64 = audio_data
                            except:
                                audio_base64 = base64.b64encode(audio_data.encode('utf-8')).decode('utf-8')
                        elif isinstance(audio_data, bytes):
                            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        else:
                            audio_base64 = base64.b64encode(str(audio_data).encode('utf-8')).decode('utf-8')
                        
                        # 发送音频段
                        audio_response = {
                            "type": "audio",
                            "segment_id": segment_id,
                            "text": segment,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }
                        
                        if processor.current_page:
                            audio_response["page"] = processor.current_page
                        
                        yield json.dumps(audio_response) + "\n"
                        logger.info(f"发送音频分段 [ID: {segment_id}]: 音频长度 {len(audio_base64)} 字符")
                        
                    except Exception as e:
                        logger.error(f"音频合成失败 [分段: {segment_id}]: {str(e)}")
                        yield json.dumps({
                            "type": "error",
                            "segment_id": segment_id,
                            "message": f"音频合成失败: {str(e)}"
                        }) + "\n"
            
            # 4. 处理最终剩余文本
            final_segment = processor.get_final_segment()
            if final_segment:
                segment_id = f"{request_id}_final"
                
                # 发送最终文本
                text_response = {
                    "type": "text",
                    "segment_id": segment_id,
                    "text": final_segment
                }
                
                if processor.current_page:
                    text_response["page"] = processor.current_page
                
                yield json.dumps(text_response) + "\n"
                logger.info(f"发送最终文本 [ID: {segment_id}]: {final_segment[:50]}...")
                
                # 生成最终语音
                try:
                    logger.info(f"为最终文本合成语音 [长度: {len(final_segment)}]")
                    
                    synthesis_result = await managers['speech_processor'].synthesize(
                        text=final_segment,
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
                    elif isinstance(audio_data, bytes):
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    else:
                        audio_base64 = base64.b64encode(str(audio_data).encode('utf-8')).decode('utf-8')
                    
                    # 发送最终音频
                    audio_response = {
                        "type": "audio",
                        "segment_id": segment_id,
                        "text": final_segment,
                        "audio": audio_base64,
                        "format": synthesis_result.format
                    }
                    
                    if processor.current_page:
                        audio_response["page"] = processor.current_page
                    
                    yield json.dumps(audio_response) + "\n"
                    logger.info(f"发送最终音频 [ID: {segment_id}]: 音频长度 {len(audio_base64)} 字符")
                    
                except Exception as e:
                    logger.error(f"最终音频合成失败: {str(e)}")
                    yield json.dumps({
                        "type": "error",
                        "segment_id": segment_id,
                        "message": f"最终音频合成失败: {str(e)}"
                    }) + "\n"
            
            # 5. 发送完成信号
            processing_time = time.time() - start_time
            yield json.dumps({
                "type": "done",
                "request_id": request_id,
                "processing_time": processing_time,
                "page": processor.current_page
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