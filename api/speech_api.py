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
import asyncio

router = APIRouter()

# 全局变量引用
def get_managers():
    """获取全局管理器实例"""
    from main import speech_processor, llm_manager, logger, mongodb_manager
    return {
        'speech_processor': speech_processor,
        'llm_manager': llm_manager,
        'logger': logger,
        'mongodb_manager': mongodb_manager
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

async def process_system_prompt_with_documents(system_prompt: str, mongodb_manager, logger, request_id: str) -> str:
    """
    处理系统提示词中的documentsId，将其替换为实际的文档数据
    
    Args:
        system_prompt: 原始系统提示词
        mongodb_manager: MongoDB管理器实例
        logger: 日志记录器
        request_id: 请求ID
    
    Returns:
        处理后的系统提示词
    """
    if not system_prompt:
        return system_prompt
    
    try:
        # 使用正则表达式匹配 {"documentsId":"..."}
        pattern = r'\{"documentsId":\s*"([^"]+)"\}'
        match = re.search(pattern, system_prompt)
        
        if not match:
            logger.info(f"系统提示词中未找到documentsId [请求ID: {request_id}]")
            return system_prompt
        
        document_id = match.group(1)
        logger.info(f"从系统提示词中提取到documentsId: {document_id} [请求ID: {request_id}]")
        
        # 使用mongodb_manager获取文档数据
        try:
            document = await mongodb_manager.get_document(document_id)
            if not document:
                logger.warning(f"未找到ID为 {document_id} 的文档 [请求ID: {request_id}]")
                return system_prompt
            
            logger.info(f"成功获取文档: {document.name}, 数据项数量: {len(document.data_list)} [请求ID: {request_id}]")
            
            # 构建新的数据格式
            document_data = []
            for item in document.data_list:
                page_data = {
                    "page": item.sequence,
                    "content": item.text
                }
                document_data.append(page_data)
            
            logger.info(f"构建文档数据完成，包含 {len(document_data)} 个页面 [请求ID: {request_id}]")
            
            # 将文档数据转换为JSON字符串
            document_data_json = json.dumps(document_data, ensure_ascii=False)
            
            # 替换原来的{"documentsId":"..."}为生成的数据
            processed_prompt = re.sub(pattern, document_data_json, system_prompt)
            
            logger.info(f"系统提示词处理完成，替换了documentsId为实际文档数据 [请求ID: {request_id}]")
            logger.debug(f"处理后的系统提示词长度: {len(processed_prompt)} [请求ID: {request_id}]")
            
            return processed_prompt
            
        except Exception as e:
            logger.error(f"获取文档 {document_id} 失败: {str(e)} [请求ID: {request_id}]")
            # 如果获取文档失败，返回原始系统提示词
            return system_prompt
            
    except Exception as e:
        logger.error(f"处理系统提示词中的documentsId失败: {str(e)} [请求ID: {request_id}]")
        return system_prompt

class TextSegmentProcessor:
    """文本分段处理器，确保不重复处理相同的文本段"""
    
    def __init__(self, request_id: str, logger, min_segment_length: int = 40):
        self.request_id = request_id
        self.logger = logger
        self.min_segment_length = min_segment_length
        self.segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
        
        # 状态管理
        self.json_buffer = ""
        self.content_buffer = ""
        self.current_page = None
        self.found_content = False
        self.segment_counter = 0
        self.json_complete = False
        self.last_processed_pos = 0  # 修改：使用位置而不是文本来跟踪处理进度
        
        self.logger.info(f"🔧 初始化文本分段处理器 [请求ID: {request_id}]")
    
    def add_chunk(self, text_chunk: str):
        """添加新的文本块"""
        if not text_chunk:
            return
            
        self.logger.debug(f"📝 添加文本块: '{text_chunk[:50]}...', 当前缓冲区长度: {len(self.content_buffer)}")
        
        # 累积到JSON缓冲区
        self.json_buffer += text_chunk
        
        # 如果还没有完成JSON解析
        if not self.json_complete:
            self._try_parse_json()
        else:
            # JSON已经解析完成，检查是否有新的JSON开始
            if text_chunk.strip().startswith('{'):
                self.logger.info("🔄 检测到新的JSON响应开始，重置状态")
                self._reset_for_new_json(text_chunk)
            else:
                # 清理并添加到content_buffer
                clean_chunk = self._clean_text_chunk(text_chunk)
                if clean_chunk and not clean_chunk.isspace():
                    old_length = len(self.content_buffer)
                    self.content_buffer += clean_chunk
                    self.logger.debug(f"➕ 添加清理后的文本: '{clean_chunk[:30]}...', 缓冲区长度: {old_length} -> {len(self.content_buffer)}")
    
    def _try_parse_json(self):
        """尝试解析JSON"""
        try:
            # 尝试解析完整的JSON
            if self.json_buffer.strip().endswith('}'):
                parsed = json.loads(self.json_buffer.strip())
                if 'content' in parsed:
                    # JSON解析完成，只取content内容
                    new_content = parsed['content']
                    self.current_page = parsed.get('page')
                    self.found_content = True
                    self.json_complete = True
                    
                    # 修改：只更新新增的内容，不重置已处理的位置
                    if len(new_content) > len(self.content_buffer):
                        self.content_buffer = new_content
                        self.logger.info(f"✅ 完整解析JSON - 页码: {self.current_page}, 内容长度: {len(self.content_buffer)}")
                    
                    # 清空JSON缓冲区，避免重复处理
                    self.json_buffer = ""
            else:
                # 尝试部分解析
                self._try_partial_parse()
        except json.JSONDecodeError:
            # 继续累积，等待更多数据
            pass
        except Exception as e:
            self.logger.warning(f"⚠️ JSON解析警告: {e}")
    
    def _try_partial_parse(self):
        """尝试部分解析JSON"""
        # 尝试部分解析页码
        if not self.current_page:
            partial_match = re.search(r'"page":\s*([^,}]+)', self.json_buffer)
            if partial_match:
                try:
                    self.current_page = json.loads(partial_match.group(1).strip())
                    self.logger.info(f"📄 提取到页码: {self.current_page}")
                except:
                    self.current_page = partial_match.group(1).strip(' "\'')
        
        # 尝试部分解析content
        if not self.found_content:
            content_match = re.search(r'"content":\s*"([^"]*(?:\\.[^"]*)*)"', self.json_buffer)
            if content_match:
                new_content = content_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                # 修改：只在内容真正增长时才更新
                if len(new_content) > len(self.content_buffer):
                    self.content_buffer = new_content
                    self.found_content = True
                    self.logger.info(f"📝 部分解析到content: {self.content_buffer[:50]}...")
    
    def _reset_for_new_json(self, text_chunk: str):
        """为新的JSON重置状态"""
        self.json_buffer = text_chunk
        self.json_complete = False
        # 修改：不重置已处理位置，避免重复处理
        # self.last_processed_pos 保持不变
    
    def _clean_text_chunk(self, text_chunk: str) -> str:
        """清理文本块"""
        clean_chunk = text_chunk
        # 移除可能的JSON结束符
        if '}' in clean_chunk:
            clean_chunk = clean_chunk.split('}')[0]
        # 清理转义字符
        clean_chunk = clean_chunk.replace('\\n', '\n').replace('\\"', '"')
        # 移除JSON格式字符
        clean_chunk = re.sub(r'^[",\s]+|[",\s]+$', '', clean_chunk)
        return clean_chunk
    
    def get_next_segment(self) -> tuple[str, bool]:
        """
        获取下一个可处理的文本段
        返回: (segment_text, has_more)
        """
        if not self.found_content or len(self.content_buffer) <= self.last_processed_pos:
            return "", False
        
        # 修改：检查剩余未处理的内容长度
        remaining_content = self.content_buffer[self.last_processed_pos:]
        if len(remaining_content) < self.min_segment_length:
            return "", False
        
        # 找到分割点（在剩余内容中查找）
        best_split_pos = -1
        
        # 查找最佳分割点
        for marker in self.segment_markers:
            # 在剩余内容中查找分割点
            pos = remaining_content.find(marker)
            while pos != -1:
                # 确保分割点满足最小长度要求
                if pos >= self.min_segment_length - 1:
                    if pos > best_split_pos:
                        best_split_pos = pos
                    break
                # 继续查找下一个分割点
                pos = remaining_content.find(marker, pos + 1)
        
        if best_split_pos > 0:
            # 提取文本段
            segment_text = remaining_content[:best_split_pos + 1].strip()
            
            if segment_text:
                # 更新已处理位置
                self.last_processed_pos += best_split_pos + 1
                self.segment_counter += 1
                
                self.logger.info(f"✂️ 提取文本段 #{self.segment_counter}: '{segment_text[:50]}...', 已处理位置: {self.last_processed_pos} / {len(self.content_buffer)}")
                
                return segment_text, True
        
        return "", False
    
    def get_final_segment(self) -> str:
        """获取最终剩余的文本段"""
        if self.last_processed_pos < len(self.content_buffer):
            final_text = self.content_buffer[self.last_processed_pos:].strip()
            
            if final_text and len(final_text) > 5:
                self.segment_counter += 1
                self.last_processed_pos = len(self.content_buffer)  # 标记为已全部处理
                
                self.logger.info(f"🏁 提取最终文本段 #{self.segment_counter}: '{final_text[:50]}...', 长度: {len(final_text)}")
                return final_text
        
        return ""
    
    def get_current_page(self):
        """获取当前页码"""
        return self.current_page
    
    def get_segment_counter(self):
        """获取已处理的段落数量"""
        return self.segment_counter

class SimpleTextSegmentProcessor:
    """修复重复内容的简化文本分段处理器"""
    
    def __init__(self, request_id: str, logger, min_segment_length: int = 40):
        self.request_id = request_id
        self.logger = logger
        self.min_segment_length = min_segment_length
        self.segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
        
        # 状态管理
        self.text_buffer = ""
        self.last_processed_pos = 0
        self.segment_counter = 0
        
        # 重复内容检测
        self.processed_segments = set()  # 存储已处理的文本段
        
        self.logger.info(f"🔧 初始化简化文本分段处理器 [请求ID: {request_id}]")
    
    def add_text(self, text_chunk: str):
        """直接添加文本块"""
        if not text_chunk:
            return
            
        old_length = len(self.text_buffer)
        self.text_buffer += text_chunk
        
        self.logger.debug(f"📝 添加文本块: '{text_chunk[:50]}...', 缓冲区长度: {old_length} -> {len(self.text_buffer)}")
    
    def _is_duplicate_content(self, segment_text: str) -> bool:
        """检查是否为重复内容"""
        # 清理文本用于比较（移除空白字符和标点）
        cleaned_text = ''.join(c for c in segment_text if c.isalnum()).lower()
        
        # 检查是否与已处理的段落重复
        for processed_segment in self.processed_segments:
            processed_cleaned = ''.join(c for c in processed_segment if c.isalnum()).lower()
            
            # 如果新段落完全包含在之前的段落中，或者重复度超过80%
            if cleaned_text in processed_cleaned or processed_cleaned in cleaned_text:
                return True
            
            # 计算相似度（简单的字符重复度）
            if len(cleaned_text) > 20 and len(processed_cleaned) > 20:
                common_chars = sum(1 for c in cleaned_text if c in processed_cleaned)
                similarity = common_chars / max(len(cleaned_text), len(processed_cleaned))
                if similarity > 0.8:  # 80%以上重复认为是重复内容
                    return True
        
        return False
    
    def get_next_segment(self) -> tuple[str, bool]:
        """获取下一个可处理的文本段，自动跳过重复内容"""
        max_attempts = 10  # 防止无限循环
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            if len(self.text_buffer) <= self.last_processed_pos:
                self.logger.debug(f"📝 没有新内容可处理: 缓冲区长度={len(self.text_buffer)}, 已处理位置={self.last_processed_pos}")
                return "", False
            
            # 检查剩余未处理的内容长度
            remaining_content = self.text_buffer[self.last_processed_pos:]
            if len(remaining_content) < self.min_segment_length:
                self.logger.debug(f"📝 剩余内容太短: {len(remaining_content)} < {self.min_segment_length}")
                return "", False
            
            # 找到分割点
            best_split_pos = -1
            best_marker = ""
            
            for marker in self.segment_markers:
                pos = remaining_content.find(marker)
                while pos != -1:
                    if pos >= self.min_segment_length - 1:
                        if pos > best_split_pos:
                            best_split_pos = pos
                            best_marker = marker
                        break
                    pos = remaining_content.find(marker, pos + 1)
            
            if best_split_pos > 0:
                # 提取文本段
                segment_text = remaining_content[:best_split_pos + 1].strip()
                
                if segment_text:
                    # 检查是否为重复内容
                    if self._is_duplicate_content(segment_text):
                        self.logger.warning(f"⚠️ 跳过重复内容段落: '{segment_text[:50]}...'")
                        # 跳过这个重复段落，继续查找下一个
                        self.last_processed_pos += best_split_pos + 1
                        continue
                    
                    # 非重复内容，正常处理
                    old_pos = self.last_processed_pos
                    self.last_processed_pos += best_split_pos + 1
                    self.segment_counter += 1
                    
                    # 记录已处理的段落
                    self.processed_segments.add(segment_text)
                    
                    self.logger.info(f"✂️ 提取文本段 #{self.segment_counter}: '{segment_text[:50]}...', 已处理位置: {old_pos} -> {self.last_processed_pos} / {len(self.text_buffer)}")
                    
                    return segment_text, True
            
            # 没有找到合适的分割点，退出循环
            break
        
        self.logger.debug(f"📝 未找到合适的非重复分割点")
        return "", False
    
    def get_final_segment(self) -> str:
        """获取最终剩余的文本段，如果是重复内容则跳过"""
        if self.last_processed_pos < len(self.text_buffer):
            final_text = self.text_buffer[self.last_processed_pos:].strip()
            
            if final_text and len(final_text) > 5:
                # 检查最终段落是否为重复内容
                if self._is_duplicate_content(final_text):
                    self.logger.warning(f"⚠️ 跳过重复的最终段落: '{final_text[:50]}...'")
                    self.last_processed_pos = len(self.text_buffer)  # 标记为已处理
                    return ""
                
                self.segment_counter += 1
                self.last_processed_pos = len(self.text_buffer)
                
                # 记录已处理的段落
                self.processed_segments.add(final_text)
                
                self.logger.info(f"🏁 提取最终文本段 #{self.segment_counter}: '{final_text[:50]}...', 长度: {len(final_text)}")
                return final_text
        
        return ""
    
    def get_segment_counter(self):
        """获取已处理的段落数量"""
        return self.segment_counter
    
@router.post("/api/chat/voice/stream")
async def voice_chat_stream(
    audio_file: UploadFile = File(...),
    llm_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None
):
    """简化的流式语音对话接口（语音输入 + 流式文本和语音输出）"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"🚀 开始简化流式语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    async def create_stream_generator():
        logger.info(f"🔧 创建简化流式生成器 [请求ID: {request_id}]")
        
        try:
            # 发送开始信号
            start_message = f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'message': 'Stream started'})}\n\n"
            logger.info(f"📡 发送流式开始信号 [请求ID: {request_id}]")
            yield start_message

            # 处理系统提示词中的documentsId
            processed_system_prompt = system_prompt
            if system_prompt:
                logger.info(f"📋 开始处理系统提示词 [请求ID: {request_id}]")
                processed_system_prompt = await process_system_prompt_with_documents(
                    system_prompt, managers['mongodb_manager'], logger, request_id
                )

            logger.info(f"📋 处理后的系统提示词 [请求ID: {request_id}]: {processed_system_prompt}")
            
            # 1. 语音识别
            try:
                audio_data = await audio_file.read()
                logger.info(f"🎤 执行语音识别 [请求ID: {request_id}] - 音频大小: {len(audio_data)} bytes")
                
                recognition_result = await managers['speech_processor'].recognize(
                    audio_data=audio_data,
                    request_id=request_id
                )
                
                user_text = recognition_result.text
                user_text = re.sub(r'<\s*\|\s*[^|]+\s*\|\s*>', '', user_text).strip()
                logger.info(f"✅ 识别结果 [请求ID: {request_id}]: {user_text}")
                
                # 发送识别结果
                recognition_message = f"data: {json.dumps({'type': 'recognition', 'request_id': request_id, 'text': user_text})}\n\n"
                yield recognition_message
                
            except Exception as e:
                logger.error(f"❌ 语音识别失败 [请求ID: {request_id}]: {str(e)}")
                error_message = f"data: {json.dumps({'type': 'error', 'message': f'语音识别失败: {str(e)}'})}\n\n"
                yield error_message
                return
            
            # 2. LLM 流式对话 - 简化版本
            try:
                logger.info(f"🤖 开始 LLM 流式对话 [请求ID: {request_id}], 请求内容: {user_text}")
                
                # 初始化简化的文本分段处理器
                text_processor = SimpleTextSegmentProcessor(request_id, logger)
                
                start_time = time.time()
                chunk_count = 0
                
                async for chunk in managers['llm_manager'].stream_chat(
                    model_name=llm_model,
                    message=user_text,
                    system_prompt=processed_system_prompt,
                    session_id=session_id,
                    request_id=request_id
                ):
                    try:
                        chunk_count += 1
                        
                        # 获取文本块 - 直接处理纯文本
                        if isinstance(chunk, dict):
                            text_chunk = chunk.get("content", "")
                        else:
                            text_chunk = str(chunk)
                        
                        if not text_chunk:
                            logger.debug(f"⏭️ 跳过空文本块 [请求ID: {request_id}] - 块 {chunk_count}")
                            continue
                        
                        logger.debug(f"📝 处理文本块 [{chunk_count}]: '{text_chunk[:30]}...'")
                        
                        # 直接添加文本块到处理器
                        text_processor.add_text(text_chunk)
                        
                        # 添加调试信息
                        # if chunk_count % 10 == 0:  # 每10个块输出一次状态
                        #     text_processor.debug_state()
                        
                        # 检查是否有可处理的文本段
                        while True:
                            segment_text, has_more = text_processor.get_next_segment()
                            if not segment_text:
                                break
                            
                            logger.info(f"📤 处理文本段 #{text_processor.get_segment_counter()}: '{segment_text[:50]}...' (长度: {len(segment_text)})")
                            
                            # 发送文本段
                            segment_id = f"{request_id}_seg_{text_processor.get_segment_counter()}"
                            
                            text_data = {
                                "type": "text",
                                "segment_id": segment_id,
                                "text": segment_text
                            }
                            
                            text_message = f"data: {json.dumps(text_data)}\n\n"
                            logger.info(f"📤 发送文本段 [{segment_id}]: {len(segment_text)} 字符")
                            yield text_message
                            
                            # 合成并发送语音
                            try:
                                logger.info(f"🎵 开始合成语音 [{segment_id}]: '{segment_text[:50]}...'")
                                
                                synthesis_result = await managers['speech_processor'].synthesize(
                                    text=segment_text,
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
                                
                                # 发送音频分段
                                audio_response = {
                                    "type": "audio",
                                    "segment_id": segment_id,
                                    "text": segment_text,
                                    "audio": audio_base64,
                                    "format": synthesis_result.format
                                }
                                
                                audio_message = f"data: {json.dumps(audio_response)}\n\n"
                                logger.info(f"🎵✅ 音频合成完成 [{segment_id}]: {len(audio_message)} 字节")
                                yield audio_message
                                
                            except Exception as e:
                                logger.error(f"❌ 音频合成失败 [{segment_id}]: {e}")
                                error_message = f"data: {json.dumps({'type': 'error', 'message': f'音频合成失败: {str(e)}'})}\n\n"
                                yield error_message
                            
                            await asyncio.sleep(0.1)
                    
                    except Exception as e:
                        logger.error(f"❌ 处理文本块失败: {e}")
                        continue
                
                logger.info(f"✅ LLM流式对话完成 [请求ID: {request_id}] - 总共处理 {chunk_count} 个文本块")
                
                # 处理最终剩余的文本
                final_text = text_processor.get_final_segment()
                if final_text:
                    logger.info(f"🏁 处理最终文本段: '{final_text[:50]}...' (长度: {len(final_text)})")
                    
                    # 发送最终文本段
                    final_segment_id = f"{request_id}_final"
                    
                    text_data = {
                        "type": "text",
                        "segment_id": final_segment_id,
                        "text": final_text
                    }
                    
                    final_text_message = f"data: {json.dumps(text_data)}\n\n"
                    logger.info(f"📤 发送最终文本段 [{final_segment_id}]: {len(final_text)} 字符")
                    yield final_text_message
                    
                    # 合成并发送最终语音
                    try:
                        logger.info(f"🎵 开始合成最终语音 [{final_segment_id}]: '{final_text[:50]}...'")
                        
                        synthesis_result = await managers['speech_processor'].synthesize(
                            text=final_text,
                            request_id=final_segment_id
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
                        audio_response = {
                            "type": "audio",
                            "segment_id": final_segment_id,
                            "text": final_text,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }
                        
                        final_audio_message = f"data: {json.dumps(audio_response)}\n\n"
                        logger.info(f"🎵✅ 最终音频合成完成 [{final_segment_id}]: {len(final_audio_message)} 字节")
                        yield final_audio_message
                        
                    except Exception as e:
                        logger.error(f"❌ 最终音频合成失败 [{final_segment_id}]: {e}")
                        error_message = f"data: {json.dumps({'type': 'error', 'message': f'最终音频合成失败: {str(e)}'})}\n\n"
                        yield error_message
                
                # 发送完成信号
                processing_time = time.time() - start_time
                done_data = {
                    'type': 'done', 
                    'request_id': request_id, 
                    'processing_time': processing_time,
                    'segments_processed': text_processor.get_segment_counter()
                }
                done_message = f"data: {json.dumps(done_data)}\n\n"
                logger.info(f"🎉 处理完成: {text_processor.get_segment_counter()} 个文本段，耗时 {processing_time:.2f}s")
                yield done_message
                
            except Exception as e:
                logger.error(f"❌ LLM流式对话失败 [请求ID: {request_id}]: {str(e)}")
                error_message = f"data: {json.dumps({'type': 'error', 'message': f'LLM对话失败: {str(e)}'})}\n\n"
                yield error_message
            
            logger.info(f"✅ 简化流式语音对话完成 [请求ID: {request_id}]")
            
        except Exception as e:
            logger.error(f"❌ 流式语音对话失败 [请求ID: {request_id}]: {str(e)}")
            error_message = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield error_message
        
        logger.info(f"🔚 简化流式生成器结束 [请求ID: {request_id}]")
    
    # 返回SSE流式响应
    logger.info(f"🚀 返回StreamingResponse [请求ID: {request_id}]")
    
    response = StreamingResponse(
        create_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )
    
    return response