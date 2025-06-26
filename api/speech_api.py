"""
修复版语音处理相关接口模块
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from starlette.responses import StreamingResponse
from typing import Optional, Set, List, Tuple
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
import hashlib
from difflib import SequenceMatcher

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

class EnhancedTextSegmentProcessor:
    """增强版文本分段处理器 - 修复重复内容问题"""
    
    def __init__(self, request_id: str, logger, min_segment_length: int = 40):
        self.request_id = request_id
        self.logger = logger
        self.min_segment_length = min_segment_length
        self.segment_markers = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
        
        # 核心状态管理
        self.accumulated_text = ""  # 累积的完整文本
        self.processed_text = ""    # 已经处理过的文本
        self.segment_counter = 0
        
        # 重复内容检测
        self.processed_segments = []  # 存储已处理的文本段
        self.segment_hashes = set()   # 存储段落哈希值
        
        self.logger.info(f"🔧 初始化增强版文本分段处理器 [请求ID: {request_id}]")
    
    def _calculate_text_hash(self, text: str) -> str:
        """计算文本的哈希值"""
        normalized_text = re.sub(r'\s+', '', text).lower()
        return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """标准化文本用于比较"""
        # 移除多余的空白字符和标点符号
        normalized = re.sub(r'[，。！？；：""''（）\s]+', '', text)
        return normalized.lower()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        norm1 = self._normalize_text_for_comparison(text1)
        norm2 = self._normalize_text_for_comparison(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # 使用序列匹配计算相似度
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity
    
    def _is_duplicate_content(self, segment_text: str) -> bool:
        """改进的重复内容检测"""
        if not segment_text.strip():
            return True
        
        # 计算当前段落的哈希值
        current_hash = self._calculate_text_hash(segment_text)
        if current_hash in self.segment_hashes:
            self.logger.debug(f"🔍 检测到完全重复的段落哈希: {current_hash}")
            return True
        
        # 检查与已处理段落的相似度
        for processed_segment in self.processed_segments:
            similarity = self._calculate_similarity(segment_text, processed_segment)
            
            if similarity > 0.85:  # 提高相似度阈值
                self.logger.debug(f"🔍 检测到高相似度段落: {similarity:.2f} - '{segment_text[:30]}...' vs '{processed_segment[:30]}...'")
                return True
            
            # 检查是否一个段落完全包含另一个段落
            norm_current = self._normalize_text_for_comparison(segment_text)
            norm_processed = self._normalize_text_for_comparison(processed_segment)
            
            if len(norm_current) > 10 and len(norm_processed) > 10:
                if norm_current in norm_processed or norm_processed in norm_current:
                    self.logger.debug(f"🔍 检测到包含关系的重复段落")
                    return True
        
        return False
    
    def update_accumulated_text(self, new_complete_text: str):
        """更新累积文本"""
        if not new_complete_text:
            return
            
        old_length = len(self.accumulated_text)
        self.accumulated_text = new_complete_text
        
        self.logger.debug(f"📝 更新累积文本: 长度 {old_length} -> {len(self.accumulated_text)}")
    
    def extract_incremental_text(self) -> str:
        """提取新增的文本内容"""
        if len(self.accumulated_text) <= len(self.processed_text):
            return ""
        
        # 确保累积文本包含已处理文本
        if not self.accumulated_text.startswith(self.processed_text):
            # 如果不是简单的前缀关系，使用更智能的差异检测
            self.logger.warning(f"⚠️ 累积文本不是已处理文本的前缀，使用智能差异检测")
            return self._extract_diff_intelligently()
        
        incremental_text = self.accumulated_text[len(self.processed_text):]
        self.logger.debug(f"📝 提取增量文本: '{incremental_text[:50]}...' (长度: {len(incremental_text)})")
        
        return incremental_text
    
    def _extract_diff_intelligently(self) -> str:
        """智能提取文本差异"""
        # 使用序列匹配器找到最长公共子序列
        matcher = SequenceMatcher(None, self.processed_text, self.accumulated_text)
        
        # 找到最佳匹配块
        matching_blocks = matcher.get_matching_blocks()
        
        if not matching_blocks:
            # 如果没有匹配块，返回全部累积文本
            return self.accumulated_text
        
        # 找到最后一个有意义的匹配块
        last_match = matching_blocks[-2] if len(matching_blocks) > 1 else matching_blocks[0]
        
        # 从最后匹配位置之后提取新文本
        start_pos = last_match.a + last_match.size
        if start_pos < len(self.accumulated_text):
            return self.accumulated_text[start_pos:]
        
        return ""
    
    def get_next_segment(self) -> Tuple[str, bool]:
        """获取下一个可处理的文本段"""
        incremental_text = self.extract_incremental_text()
        
        if not incremental_text:
            return "", False
        
        # 在增量文本中查找分割点
        if len(incremental_text) < self.min_segment_length:
            return "", False
        
        # 找到最佳分割点
        best_split_pos = -1
        
        for marker in self.segment_markers:
            pos = incremental_text.find(marker)
            while pos != -1:
                if pos >= self.min_segment_length - 1:
                    if pos > best_split_pos:
                        best_split_pos = pos
                    break
                pos = incremental_text.find(marker, pos + 1)
        
        if best_split_pos <= 0:
            return "", False
        
        # 提取文本段
        segment_text = incremental_text[:best_split_pos + 1].strip()
        
        if not segment_text:
            return "", False
        
        # 检查是否为重复内容
        if self._is_duplicate_content(segment_text):
            self.logger.warning(f"⚠️ 跳过重复内容段落: '{segment_text[:50]}...'")
            # 标记这部分文本为已处理，但不生成输出
            self.processed_text += incremental_text[:best_split_pos + 1]
            return self.get_next_segment()  # 递归查找下一个非重复段落
        
        # 更新已处理文本
        self.processed_text += incremental_text[:best_split_pos + 1]
        self.segment_counter += 1
        
        # 记录已处理的段落
        self.processed_segments.append(segment_text)
        self.segment_hashes.add(self._calculate_text_hash(segment_text))
        
        self.logger.info(f"✂️ 提取文本段 #{self.segment_counter}: '{segment_text[:50]}...' (长度: {len(segment_text)})")
        
        return segment_text, True
    
    def get_final_segment(self) -> str:
        """获取最终剩余的文本段"""
        incremental_text = self.extract_incremental_text()
        
        if not incremental_text or len(incremental_text.strip()) <= 5:
            return ""
        
        final_text = incremental_text.strip()
        
        # 检查最终段落是否为重复内容
        if self._is_duplicate_content(final_text):
            self.logger.warning(f"⚠️ 跳过重复的最终段落: '{final_text[:50]}...'")
            return ""
        
        self.segment_counter += 1
        self.processed_text = self.accumulated_text  # 标记全部为已处理
        
        # 记录已处理的段落
        self.processed_segments.append(final_text)
        self.segment_hashes.add(self._calculate_text_hash(final_text))
        
        self.logger.info(f"🏁 提取最终文本段 #{self.segment_counter}: '{final_text[:50]}...' (长度: {len(final_text)})")
        return final_text
    
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
    """修复版流式语音对话接口"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"🚀 开始修复版流式语音对话 [请求ID: {request_id}] - 会话ID: {session_id}")
    
    async def create_stream_generator():
        logger.info(f"🔧 创建修复版流式生成器 [请求ID: {request_id}]")
        
        try:
            # 发送开始信号
            start_message = f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'message': 'Stream started'})}\n\n"
            logger.info(f"📡 发送流式开始信号 [请求ID: {request_id}]")
            yield start_message

            # 处理系统提示词
            processed_system_prompt = system_prompt
            if system_prompt:
                logger.info(f"📋 开始处理系统提示词 [请求ID: {request_id}]")
                processed_system_prompt = await process_system_prompt_with_documents(
                    system_prompt, managers['mongodb_manager'], logger, request_id
                )

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
            
            # 2. LLM 流式对话 - 使用增强版处理器
            try:
                logger.info(f"🤖 开始 LLM 流式对话 [请求ID: {request_id}], 请求内容: {user_text}")
                
                # 初始化增强版文本分段处理器
                text_processor = EnhancedTextSegmentProcessor(request_id, logger)
                
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
                        
                        # 获取当前完整文本
                        if isinstance(chunk, dict):
                            current_complete_text = chunk.get("content", "")
                        else:
                            current_complete_text = str(chunk)
                        
                        if not current_complete_text:
                            logger.debug(f"⏭️ 跳过空文本块 [请求ID: {request_id}] - 块 {chunk_count}")
                            continue
                        
                        logger.debug(f"📝 处理文本块 [{chunk_count}]: '{current_complete_text[:50]}...' (长度: {len(current_complete_text)})")
                        
                        # 更新累积文本
                        text_processor.update_accumulated_text(current_complete_text)
                        
                        # 检查是否有可处理的文本段
                        while True:
                            segment_text, has_more = text_processor.get_next_segment()
                            if not segment_text:
                                break
                            
                            logger.info(f"📤 处理文本段 #{text_processor.get_segment_counter()}: '{segment_text[:50]}...' (长度: {len(segment_text)})")
                            
                            # 生成段落ID
                            segment_id = f"{request_id}_seg_{text_processor.get_segment_counter()}"
                            
                            # 发送文本段
                            text_data = {
                                "type": "text",
                                "segment_id": segment_id,
                                "text": segment_text
                            }
                            
                            text_message = f"data: {json.dumps(text_data)}\n\n"
                            yield text_message
                            
                            # 合成并发送语音
                            try:                                
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
                                
                                # 发送音频响应
                                audio_response = {
                                    "type": "audio",
                                    "segment_id": segment_id,
                                    "text": segment_text,
                                    "audio": audio_base64,
                                    "format": synthesis_result.format
                                }
                                
                                audio_message = f"data: {json.dumps(audio_response)}\n\n"
                                logger.info(f"🎵✅ 音频合成完成 [{segment_id}]: {len(audio_base64)} bytes base64")
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
                    
                    final_segment_counter = text_processor.get_segment_counter()
                    final_segment_id = f"{request_id}_seg_{final_segment_counter}"
                    
                    # 发送最终文本段
                    text_data = {
                        "type": "text",
                        "segment_id": final_segment_id,
                        "text": final_text
                    }
                    final_text_message = f"data: {json.dumps(text_data)}\n\n"
                    yield final_text_message
                    
                    # 合成并发送最终语音
                    try:
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
                        
                        audio_response = {
                            "type": "audio",
                            "segment_id": final_segment_id,
                            "text": final_text,
                            "audio": audio_base64,
                            "format": synthesis_result.format
                        }
                        
                        final_audio_message = f"data: {json.dumps(audio_response)}\n\n"
                        yield final_audio_message
                        
                    except Exception as e:
                        logger.error(f"❌ 最终音频合成失败: {e}")
                
                # 发送完成信号
                processing_time = time.time() - start_time
                done_data = {
                    'type': 'done', 
                    'request_id': request_id, 
                    'processing_time': processing_time,
                    'segments_processed': text_processor.get_segment_counter()
                }
                done_message = f"data: {json.dumps(done_data)}\n\n"
                yield done_message
                
            except Exception as e:
                logger.error(f"❌ LLM流式对话失败: {e}")
                error_message = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                yield error_message
            
        except Exception as e:
            logger.error(f"❌ 流式语音对话失败: {e}")
            error_message = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield error_message
    
    return StreamingResponse(
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