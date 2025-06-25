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
    """流式语音对话接口（测试版本）"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"开始流式语音对话测试 [请求ID: {request_id}]")
    
    async def simple_test_generator():
        # 测试消息 1
        message1 = f"data: {json.dumps({'type': 'test', 'request_id': request_id, 'message': 'Test message 1'})}\n\n"
        logger.info(f"发送测试消息1: {message1.strip()}")
        yield message1
        
        await asyncio.sleep(0.5)  # 等待0.5秒
        
        # 测试消息 2
        message2 = f"data: {json.dumps({'type': 'test', 'request_id': request_id, 'message': 'Test message 2'})}\n\n"
        logger.info(f"发送测试消息2: {message2.strip()}")
        yield message2
        
        await asyncio.sleep(0.5)  # 等待0.5秒
        
        # 完成消息
        done_message = f"data: {json.dumps({'type': 'done', 'request_id': request_id, 'processing_time': 1.0})}\n\n"
        logger.info(f"发送完成消息: {done_message.strip()}")
        yield done_message
        
        logger.info(f"简单测试生成器结束 [请求ID: {request_id}]")
    
    return StreamingResponse(
        simple_test_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        }
    )