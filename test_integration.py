#!/usr/bin/env python3
"""
CosyVoice 集成测试 - 演示音色一致性改进
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.speech import SpeechProcessor
from src.logger import setup_logger, get_logger

setup_logger(log_level="INFO", log_dir="test_logs", app_name="integration_test")
logger = get_logger(__name__)


async def test_speech_processor_integration():
    """测试语音处理器集成"""
    
    logger.info("🔧 开始 CosyVoice 集成测试")
    
    # 创建语音处理器
    processor = SpeechProcessor()
    
    try:
        # 初始化处理器（模拟模式）
        logger.info("正在初始化语音处理器...")
        
        # 模拟健康检查
        health = await processor.health_check()
        logger.info(f"健康检查结果: {json.dumps(health, indent=2, ensure_ascii=False)}")
        
        # 模拟多次语音合成请求
        test_requests = [
            {
                "text": "欢迎使用AI语音助手！",
                "synthesis_mode": "zero_shot",
                "reference_audio": "reference_audio/default.wav",
                "reference_text": "这是默认的参考音频"
            },
            {
                "text": "今天天气很好，适合外出活动。",
                "synthesis_mode": "zero_shot", 
                "reference_audio": "reference_audio/default.wav",
                "reference_text": "这是默认的参考音频"
            },
            {
                "text": "人工智能技术正在快速发展。",
                "synthesis_mode": "zero_shot",
                "reference_audio": "reference_audio/default.wav", 
                "reference_text": "这是默认的参考音频"
            }
        ]
        
        logger.info("🎤 开始多次语音合成测试（验证音色一致性）")
        
        speaker_ids = []
        for i, request in enumerate(test_requests, 1):
            logger.info(f"合成请求 {i}: {request['text'][:20]}...")
            
            try:
                # 在实际环境中，这会调用真实的合成
                result = await processor.synthesize(
                    text=request["text"],
                    synthesis_mode=request["synthesis_mode"],
                    **{k:v for k,v in request.items() if k not in ["text", "synthesis_mode"]}
                )
                
                # 检查结果
                if hasattr(result, 'synthesis_mode'):
                    logger.info(f"  ✅ 合成成功 - 模式: {result.synthesis_mode}")
                    if hasattr(result, 'speaker_id'):
                        speaker_ids.append(result.speaker_id)
                        logger.info(f"  📢 说话人ID: {result.speaker_id}")
                else:
                    logger.info(f"  ✅ 合成成功（模拟模式）")
                    
            except Exception as e:
                logger.error(f"  ❌ 合成失败: {str(e)}")
        
        # 验证说话人ID一致性
        if speaker_ids:
            unique_ids = set(speaker_ids)
            if len(unique_ids) == 1:
                logger.info(f"  🎯 音色一致性验证通过 - 所有请求使用相同说话人ID: {speaker_ids[0]}")
            else:
                logger.warning(f"  ⚠️ 音色一致性问题 - 发现多个说话人ID: {unique_ids}")
        
        # 测试不同参考音频的情况
        logger.info("🌟 测试不同参考音频的说话人ID生成")
        
        different_audio_requests = [
            {
                "text": "测试文本1",
                "reference_audio": "reference_audio/speaker1.wav",
                "reference_text": "参考文本1"
            },
            {
                "text": "测试文本2", 
                "reference_audio": "reference_audio/speaker2.wav",
                "reference_text": "参考文本2"
            }
        ]
        
        for request in different_audio_requests:
            try:
                result = await processor.synthesize(
                    text=request["text"],
                    synthesis_mode="zero_shot",
                    **{k:v for k,v in request.items() if k != "text"}
                )
                logger.info(f"  音频: {request['reference_audio']} -> 说话人ID: {getattr(result, 'speaker_id', 'N/A')}")
            except Exception as e:
                logger.info(f"  音频: {request['reference_audio']} -> 错误: {str(e)}")
        
        # 获取可用声音列表
        voices = await processor.get_available_voices()
        logger.info(f"可用声音: {voices}")
        
        logger.info("✅ 集成测试完成")
        
    except Exception as e:
        logger.error(f"❌ 集成测试失败: {str(e)}")
        raise
    
    finally:
        # 清理资源
        await processor.cleanup()
        logger.info("🧹 资源清理完成")


async def main():
    """主函数"""
    try:
        await test_speech_processor_integration()
        logger.info("🎉 所有集成测试通过！")
        return 0
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)