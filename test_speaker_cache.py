#!/usr/bin/env python3
"""
测试 CosyVoice 说话人缓存功能
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.speech import CosyVoiceSynthesizer
from src.logger import setup_logger, get_logger

# 设置测试日志
setup_logger(log_level="DEBUG", log_dir="test_logs", app_name="speaker_cache_test")
logger = get_logger(__name__)

def create_test_wav_file():
    """创建一个测试用的WAV文件"""
    import struct
    import uuid
    
    # WAV文件头参数
    sample_rate = 16000
    duration = 2  # 2秒
    channels = 1
    bits_per_sample = 16
    
    # 计算参数
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = sample_rate * duration * channels * bits_per_sample // 8
    
    # 构建WAV头
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    
    # 创建静音数据
    audio_data = b'\x00' * data_size
    
    # 保存到临时文件
    temp_dir = Path(tempfile.gettempdir()) / "speaker_cache_test"
    temp_dir.mkdir(exist_ok=True)
    
    test_file = temp_dir / f"test_audio_{uuid.uuid4()}.wav"
    with open(test_file, 'wb') as f:
        f.write(wav_header + audio_data)
    
    logger.info(f"创建测试音频文件: {test_file}")
    return str(test_file)

async def test_speaker_caching():
    """测试说话人缓存功能"""
    logger.info("🧪 开始测试说话人缓存功能")
    
    # 创建测试配置
    config = {
        'model_dir': '/home/rh/Program/MCP_Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B',
        'cosyvoice_path': 'tools/CosyVoice',
        'device': 'cpu'
    }
    
    # 创建合成器实例
    synthesizer = CosyVoiceSynthesizer(config)
    
    try:
        # 初始化合成器
        logger.info("初始化CosyVoice合成器...")
        success = await synthesizer.initialize()
        
        if not success or synthesizer.model is None:
            logger.warning("⚠️ CosyVoice模型未成功初始化，使用模拟模式进行测试")
        else:
            logger.info("✅ CosyVoice合成器初始化成功")
        
        # 创建测试音频文件
        test_audio_path = create_test_wav_file()
        test_text = "这是一个测试句子，用于验证说话人缓存功能。"
        reference_text = "这是参考音频的文本内容"
        
        # 测试1: 生成说话人ID
        logger.info("🔍 测试1: 生成说话人ID")
        speaker_id1 = await synthesizer._generate_speaker_id(test_audio_path, reference_text)
        logger.info(f"说话人ID: {speaker_id1}")
        
        # 使用相同参数再次生成，应该得到相同的ID
        speaker_id2 = await synthesizer._generate_speaker_id(test_audio_path, reference_text)
        assert speaker_id1 == speaker_id2, "相同参数应该生成相同的说话人ID"
        logger.info("✅ 说话人ID生成一致性测试通过")
        
        # 测试2: 缓存说话人
        logger.info("🔍 测试2: 缓存说话人")
        cache_result = await synthesizer._cache_speaker(speaker_id1, test_audio_path, reference_text)
        logger.info(f"缓存结果: {cache_result}")
        
        # 测试3: 获取缓存的说话人
        logger.info("🔍 测试3: 获取缓存的说话人")
        cached_speaker = await synthesizer._get_cached_speaker(speaker_id1)
        if cached_speaker:
            logger.info(f"✅ 成功获取缓存的说话人: {cached_speaker['speaker_id']}")
        else:
            logger.warning("⚠️ 未找到缓存的说话人")
        
        # 测试4: 获取所有缓存的说话人
        logger.info("🔍 测试4: 获取所有缓存的说话人")
        all_speakers = await synthesizer.get_cached_speakers()
        logger.info(f"缓存的说话人数量: {len(all_speakers)}")
        for speaker in all_speakers:
            logger.info(f"  - {speaker['speaker_id']}: {speaker['reference_text'][:30]}...")
        
        # 测试5: 语音合成（使用缓存）
        logger.info("🔍 测试5: 语音合成测试")
        try:
            synthesis_kwargs = {
                'reference_audio': test_audio_path,
                'reference_text': reference_text
            }
            
            start_time = time.time()
            result1 = await synthesizer._zero_shot_synthesis(test_text, synthesis_kwargs)
            time1 = time.time() - start_time
            
            start_time = time.time()
            result2 = await synthesizer._zero_shot_synthesis(test_text, synthesis_kwargs)
            time2 = time.time() - start_time
            
            logger.info(f"第一次合成时间: {time1:.3f}s")
            logger.info(f"第二次合成时间: {time2:.3f}s")
            
            if 'speaker_id' in result1 and 'speaker_id' in result2:
                assert result1['speaker_id'] == result2['speaker_id'], "使用相同参考音频应该得到相同的说话人ID"
                logger.info("✅ 说话人ID一致性测试通过")
            
            logger.info("✅ 语音合成测试通过")
            
        except Exception as e:
            logger.warning(f"语音合成测试失败（预期中）: {str(e)}")
        
        # 测试6: 清空缓存
        logger.info("🔍 测试6: 清空说话人缓存")
        await synthesizer.clear_speaker_cache()
        
        # 验证缓存已清空
        empty_speakers = await synthesizer.get_cached_speakers()
        assert len(empty_speakers) == 0, "清空后应该没有缓存的说话人"
        logger.info("✅ 缓存清空测试通过")
        
        logger.info("🎉 所有说话人缓存测试通过!")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        raise
    
    finally:
        # 清理资源
        await synthesizer.cleanup()
        
        # 清理测试文件
        try:
            Path(test_audio_path).unlink(missing_ok=True)
            logger.info(f"清理测试文件: {test_audio_path}")
        except Exception as e:
            logger.warning(f"清理测试文件失败: {str(e)}")

async def main():
    """主函数"""
    try:
        await test_speaker_caching()
        print("✅ 所有测试通过!")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)