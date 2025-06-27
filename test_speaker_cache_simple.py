#!/usr/bin/env python3
"""
简单测试 CosyVoice 说话人缓存功能的核心逻辑
"""

import asyncio
import sys
import tempfile
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# 模拟 AsyncCache
class AsyncCache:
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expire_time = self.cache[key]
            if time.time() < expire_time:
                return value
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)
    
    async def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        self.cache.clear()

def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# 模拟 CosyVoice 说话人缓存功能
class MockCosyVoiceSynthesizer:
    def __init__(self):
        self.speaker_cache = AsyncCache(default_ttl=7200)
        self.model = None  # 模拟模式
    
    async def _generate_speaker_id(self, reference_audio_path: str, reference_text: str) -> str:
        try:
            with open(reference_audio_path, 'rb') as f:
                audio_content = f.read()
            
            combined_content = audio_content + reference_text.encode('utf-8')
            speaker_id = calculate_sha256(combined_content)[:16]
            
            print(f"生成说话人ID: {speaker_id} (音频: {Path(reference_audio_path).name}, 文本: {reference_text[:20]}...)")
            return speaker_id
            
        except Exception as e:
            print(f"生成说话人ID失败: {str(e)}")
            fallback_content = f"{reference_audio_path}:{reference_text}".encode('utf-8')
            return calculate_sha256(fallback_content)[:16]
    
    async def _cache_speaker(self, speaker_id: str, reference_audio_path: str, reference_text: str) -> bool:
        try:
            cached_speaker = await self.speaker_cache.get(speaker_id)
            if cached_speaker is not None:
                print(f"说话人 {speaker_id} 已存在缓存中")
                return True
            
            # 模拟缓存说话人信息
            speaker_info = {
                "speaker_id": speaker_id,
                "reference_audio_path": reference_audio_path,
                "reference_text": reference_text,
                "cached_at": time.time()
            }
            await self.speaker_cache.set(speaker_id, speaker_info)
            
            print(f"✅ 说话人 {speaker_id} 已成功缓存")
            return True
                
        except Exception as e:
            print(f"缓存说话人失败: {str(e)}")
            return False
    
    async def _get_cached_speaker(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        try:
            speaker_info = await self.speaker_cache.get(speaker_id)
            if speaker_info:
                print(f"找到缓存的说话人: {speaker_id}")
                return speaker_info
            else:
                print(f"未找到缓存的说话人: {speaker_id}")
                return None
        except Exception as e:
            print(f"获取缓存说话人失败: {str(e)}")
            return None
    
    async def clear_speaker_cache(self) -> None:
        try:
            await self.speaker_cache.clear()
            print("✅ 说话人缓存已清空")
        except Exception as e:
            print(f"清空说话人缓存失败: {str(e)}")
    
    async def get_cached_speakers(self) -> list:
        try:
            speakers = []
            for key, (value, _) in self.speaker_cache.cache.items():
                if isinstance(value, dict) and 'speaker_id' in value:
                    speaker_meta = {
                        "speaker_id": value["speaker_id"],
                        "reference_audio_path": value["reference_audio_path"],
                        "reference_text": value["reference_text"],
                        "cached_at": value["cached_at"]
                    }
                    speakers.append(speaker_meta)
            
            print(f"获取到 {len(speakers)} 个缓存的说话人")
            return speakers
            
        except Exception as e:
            print(f"获取缓存说话人列表失败: {str(e)}")
            return []

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
    
    print(f"创建测试音频文件: {test_file}")
    return str(test_file)

async def test_speaker_caching():
    """测试说话人缓存功能"""
    print("🧪 开始测试说话人缓存功能")
    
    # 创建合成器实例
    synthesizer = MockCosyVoiceSynthesizer()
    
    try:
        # 创建测试音频文件
        test_audio_path = create_test_wav_file()
        reference_text = "这是参考音频的文本内容"
        
        # 测试1: 生成说话人ID
        print("\n🔍 测试1: 生成说话人ID")
        speaker_id1 = await synthesizer._generate_speaker_id(test_audio_path, reference_text)
        print(f"说话人ID: {speaker_id1}")
        
        # 使用相同参数再次生成，应该得到相同的ID
        speaker_id2 = await synthesizer._generate_speaker_id(test_audio_path, reference_text)
        assert speaker_id1 == speaker_id2, "相同参数应该生成相同的说话人ID"
        print("✅ 说话人ID生成一致性测试通过")
        
        # 测试2: 缓存说话人
        print("\n🔍 测试2: 缓存说话人")
        cache_result = await synthesizer._cache_speaker(speaker_id1, test_audio_path, reference_text)
        print(f"缓存结果: {cache_result}")
        
        # 测试3: 获取缓存的说话人
        print("\n🔍 测试3: 获取缓存的说话人")
        cached_speaker = await synthesizer._get_cached_speaker(speaker_id1)
        if cached_speaker:
            print(f"✅ 成功获取缓存的说话人: {cached_speaker['speaker_id']}")
        else:
            print("⚠️ 未找到缓存的说话人")
        
        # 测试4: 获取所有缓存的说话人
        print("\n🔍 测试4: 获取所有缓存的说话人")
        all_speakers = await synthesizer.get_cached_speakers()
        print(f"缓存的说话人数量: {len(all_speakers)}")
        for speaker in all_speakers:
            print(f"  - {speaker['speaker_id']}: {speaker['reference_text'][:30]}...")
        
        # 测试5: 缓存多个不同的说话人
        print("\n🔍 测试5: 缓存多个不同的说话人")
        
        # 创建第二个测试文件
        test_audio_path2 = create_test_wav_file()
        reference_text2 = "这是第二个参考音频的文本内容"
        
        speaker_id3 = await synthesizer._generate_speaker_id(test_audio_path2, reference_text2)
        await synthesizer._cache_speaker(speaker_id3, test_audio_path2, reference_text2)
        
        # 验证现在有两个不同的说话人
        all_speakers = await synthesizer.get_cached_speakers()
        assert len(all_speakers) == 2, "应该有两个不同的说话人"
        print("✅ 多说话人缓存测试通过")
        
        # 测试6: 清空缓存
        print("\n🔍 测试6: 清空说话人缓存")
        await synthesizer.clear_speaker_cache()
        
        # 验证缓存已清空
        empty_speakers = await synthesizer.get_cached_speakers()
        assert len(empty_speakers) == 0, "清空后应该没有缓存的说话人"
        print("✅ 缓存清空测试通过")
        
        print("\n🎉 所有说话人缓存测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        raise
    
    finally:
        # 清理测试文件
        try:
            if 'test_audio_path' in locals():
                Path(test_audio_path).unlink(missing_ok=True)
                print(f"清理测试文件: {test_audio_path}")
            if 'test_audio_path2' in locals():
                Path(test_audio_path2).unlink(missing_ok=True)
                print(f"清理测试文件: {test_audio_path2}")
        except Exception as e:
            print(f"清理测试文件失败: {str(e)}")

async def main():
    """主函数"""
    try:
        await test_speaker_caching()
        print("\n✅ 所有测试通过!")
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)