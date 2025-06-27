# CosyVoice 说话人缓存功能使用指南

## 概述

CosyVoice 说话人缓存功能解决了语音合成中音色不一致的问题，通过缓存说话人特征，确保使用相同参考音频进行合成时能获得一致的音色和语调。

## 主要特性

### 1. 自动说话人识别
- 基于参考音频内容和参考文本自动生成唯一的说话人ID
- 相同的参考音频+文本组合始终产生相同的说话人ID
- 使用SHA256哈希确保ID的唯一性和一致性

### 2. 智能缓存机制
- 首次使用时自动缓存说话人特征到CosyVoice模型中
- 后续合成复用缓存的说话人信息，提升性能
- 支持多个不同说话人的同时缓存

### 3. 兼容性设计
- 完全兼容现有的语音合成API
- 无需修改现有代码即可享受缓存带来的优势
- 当缓存失败时自动回退到传统合成方式

## 使用示例

### 基本语音合成（自动缓存）

```python
from src.speech import speech_processor

# 初始化语音处理器
await speech_processor.initialize()

# 第一次合成 - 会自动缓存说话人
result1 = await speech_processor.synthesize(
    text="你好，这是第一段测试语音",
    synthesis_mode="zero_shot",
    reference_audio="reference_audio/speaker1.wav",
    reference_text="这是说话人1的参考语音内容"
)

# 第二次合成 - 使用缓存的说话人，音色保持一致
result2 = await speech_processor.synthesize(
    text="这是第二段测试语音，应该和第一段音色一致",
    synthesis_mode="zero_shot",
    reference_audio="reference_audio/speaker1.wav",
    reference_text="这是说话人1的参考语音内容"
)

print(f"说话人ID一致: {result1.get('speaker_id') == result2.get('speaker_id')}")
```

### 指令式合成（同样支持缓存）

```python
# 指令式合成也会自动缓存说话人
result = await speech_processor.synthesize(
    text="请用温和的语调朗读这段文字",
    synthesis_mode="instruct",
    instruction="用温和甜美的女声朗读",
    reference_audio="reference_audio/female_voice.wav"
)
```

### 说话人管理

```python
# 查看所有缓存的说话人
cached_speakers = await speech_processor.get_cached_speakers()
print(f"当前缓存了 {len(cached_speakers['cosyvoice'])} 个说话人")

for speaker in cached_speakers['cosyvoice']:
    print(f"说话人ID: {speaker['speaker_id']}")
    print(f"参考音频: {speaker['reference_audio_path']}")
    print(f"参考文本: {speaker['reference_text']}")
    print(f"缓存时间: {speaker['cached_at']}")

# 清空指定合成器的说话人缓存
await speech_processor.clear_speaker_cache('cosyvoice')

# 或清空所有合成器的缓存
await speech_processor.clear_speaker_cache()
```

## 技术实现细节

### 说话人ID生成算法

```python
# 说话人ID基于参考音频内容和参考文本生成
audio_content = read_audio_file(reference_audio_path)
combined_content = audio_content + reference_text.encode('utf-8')
speaker_id = sha256(combined_content).hexdigest()[:16]
```

### 缓存策略

1. **优先使用CosyVoice原生缓存**：调用`model.add_zero_shot_spk()`
2. **本地缓存作为fallback**：当原生缓存不可用时使用AsyncCache
3. **智能过期管理**：默认2小时TTL，自动清理过期缓存

## 性能优化建议

### 1. 预热常用说话人

```python
# 在应用启动时预热常用的说话人
common_speakers = [
    {"audio": "speaker1.wav", "text": "说话人1的参考文本"},
    {"audio": "speaker2.wav", "text": "说话人2的参考文本"},
]

for speaker in common_speakers:
    await speech_processor.synthesize(
        text="预热测试",  # 短文本用于预热
        reference_audio=speaker["audio"],
        reference_text=speaker["text"]
    )
```

### 2. 监控缓存效果

```python
import time

start_time = time.time()
result1 = await speech_processor.synthesize(text="测试1", ...)
first_synthesis_time = time.time() - start_time

start_time = time.time()  
result2 = await speech_processor.synthesize(text="测试2", ...)
second_synthesis_time = time.time() - start_time

print(f"首次合成: {first_synthesis_time:.3f}s")
print(f"缓存合成: {second_synthesis_time:.3f}s")
print(f"性能提升: {(first_synthesis_time/second_synthesis_time - 1)*100:.1f}%")
```

## 最佳实践

### 参考文本处理

为获得最佳效果，建议：

1. **提供准确的参考文本**：确保参考文本与参考音频内容匹配
2. **使用描述性文本**：如果不知道确切内容，可以提供描述性文本
3. **保持文本一致**：相同说话人使用相同的参考文本

```python
# 好的做法
await speech_processor.synthesize(
    text="今天天气真不错",
    reference_audio="speaker.wav",
    reference_text="这是一段清晰自然的中文语音样本"  # 描述性但一致
)

# 更好的做法（如果知道确切内容）
await speech_processor.synthesize(
    text="今天天气真不错",
    reference_audio="speaker.wav", 
    reference_text="大家好，我是主播小王"  # 确切的音频内容
)
```

## 故障排除

### 常见问题

1. **说话人ID不一致**
   - 检查参考音频路径和参考文本是否完全一致
   - 确保音频文件内容没有变化

2. **缓存不生效**
   - 检查CosyVoice模型是否支持`add_zero_shot_spk`方法
   - 查看日志确认缓存操作是否成功

3. **音色仍然不一致**
   - 确保使用的是相同的synthesis_mode
   - 检查参考音频质量和时长（建议5-30秒）

### 调试模式

启用调试日志查看详细的缓存操作：

```python
import logging
logging.getLogger('src.speech').setLevel(logging.DEBUG)
```

## 版本兼容性

- 要求 CosyVoice2-0.5B 或更高版本
- 向后兼容所有现有的语音合成API
- 如果模型不支持原生缓存，会自动使用本地缓存fallback

## 注意事项

1. **内存使用**：每个缓存的说话人会占用一定内存，建议定期清理不需要的缓存
2. **文件依赖**：说话人缓存依赖于参考音频文件，删除文件可能影响缓存效果
3. **并发安全**：缓存操作是线程安全的，支持并发访问

---

更多详细信息请参考代码中的文档字符串和示例。