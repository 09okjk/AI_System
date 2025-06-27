# CosyVoice 语音合成音色一致性改进说明

## 问题背景

在原始实现中，CosyVoice 语音合成存在以下问题：
1. **音色语调不稳定** - 每次合成都重新分析参考音频，导致音色不一致
2. **缺少说话人缓存机制** - 没有利用 CosyVoice 的说话人缓存功能
3. **参考音频重复处理** - 每次合成都重新加载和处理相同的参考音频
4. **无说话人ID管理** - 无法重用相同的音色设置

## 解决方案

### 1. 说话人缓存机制

新实现添加了完整的说话人缓存系统：

```python
# 在 CosyVoiceSynthesizer 类中添加
self.speaker_cache = {}  # 缓存说话人信息
self.default_speaker_id = None  # 默认说话人ID
```

### 2. 说话人ID生成

基于参考音频路径和文本生成唯一的说话人ID：

```python
def _generate_speaker_id(self, reference_audio_path: str, reference_text: str) -> str:
    """生成说话人ID"""
    import hashlib
    content = f"{reference_audio_path}:{reference_text}"
    return f"spk_{hashlib.md5(content.encode()).hexdigest()[:8]}"
```

### 3. 音频数据缓存

预处理和缓存音频数据，避免重复加载：

```python
# 缓存包含预处理的音频数据
self.speaker_cache[speaker_id] = {
    'path': reference_audio_path,
    'text': reference_text,
    'processed_path': processed_audio_path,
    'audio_data': reference_audio,  # 缓存加载的音频数据
    'cached_time': time.time()
}
```

### 4. 一致性合成

所有合成方法都使用缓存的音频数据：

```python
# 获取缓存的音频数据
cache_info = self.speaker_cache[speaker_id]
reference_audio = cache_info['audio_data']

# 使用缓存数据进行合成
for i, result in enumerate(self.model.inference_zero_shot(
    text, reference_text, reference_audio, stream=stream
)):
    output_audio = result['tts_speech']
```

## 功能特性

### 📌 核心改进

1. **音色一致性保证** - 相同参考音频和文本的多次合成保持完全一致的音色
2. **性能优化** - 音频预处理只执行一次，后续合成直接使用缓存数据
3. **自动格式转换** - 支持非WAV格式音频的自动转换和缓存
4. **智能缓存管理** - 基于音频路径和文本的智能缓存策略
5. **资源清理** - 完整的缓存清理和临时文件管理

### 🔧 API 增强

#### 新增方法

- `get_speaker_info()` - 获取说话人缓存信息
- `clear_speaker_cache()` - 清理所有说话人缓存
- `cleanup()` - 资源清理
- `_get_or_create_speaker()` - 说话人缓存管理
- `_remove_speaker()` - 移除特定说话人缓存

#### 增强的合成方法

所有合成方法（`_zero_shot_synthesis`, `_cross_lingual_synthesis`, `_instruct_synthesis`）都支持：
- 说话人ID管理
- 音频数据缓存
- 一致性保证
- 详细日志记录

## 使用示例

### 基本使用

```python
from src.speech import CosyVoiceSynthesizer

# 配置
config = {
    'model_dir': '/path/to/cosyvoice/model',
    'reference_audio': 'reference_audio/speaker.wav',
    'reference_text': '这是参考音频的文本',
}

# 初始化
synthesizer = CosyVoiceSynthesizer(config)
await synthesizer.initialize()

# 多次合成（保持音色一致）
texts = [
    "今天天气很好。",
    "人工智能发展迅速。", 
    "音乐让人心情愉悦。"
]

for text in texts:
    result = await synthesizer.synthesize(
        text=text,
        synthesis_mode="zero_shot",
        reference_audio="reference_audio/speaker.wav",
        reference_text="这是参考音频的文本"
    )
    # 所有结果将使用相同的说话人ID，保证音色一致
    print(f"说话人ID: {result.get('speaker_id')}")
```

### 高级使用

```python
# 手动指定说话人ID
result = await synthesizer.synthesize(
    text="测试文本",
    synthesis_mode="zero_shot",
    speaker_id="my_custom_speaker_01",  # 自定义说话人ID
    reference_audio="reference_audio/speaker.wav",
    reference_text="参考文本"
)

# 获取缓存信息
cache_info = await synthesizer.get_speaker_info()
print(f"缓存的说话人数量: {cache_info['total_speakers']}")
print(f"默认说话人: {cache_info['default_speaker']}")

# 清理缓存
await synthesizer.clear_speaker_cache()
```

### 跨语言和指令式合成

```python
# 跨语言合成（保持音色一致）
result = await synthesizer.synthesize(
    text="Hello, this is a cross-lingual test.",
    synthesis_mode="cross_lingual",
    reference_audio="reference_audio/chinese_speaker.wav"
)

# 指令式合成（保持音色一致）
result = await synthesizer.synthesize(
    text="这是一段需要情感朗读的文本。",
    synthesis_mode="instruct",
    instruction="用温和感人的语调朗读",
    reference_audio="reference_audio/speaker.wav"
)
```

## 技术细节

### 说话人ID生成规则

- 基于 `reference_audio_path` + `reference_text` 的MD5哈希
- 格式：`spk_{8位哈希值}`
- 相同参数生成相同ID，确保一致性

### 缓存策略

1. **智能缓存** - 检查参数变化，只在必要时重新缓存
2. **格式转换** - 自动处理非WAV格式音频
3. **内存管理** - 缓存预处理的音频数据，避免重复计算
4. **清理机制** - 自动清理临时文件和过期缓存

### 错误处理

- 音频文件不存在检查
- 格式转换失败处理
- 模型加载错误处理
- 缓存操作异常处理

## 测试验证

运行测试脚本验证功能：

```bash
python test_cosyvoice_consistency.py
```

测试包括：
- 说话人ID生成一致性
- 缓存机制正确性
- 边界条件处理
- 资源清理验证

## 性能改进

1. **减少音频处理** - 缓存预处理结果，避免重复计算
2. **降低IO开销** - 音频数据内存缓存，减少文件读取
3. **提高合成速度** - 跳过重复的参考音频分析过程
4. **优化资源使用** - 智能缓存管理，及时清理无用数据

## 兼容性

- 完全向后兼容原有API
- 新功能为可选特性，不影响现有代码
- 支持所有CosyVoice合成模式
- 保持原有错误处理逻辑