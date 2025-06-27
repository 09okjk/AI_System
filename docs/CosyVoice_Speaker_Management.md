# CosyVoice Speaker Consistency Features

This document describes the new speaker ID caching mechanism implemented to resolve voice consistency issues in CosyVoice synthesis.

## Problem Solved

Before this implementation, CosyVoice had the following issues:
1. **Inconsistent voice tone**: Each synthesis call processed reference audio from scratch
2. **Different voice per segment**: Each synthesized speech segment had different voice characteristics  
3. **No speaker caching**: Reference audio was reprocessed every time, causing overhead

## Solution Overview

The new speaker management system provides:
- **Speaker ID caching** with automatic and manual management
- **Voice consistency** across multiple synthesis calls
- **Performance optimization** by reusing processed speaker data
- **Persistent storage** of speaker information across sessions

## Configuration

Add these environment variables to your `.env` file:

```bash
# Reference audio configuration
COSYVOICE_REF_AUDIO=reference_audio/your_audio.wav
COSYVOICE_REF_TEXT=参考音频中所说的文本

# Speaker cache directory  
COSYVOICE_SPEAKER_CACHE_DIR=/tmp/ai_system_speakers
```

## API Usage

### Basic Synthesis with Speaker Consistency

```python
from src.speech import speech_processor

# Initialize the processor
await speech_processor.initialize()

# Method 1: Use default speaker (automatically created from reference audio)
result = await speech_processor.synthesize(
    text="要合成的文本",
    synthesis_mode="zero_shot"
)

# Method 2: Specify a speaker ID for consistent voice
result = await speech_processor.synthesize(
    text="要合成的文本", 
    synthesis_mode="zero_shot",
    speaker_id="my_custom_speaker"
)

# Method 3: Create speaker automatically from new reference audio
result = await speech_processor.synthesize(
    text="要合成的文本",
    synthesis_mode="zero_shot", 
    reference_audio="path/to/new/audio.wav",
    reference_text="新音频对应的文本"
)
```

### Speaker Management

```python
# Get the synthesizer instance
synthesizer = speech_processor.synthesizers['cosyvoice']

# Add a new speaker manually
success = await synthesizer.add_speaker(
    speaker_id="custom_voice",
    reference_audio_path="path/to/audio.wav", 
    reference_text="音频对应的文本"
)

# List all cached speakers
speakers = synthesizer.list_cached_speakers()
print(f"Available speakers: {speakers}")

# Get speaker information
info = synthesizer.get_speaker_info("custom_voice")
if info:
    print(f"Speaker: {info['speaker_id']}")
    print(f"Text: {info['reference_text']}")
    print(f"Created: {info['created_at']}")

# Remove a speaker
success = await synthesizer.remove_speaker("custom_voice")
```

### Voice Consistency Validation

```python
# Test voice consistency with multiple texts
test_texts = [
    "第一段测试文本",
    "第二段测试文本", 
    "第三段测试文本"
]

validation_result = await synthesizer.validate_speaker_consistency(
    speaker_id="custom_voice",
    test_texts=test_texts
)

if validation_result["success"]:
    print(f"Consistency test passed for {validation_result['test_count']} texts")
    for result in validation_result["results"]:
        print(f"Text: {result['text'][:20]}... Duration: {result['duration']:.2f}s")
else:
    print(f"Consistency test failed: {validation_result['error']}")
```

## Advanced Features

### Automatic Speaker ID Generation

When reference audio is provided without a speaker_id, the system automatically generates one based on the audio path and text:

```python
# This creates a speaker with ID based on content hash
result = await speech_processor.synthesize(
    text="测试文本",
    reference_audio="new_voice.wav",
    reference_text="新的参考文本"
)

# The speaker_id will be something like "3aed70d937ee"
print(f"Auto-generated speaker: {result['speaker_id']}")
```

### Speaker Cache Persistence

Speaker information is automatically saved to disk and reloaded when the application restarts:

```bash
# View cached speakers on disk
ls $COSYVOICE_SPEAKER_CACHE_DIR/
# Output: default_speaker.json  custom_voice.json  3aed70d937ee.json

# View speaker info
cat $COSYVOICE_SPEAKER_CACHE_DIR/default_speaker.json
```

### Multiple Synthesis Modes

Speaker caching works with all CosyVoice synthesis modes:

```python
# Zero-shot synthesis with speaker consistency
await synthesizer.synthesize(
    text="文本", 
    synthesis_mode="zero_shot",
    speaker_id="my_speaker"
)

# Cross-lingual synthesis with speaker consistency  
await synthesizer.synthesize(
    text="Text with [laughter] markers",
    synthesis_mode="cross_lingual", 
    speaker_id="my_speaker"
)

# Instruct synthesis with speaker consistency
await synthesizer.synthesize(
    text="文本",
    synthesis_mode="instruct",
    instruction="用温和的声音说", 
    speaker_id="my_speaker"
)
```

## Error Handling

The system gracefully handles various error conditions:

```python
# Non-existent speaker falls back to traditional synthesis
result = await synthesizer.synthesize(
    text="文本",
    speaker_id="non_existent_speaker",
    reference_audio="fallback.wav",  # Will be used as fallback
    reference_text="备用参考文本"
)

# Missing reference audio
try:
    await synthesizer.add_speaker(
        speaker_id="test",
        reference_audio_path="missing.wav",
        reference_text="文本"
    )
except Exception as e:
    print(f"Error: {e}")  # "参考音频文件不存在: missing.wav"
```

## Performance Benefits

The speaker caching system provides significant performance improvements:

1. **Reduced Processing Time**: Reference audio is processed once and reused
2. **Memory Efficiency**: Cached speaker data is smaller than raw audio
3. **Consistent Quality**: Same speaker always produces identical voice characteristics
4. **Scalability**: Supports multiple speakers without increasing processing overhead

## Migration Guide

Existing code will continue to work without changes:

```python
# This still works exactly as before
result = await speech_processor.synthesize("文本")

# But now also provides consistent voice if you use the same reference audio
```

To take advantage of speaker consistency, simply add `speaker_id` parameters:

```python
# Before (inconsistent voice)
for text in texts:
    result = await synthesize(text, reference_audio="voice.wav")

# After (consistent voice)  
for text in texts:
    result = await synthesize(text, speaker_id="my_voice")
```

## Best Practices

1. **Use meaningful speaker IDs**: Choose descriptive names like "female_chinese_calm" instead of random strings

2. **Cache frequently used speakers**: Add important speakers manually instead of relying on auto-generation

3. **Validate consistency**: Use the validation function to ensure voice quality meets your requirements

4. **Monitor cache size**: Periodically clean up unused speakers to save disk space

5. **Backup speaker cache**: Include the speaker cache directory in your backup strategy for production systems

## Troubleshooting

### Speaker not found
```python
speakers = synthesizer.list_cached_speakers()
print(f"Available speakers: {speakers}")
```

### Cache directory issues
```bash
# Check if directory exists and is writable
ls -la $COSYVOICE_SPEAKER_CACHE_DIR
```

### Voice quality problems
```python
# Test speaker consistency
validation = await synthesizer.validate_speaker_consistency(
    speaker_id="problematic_speaker",
    test_texts=["测试文本1", "测试文本2"]
)
```

### Performance issues
```python
# Check cache statistics
print(f"Cached speakers: {len(synthesizer.speaker_cache)}")
print(f"Cache directory: {synthesizer.speaker_cache_dir}")
```