# 参考音频文件夹

这个文件夹用于存放CosyVoice2语音合成所需的参考音频文件。

## 用途

CosyVoice2模型需要一个参考音频来模仿其声音特征。您可以在这里放置不同的音频文件，以便生成不同风格的合成语音。

## 格式要求

- 推荐使用WAV格式音频（16kHz或24kHz采样率，16位深度）
- 如果使用MP3或其他格式，系统会自动转换为WAV格式
- 音频长度建议在5-30秒之间，内容清晰、背景噪音小

## 使用方法

1. 将您的参考音频文件放在此文件夹中
2. 在配置文件中指定参考音频路径：

   ```bash
   COSYVOICE_REF_AUDIO=reference_audio/your_audio_file.wav
   ```

   或者在代码中直接指定：

   ```python
   result = await synthesize_speech(
       "要合成的文本", 
       reference_audio="reference_audio/your_audio_file.wav",
       reference_text="参考音频中所说的文本"
   )
   ```

## 默认参考音频

默认参考音频文件: `default_reference.wav` ,
默认参考文本: "这是一段用于语音合成的参考音频"
