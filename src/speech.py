"""
语音处理器 - 修复音频输出问题
主要修复CosyVoice音频输出异常的问题
"""

import asyncio
import base64
import tempfile
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np
import torch

from .logger import get_logger, log_speech_operation
from .models import SpeechRecognitionResponse, SpeechSynthesisResponse, AudioFormat
from .utils import generate_response_id

logger = get_logger(__name__)

# 添加音频格式转换函数
def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None, sample_rate: int = 16000) -> str:
    """
    将任意音频格式转换为WAV格式
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出WAV文件路径，如果为None则生成一个临时文件
        sample_rate: 输出音频的采样率
        
    Returns:
        WAV文件的路径
    """
    try:
        import librosa
        import soundfile as sf
        
        # 如果未指定输出路径，创建临时文件
        if not output_path:
            temp_dir = Path(tempfile.gettempdir()) / "ai_system_audio"
            temp_dir.mkdir(exist_ok=True)
            output_path = str(temp_dir / f"{uuid.uuid4()}.wav")
        
        # 加载音频文件
        logger.info(f"正在转换音频: {input_path} -> {output_path}")
        y, sr = librosa.load(input_path, sr=sample_rate)
        
        # 保存为WAV格式
        sf.write(output_path, y, sample_rate, subtype='PCM_16')
        logger.info(f"音频转换完成: {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"音频格式转换失败: {str(e)}")
        raise

class SpeechRecognizer:
    """语音识别器基类"""    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.model = None
        self.device = config.get('device', 'cpu')

    async def initialize(self) -> bool:
        """初始化识别器"""
        try:
            await self._setup()
            self.is_initialized = True
            logger.info(f"✅ {self.__class__.__name__} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ {self.__class__.__name__} 初始化失败: {str(e)}")
            return False

    async def _setup(self):
        """设置识别器（由子类实现）"""
        pass

    async def recognize(self, 
                        audio_data: bytes, 
                        language: str = "zh-CN",
                        **kwargs) -> Dict[str, Any]:
        """识别语音（由子类实现）"""
        raise NotImplementedError

class SenseVoiceRecognizer(SpeechRecognizer):
    """SenseVoice 语音识别器 - 基于官方实现优化"""
    
    async def _setup(self):
        """设置 SenseVoice - 根据官方最佳实践"""
        try:
            from funasr import AutoModel
            
            # 根据官方推荐的配置
            model_config = {
                "model": "iic/SenseVoiceSmall",  # 使用官方推荐的模型
                "vad_model": "fsmn-vad",
                "punc_model": "ct-punc", 
                "device": self.device,
                "hub": "ms",  # 使用 modelscope
                "ncpu": self.config.get('ncpu', 4),
                "batch_size": self.config.get('batch_size', 1)
            }
            
            # 支持 VAD 的配置
            vad_kwargs = {
                "max_single_segment_time": self.config.get('max_single_segment_time', 60000),  # 60秒
                "batch_size_s": self.config.get('batch_size_s', 300),  # 动态batch，总音频时长300秒
                "batch_size_threshold_s": self.config.get('batch_size_threshold_s', 60)  # 阈值60秒
            }
            
            # 初始化模型
            self.model = AutoModel(
                model=model_config["model"],
                vad_model=model_config["vad_model"],
                vad_kwargs=vad_kwargs,
                punc_model=model_config["punc_model"],
                device=model_config["device"],
                hub=model_config["hub"],
                ncpu=model_config["ncpu"],
                batch_size=model_config["batch_size"]
            )
            
            # 存储模型配置用于后续使用
            self.model_config = model_config
            self.vad_kwargs = vad_kwargs
            
            logger.info(f"✅ SenseVoice 模型初始化成功 - 设备: {self.device}")
            
        except ImportError:
            logger.warning("⚠️ FunASR 未安装，使用模拟识别器")
            self.model = None
        except Exception as e:
            logger.error(f"❌ SenseVoice 初始化失败: {str(e)}")
            raise
    
    async def recognize(self, 
                       audio_data: bytes, 
                       language: str = "zh-CN",
                       **kwargs) -> Dict[str, Any]:
        """SenseVoice 语音识别 - 优化版本"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # 模拟识别结果
                await asyncio.sleep(0.5)
                
                processing_time = time.time() - start_time
                
                return {
                    "text": "这是一个模拟的语音识别结果",
                    "language": language,
                    "confidence": 0.95,
                    "processing_time": processing_time,
                    "model_used": "mock_sensevoice",
                    "segments": []
                }
            
            # 保存音频到临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # 使用官方推荐的参数执行识别
                generation_kwargs = {
                    "hotword": kwargs.get('hotword', ''),  # 热词
                    "batch_size_s": self.vad_kwargs.get('batch_size_s', 300),
                    "batch_size_threshold_s": self.vad_kwargs.get('batch_size_threshold_s', 60)
                }
                
                # 添加自定义参数
                generation_kwargs.update(kwargs)
                
                # 执行识别
                result = self.model.generate(
                    input=temp_audio_path,
                    **generation_kwargs
                )
                
                processing_time = time.time() - start_time
                
                # 处理识别结果
                if result and len(result) > 0:
                    # SenseVoice 返回格式通常是列表
                    first_result = result[0] if isinstance(result, list) else result
                    
                    # 提取文本
                    text = first_result.get('text', '') if isinstance(first_result, dict) else str(first_result)

                    # 提取置信度（如果可用）
                    confidence = first_result.get('confidence', 0.9) if isinstance(first_result, dict) else 0.9
                    
                    # 提取时间戳信息（如果可用）
                    segments = first_result.get('timestamps', []) if isinstance(first_result, dict) else []
                    
                else:
                    text = ""
                    confidence = 0.0
                    segments = []
                
                log_speech_operation(
                    logger, "recognition", "sensevoice", 
                    len(audio_data), len(text), processing_time, 
                    True, language
                )
                
                return {
                    "text": text,
                    "language": language,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "model_used": "sensevoice",
                    "segments": segments,
                    "raw_result": result  # 保留原始结果以便调试
                }
                
            finally:
                # 清理临时文件
                Path(temp_audio_path).unlink(missing_ok=True)
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "recognition", "sensevoice", 
                len(audio_data), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"SenseVoice 语音识别失败: {error_msg}")

class SpeechSynthesizer:
    """语音合成器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化合成器"""
        try:
            await self._setup()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ 语音合成器初始化失败: {str(e)}")
            return False
    
    async def _setup(self):
        """设置合成器（由子类实现）"""
        pass
    
    async def synthesize(self, 
                        text: str, 
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        **kwargs) -> Dict[str, Any]:
        """合成语音（由子类实现）"""
        raise NotImplementedError

class CosyVoiceSynthesizer(SpeechSynthesizer):
    """CosyVoice 语音合成器 - 修复音频输出问题"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Speaker管理
        self.speakers_cache = {}  # 缓存已加载的speaker
        self.default_speaker_id = None  # 默认speaker ID
        self.speaker_info_file = None  # speaker信息文件路径
        
    async def _setup(self):
        """设置 CosyVoice - 根据官方最佳实践"""
        try:
            # 设置 CosyVoice 路径 - 修复路径配置
            import sys
            
            # 添加 CosyVoice 主目录
            cosyvoice_path = self.config.get('cosyvoice_path', 'tools/CosyVoice')
            if cosyvoice_path not in sys.path:
                sys.path.append(cosyvoice_path)
                
            # 添加 Matcha-TTS 路径（根据官方文档）
            matcha_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS')
            if os.path.exists(matcha_path) and matcha_path not in sys.path:
                sys.path.append(matcha_path)
                
            logger.info(f"CosyVoice 路径: {cosyvoice_path}")
            logger.info(f"Matcha-TTS 路径: {matcha_path}")
            logger.info(f"当前 Python 路径: {sys.path[-2:]}")  # 显示最后添加的路径
            
            # 导入 CosyVoice - 添加更详细的错误信息
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                from cosyvoice.utils.file_utils import load_wav
                import torchaudio
                logger.info("✅ CosyVoice 模块导入成功")
            except ImportError as import_err:
                logger.error(f"❌ CosyVoice 模块导入失败: {import_err}")
                logger.error(f"检查路径: {[p for p in sys.path if 'CosyVoice' in p]}")
                raise
            
            # 模型配置
            model_dir = self.config.get('model_dir', '/home/rh/Program/MCP_Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B')
            logger.info(f"CosyVoice 模型目录: {model_dir}")
            logger.info(f"CosyVoice 模型目录存在: {Path(model_dir).exists()}")

            # 检查模型目录
            if not Path(model_dir).exists():
                raise FileNotFoundError(f"CosyVoice 模型目录不存在: {model_dir}")
            
            # 初始化模型
            self.model = CosyVoice2(
                model_dir=model_dir,
                load_jit=self.config.get('load_jit', False),
                load_trt=self.config.get('load_trt', False),
                fp16=self.config.get('fp16', False)
            )
            
            # 保存工具函数
            self.load_wav = load_wav
            self.torchaudio = torchaudio
            
            # 设置speaker信息文件路径
            self.speaker_info_file = Path(model_dir) / "custom_speakers.json"
            
            # 加载已保存的speaker信息
            await self._load_saved_speakers()
            
            # 设置默认参考音频和speaker
            await self._setup_default_speaker()
            
            logger.info(f"✅ CosyVoice 模型初始化成功 - 模型路径: {model_dir}")
            logger.info(f"📢 默认speaker ID: {self.default_speaker_id}")
            logger.info(f"🎵 模型采样率: {self.model.sample_rate}")
            
        except ImportError as e:
            logger.warning(f"⚠️ CosyVoice 未正确安装: {str(e)}")
            self.model = None
        except Exception as e:
            logger.error(f"❌ CosyVoice 初始化失败: {str(e)}")
            raise
    
    async def _load_saved_speakers(self):
        """加载已保存的speaker信息"""
        try:
            if self.speaker_info_file and Path(self.speaker_info_file).exists():
                with open(self.speaker_info_file, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                    
                # 尝试加载保存的speaker信息到模型中
                if hasattr(self.model, 'load_spkinfo') and 'speakers' in speaker_data:
                    try:
                        # CosyVoice2 可能需要特定的speaker信息格式
                        self.model.load_spkinfo()
                        logger.info(f"✅ 已加载 {len(speaker_data['speakers'])} 个保存的speaker")
                    except Exception as e:
                        logger.warning(f"⚠️ 加载speaker信息失败: {str(e)}")
                
                self.speakers_cache = speaker_data.get('speakers', {})
                self.default_speaker_id = speaker_data.get('default_speaker_id')
                
                logger.info(f"📂 加载了 {len(self.speakers_cache)} 个缓存的speaker")
                
        except Exception as e:
            logger.warning(f"⚠️ 加载speaker信息文件失败: {str(e)}")
            self.speakers_cache = {}
    
    async def _setup_default_speaker(self):
        """设置默认speaker"""
        reference_audio_path = self.config.get('reference_audio', None)
        reference_text = self.config.get('reference_text', '参考音频文本')
        
        if reference_audio_path and Path(reference_audio_path).exists():
            try:
                # 如果没有默认speaker，创建一个
                if not self.default_speaker_id:
                    speaker_id = await self._add_speaker(
                        reference_text=reference_text,
                        reference_audio_path=reference_audio_path,
                        speaker_id='default_speaker'
                    )
                    self.default_speaker_id = speaker_id
                    logger.info(f"✅ 创建默认speaker: {speaker_id}")
                else:
                    logger.info(f"📢 使用已存在的默认speaker: {self.default_speaker_id}")
                    
            except Exception as e:
                logger.warning(f"⚠️ 设置默认speaker失败: {str(e)}")
                self.default_speaker_id = None
        else:
            logger.warning("⚠️ 未配置默认参考音频，需要在合成时提供")
    
    async def _add_speaker(self, reference_text: str, reference_audio_path: str, speaker_id: str = None) -> str:
        """添加新的speaker到模型中"""
        try:
            # 生成speaker ID
            if not speaker_id:
                speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
            
            # 检查speaker是否已存在
            if speaker_id in self.speakers_cache:
                logger.info(f"📢 Speaker {speaker_id} 已存在，跳过添加")
                return speaker_id
            
            # 确保音频格式正确
            processed_audio_path = await self._prepare_reference_audio(reference_audio_path)
            
            # 加载参考音频
            reference_audio = self.load_wav(processed_audio_path, self.model.sample_rate)
            
            # 添加zero-shot speaker到模型
            success = self.model.add_zero_shot_spk(reference_text, reference_audio, speaker_id)
            
            if success:
                # 缓存speaker信息
                self.speakers_cache[speaker_id] = {
                    'reference_text': reference_text,
                    'reference_audio_path': reference_audio_path,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # 保存speaker信息
                await self._save_speaker_info()
                
                logger.info(f"✅ 成功添加speaker: {speaker_id}")
                return speaker_id
            else:
                raise Exception(f"模型添加speaker失败: {speaker_id}")
                
        except Exception as e:
            logger.error(f"❌ 添加speaker失败: {str(e)}")
            raise
    
    async def _prepare_reference_audio(self, audio_path: str) -> str:
        """准备参考音频（确保格式正确）"""
        try:
            # 检查音频文件是否存在
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"参考音频文件不存在: {audio_path}")
            
            # 检查音频格式
            audio_ext = Path(audio_path).suffix.lower()
            
            # 如果不是WAV格式，进行转换
            if audio_ext != '.wav':
                logger.info(f"参考音频非WAV格式 ({audio_ext})，进行格式转换")
                converted_path = convert_audio_to_wav(audio_path, sample_rate=self.model.sample_rate)
                return converted_path
            else:
                # 验证采样率是否正确
                try:
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None)
                    if sr != self.model.sample_rate:
                        logger.info(f"参考音频采样率不匹配 ({sr} vs {self.model.sample_rate})，重新采样")
                        converted_path = convert_audio_to_wav(audio_path, sample_rate=self.model.sample_rate)
                        return converted_path
                except Exception:
                    pass  # 如果检查失败，直接使用原文件
                
                return audio_path
                
        except Exception as e:
            logger.error(f"❌ 准备参考音频失败: {str(e)}")
            raise
    
    async def _save_speaker_info(self):
        """保存speaker信息到文件"""
        try:
            if self.speaker_info_file:
                speaker_data = {
                    'speakers': self.speakers_cache,
                    'default_speaker_id': self.default_speaker_id,
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                # 确保目录存在
                self.speaker_info_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存到文件
                with open(self.speaker_info_file, 'w', encoding='utf-8') as f:
                    json.dump(speaker_data, f, ensure_ascii=False, indent=2)
                
                # 同时保存模型的speaker信息
                if hasattr(self.model, 'save_spkinfo'):
                    try:
                        self.model.save_spkinfo()
                    except Exception as e:
                        logger.warning(f"⚠️ 保存模型speaker信息失败: {str(e)}")
                
                logger.info(f"💾 Speaker信息已保存到: {self.speaker_info_file}")
                
        except Exception as e:
            logger.error(f"❌ 保存speaker信息失败: {str(e)}")
    
    async def synthesize(self, 
                        text: str, 
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        synthesis_mode: str = "zero_shot",
                        reference_audio: Optional[str] = None,
                        reference_text: Optional[str] = None,
                        speaker_id: Optional[str] = None,
                        instruction: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """CosyVoice 语音合成 - 修复音频输出问题"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # 模拟合成结果
                await asyncio.sleep(1.5)
                
                processing_time = time.time() - start_time
                
                # 创建模拟音频数据（静音）
                sample_rate = 22050
                duration = max(len(text) / 10, 1.0)  # 基于文本长度估算时长
                audio_samples = int(sample_rate * duration)
                mock_audio = b'\x00' * (audio_samples * 2)  # 16位PCM
                
                audio_base64 = base64.b64encode(mock_audio).decode('utf-8')
                
                return {
                    "audio_data": audio_base64,
                    "format": AudioFormat.WAV,
                    "duration": duration,
                    "processing_time": processing_time,
                    "model_used": "mock_cosyvoice",
                    "synthesis_mode": synthesis_mode
                }
            
            # 获取或创建speaker
            target_speaker_id = await self._get_or_create_speaker(
                speaker_id=speaker_id,
                reference_audio=reference_audio,
                reference_text=reference_text
            )
            
            # 根据合成模式选择不同的方法
            if synthesis_mode == "zero_shot":
                result = await self._zero_shot_synthesis_with_speaker(text, target_speaker_id, kwargs)
            elif synthesis_mode == "cross_lingual":
                result = await self._cross_lingual_synthesis_with_speaker(text, target_speaker_id, kwargs)
            elif synthesis_mode == "instruct":
                result = await self._instruct_synthesis_with_speaker(text, target_speaker_id, instruction, kwargs)
            else:
                # 默认使用零样本合成
                result = await self._zero_shot_synthesis_with_speaker(text, target_speaker_id, kwargs)
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["synthesis_mode"] = synthesis_mode
            result["speaker_id"] = target_speaker_id
            
            log_speech_operation(
                logger, "synthesis", "cosyvoice", 
                len(text), len(result["audio_data"]), processing_time, 
                True, language
            )
            
            logger.info(f"✅ 合成完成 - Speaker: {target_speaker_id}, 模式: {synthesis_mode}, 时长: {result.get('duration', 0):.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "synthesis", "cosyvoice", 
                len(text), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"CosyVoice 合成失败: {error_msg}")
    
    async def _get_or_create_speaker(self, 
                                speaker_id: Optional[str] = None,
                                reference_audio: Optional[str] = None,
                                reference_text: Optional[str] = None) -> str:
        """获取或创建speaker - 增强验证"""
        try:
            # 如果指定了speaker_id且存在，验证其完整性
            if speaker_id and speaker_id in self.speakers_cache:
                speaker_info = self.speakers_cache[speaker_id]
                
                # 验证speaker信息完整性
                if not speaker_info.get('reference_text'):
                    logger.warning(f"⚠️ Speaker {speaker_id} 缺少参考文本")
                if not speaker_info.get('reference_audio_path') or not Path(speaker_info['reference_audio_path']).exists():
                    logger.warning(f"⚠️ Speaker {speaker_id} 参考音频文件不存在")
                    
                logger.info(f"📢 使用已存在的speaker: {speaker_id}")
                return speaker_id
            
            # 如果提供了新的参考音频，创建新speaker
            if reference_audio and reference_text:
                new_speaker_id = speaker_id or f"custom_speaker_{uuid.uuid4().hex[:8]}"
                return await self._add_speaker(reference_text, reference_audio, new_speaker_id)
            
            # 使用默认speaker
            if self.default_speaker_id:
                logger.info(f"📢 使用默认speaker: {self.default_speaker_id}")
                return self.default_speaker_id
            
            # 如果没有任何可用speaker，抛出错误
            raise ValueError("没有可用的speaker，请提供reference_audio和reference_text")
            
        except Exception as e:
            logger.error(f"❌ 获取或创建speaker失败: {str(e)}")
            raise
    
    async def _zero_shot_synthesis_with_speaker(self, text: str, speaker_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定speaker进行零样本语音合成 - 修复版本"""
        try:
            # 获取speaker的参考信息
            if speaker_id not in self.speakers_cache:
                raise ValueError(f"Speaker {speaker_id} 不存在")
            
            speaker_info = self.speakers_cache[speaker_id]
            reference_text = speaker_info['reference_text']
            reference_audio_path = speaker_info['reference_audio_path']
            
            # 准备参考音频
            processed_audio_path = await self._prepare_reference_audio(reference_audio_path)
            reference_audio = self.load_wav(processed_audio_path, self.model.sample_rate)
            
            output_audio = None
            stream = kwargs.get('stream', False)
            
            logger.info(f"🎤 开始零样本合成 - Speaker: {speaker_id}, 文本: {text[:50]}...")
            logger.info(f"📝 使用参考文本: {reference_text[:30]}...")
            
            # 方法1：使用完整的参考音频和文本（推荐）
            try:
                for i, result in enumerate(self.model.inference_zero_shot(
                    text, reference_text, reference_audio, stream=stream
                )):
                    output_audio = result['tts_speech']
                    logger.info(f"🔊 生成音频张量形状: {output_audio.shape}")
                    logger.info(f"🔊 音频数据类型: {output_audio.dtype}")
                    logger.info(f"🔊 音频值范围: [{output_audio.min():.6f}, {output_audio.max():.6f}]")
                    if not stream:
                        break
            except Exception as e:
                logger.warning(f"⚠️ 方法1失败，尝试方法2: {str(e)}")
                
                # 方法2：尝试使用speaker_id但提供参考信息
                for i, result in enumerate(self.model.inference_zero_shot(
                    text, reference_text, reference_audio, zero_shot_spk_id=speaker_id, stream=stream
                )):
                    output_audio = result['tts_speech']
                    logger.info(f"🔊 生成音频张量形状: {output_audio.shape}")
                    if not stream:
                        break
            
            if output_audio is None:
                raise Exception("零样本合成失败，未生成音频")
            
            return await self._process_output_audio(output_audio)
            
        except Exception as e:
            logger.error(f"❌ 零样本合成失败: {str(e)}")
            raise
    
    async def _cross_lingual_synthesis_with_speaker(self, text: str, speaker_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定speaker进行跨语言语音合成"""
        try:
            # 获取speaker的参考音频
            if speaker_id not in self.speakers_cache:
                raise ValueError(f"Speaker {speaker_id} 不存在")
            
            speaker_info = self.speakers_cache[speaker_id]
            reference_audio_path = speaker_info['reference_audio_path']
            
            # 准备参考音频
            processed_audio_path = await self._prepare_reference_audio(reference_audio_path)
            reference_audio = self.load_wav(processed_audio_path, self.model.sample_rate)
            
            # 执行跨语言合成
            output_audio = None
            stream = kwargs.get('stream', False)
            
            for i, result in enumerate(self.model.inference_cross_lingual(
                text, reference_audio, stream=stream
            )):
                output_audio = result['tts_speech']
                if not stream:
                    break
            
            if output_audio is None:
                raise Exception("跨语言合成失败，未生成音频")
            
            return await self._process_output_audio(output_audio)
            
        except Exception as e:
            logger.error(f"❌ 跨语言合成失败: {str(e)}")
            raise
    
    async def _instruct_synthesis_with_speaker(self, text: str, speaker_id: str, instruction: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定speaker进行指令式语音合成"""
        try:
            # 获取speaker的参考音频
            if speaker_id not in self.speakers_cache:
                raise ValueError(f"Speaker {speaker_id} 不存在")
            
            speaker_info = self.speakers_cache[speaker_id]
            reference_audio_path = speaker_info['reference_audio_path']
            
            # 准备参考音频
            processed_audio_path = await self._prepare_reference_audio(reference_audio_path)
            reference_audio = self.load_wav(processed_audio_path, self.model.sample_rate)
            
            # 使用默认指令如果未提供
            if not instruction:
                instruction = '用温和清晰的声音朗读'
            
            # 执行指令式合成
            output_audio = None
            stream = kwargs.get('stream', False)
            
            for i, result in enumerate(self.model.inference_instruct2(
                text, instruction, reference_audio, stream=stream
            )):
                output_audio = result['tts_speech']
                if not stream:
                    break
            
            if output_audio is None:
                raise Exception("指令式合成失败，未生成音频")
            
            return await self._process_output_audio(output_audio)
            
        except Exception as e:
            logger.error(f"❌ 指令式合成失败: {str(e)}")
            raise

    async def _process_output_audio(self, output_audio) -> Dict[str, Any]:
        """处理输出音频 - 修复采样率不匹配问题"""
        import io
        import tempfile
        
        try:
            logger.info(f"🔧 开始处理音频 - 张量形状: {output_audio.shape}")
            logger.info(f"🔧 音频数据类型: {output_audio.dtype}")
            logger.info(f"🔧 设备: {output_audio.device}")
            logger.info(f"🔧 原始采样率: {self.model.sample_rate}")
            
            # 确保音频在CPU上且为正确的数据类型
            if output_audio.device.type != 'cpu':
                output_audio = output_audio.cpu()
                logger.info("📱 音频已移至CPU")
            
            # 确保音频为float32类型
            if output_audio.dtype != torch.float32:
                output_audio = output_audio.float()
                logger.info(f"🔄 音频类型已转换为: {output_audio.dtype}")
            
            # 检查音频维度，确保是正确的格式 [channels, samples]
            if len(output_audio.shape) == 1:
                output_audio = output_audio.unsqueeze(0)
                logger.info(f"📏 添加通道维度: {output_audio.shape}")
            elif len(output_audio.shape) == 3:
                output_audio = output_audio.squeeze(0)
                logger.info(f"📏 压缩维度: {output_audio.shape}")
            
            # 检查通道数，如果是多通道，只使用第一个通道
            if output_audio.shape[0] > 1:
                output_audio = output_audio[0:1]
                logger.info(f"🎵 使用单声道: {output_audio.shape}")
            
            # 归一化音频到合适的范围
            max_val = output_audio.abs().max()
            if max_val > 1.0:
                output_audio = output_audio / max_val
                logger.info(f"🔊 音频已归一化，最大值从 {max_val:.6f} 归一化到 1.0")
            
            # **关键修复：采样率重采样**
            target_sample_rate = 22050  # 使用标准采样率
            if self.model.sample_rate != target_sample_rate:
                logger.info(f"🔄 重采样: {self.model.sample_rate}Hz -> {target_sample_rate}Hz")
                
                # 使用 torchaudio 进行重采样
                import torchaudio.transforms as T
                resampler = T.Resample(
                    orig_freq=self.model.sample_rate,
                    new_freq=target_sample_rate,
                    dtype=output_audio.dtype
                )
                output_audio = resampler(output_audio)
                logger.info(f"✅ 重采样完成 - 新形状: {output_audio.shape}")
                
                # 更新采样率
                actual_sample_rate = target_sample_rate
            else:
                actual_sample_rate = self.model.sample_rate
            
            # 检查音频是否包含NaN或无穷大
            if torch.isnan(output_audio).any():
                logger.error("❌ 检测到NaN值，用零替换")
                output_audio = torch.nan_to_num(output_audio, nan=0.0)
            
            if torch.isinf(output_audio).any():
                logger.error("❌ 检测到无穷大值，用零替换")
                output_audio = torch.nan_to_num(output_audio, posinf=0.0, neginf=0.0)
            
            # 使用临时文件保存音频
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            try:
                # 使用torchaudio保存音频，使用标准参数
                self.torchaudio.save(
                    temp_audio_path,
                    output_audio,
                    actual_sample_rate,  # 使用实际的采样率
                    format='wav',
                    encoding='PCM_S',
                    bits_per_sample=16
                )
                
                logger.info(f"💾 音频已保存到临时文件: {temp_audio_path} (采样率: {actual_sample_rate})")
                
                # 读取保存的WAV文件
                with open(temp_audio_path, 'rb') as f:
                    audio_data = f.read()
                
                logger.info(f"📁 WAV文件大小: {len(audio_data)} 字节")
                
                # **验证音频文件**
                try:
                    import wave
                    with wave.open(temp_audio_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        duration = frames / sample_rate
                        logger.info(f"📊 WAV验证 - 时长: {duration:.2f}s, 采样率: {sample_rate}, 通道: {channels}")
                except Exception as e:
                    logger.warning(f"⚠️ WAV文件验证失败: {str(e)}")
                
                # 编码为base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # 计算时长
                duration = output_audio.shape[1] / actual_sample_rate
                
                logger.info(f"✅ 音频处理完成 - 时长: {duration:.2f}s, Base64长度: {len(audio_base64)}, 采样率: {actual_sample_rate}")
                
                test_audio_path = f"/tmp/test_audio_{int(time.time())}.wav"
                try:
                    import shutil
                    shutil.copy2(temp_audio_path, test_audio_path)
                    logger.info(f"🎵 测试音频已保存到: {test_audio_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 保存测试音频失败: {str(e)}")
                
                return {
                    "audio_data": audio_base64,
                    "format": AudioFormat.WAV,
                    "duration": duration,
                    "model_used": "cosyvoice",
                    "sample_rate": actual_sample_rate,  # 返回实际使用的采样率
                    "channels": output_audio.shape[0],
                    "audio_shape": list(output_audio.shape)
                }
                
            finally:
                # 清理临时文件
                try:
                    Path(temp_audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ 音频处理失败: {str(e)}")
            logger.error(f"音频张量信息: shape={output_audio.shape}, dtype={output_audio.dtype}")
            raise Exception(f"音频处理失败: {str(e)}")
    
    async def get_speaker_list(self) -> Dict[str, Any]:
        """获取可用的speaker列表"""
        return {
            'speakers': list(self.speakers_cache.keys()),
            'default_speaker': self.default_speaker_id,
            'count': len(self.speakers_cache)
        }
    
    async def remove_speaker(self, speaker_id: str) -> bool:
        """删除指定的speaker"""
        try:
            if speaker_id in self.speakers_cache:
                del self.speakers_cache[speaker_id]
                
                # 如果删除的是默认speaker，清除默认设置
                if self.default_speaker_id == speaker_id:
                    self.default_speaker_id = None
                
                # 保存更新后的speaker信息
                await self._save_speaker_info()
                
                logger.info(f"✅ 已删除speaker: {speaker_id}")
                return True
            else:
                logger.warning(f"⚠️ Speaker不存在: {speaker_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 删除speaker失败: {str(e)}")
            return False

class MockSynthesizer(SpeechSynthesizer):
    """模拟语音合成器"""
    
    async def _setup(self):
        """设置模拟合成器"""
        logger.info("✅ 模拟合成器设置成功")
    
    async def synthesize(self, 
                       text: str, 
                       voice: Optional[str] = None,
                       language: str = "zh-CN",
                       speed: float = 1.0,
                       pitch: float = 8.0,
                       **kwargs) -> Dict[str, Any]:
        """模拟语音合成"""
        logger.info(f"🔊 开始模拟语音合成 - 文本长度: {len(text)}")
        start_time = time.time()
        
        # 生成模拟的静音音频
        await asyncio.sleep(0.2)  # 模拟处理时间
        
        # 生成音频时长
        sample_rate = 16000
        duration = min(len(text) / 5, 10)  # 最长10秒
        duration = max(duration, 1.0)  # 最短1秒
        
        # 创建静音音频
        samples = int(duration * sample_rate)
        audio_data = bytes(samples * 2)  # 16-bit PCM
        
        processing_time = time.time() - start_time
        
        return {
            "audio_data": base64.b64encode(audio_data).decode('utf-8'),
            "format": AudioFormat.PCM_16,
            "duration": duration,
            "processing_time": processing_time,
            "model_used": "mock_synthesizer"
        }

class SpeechProcessor:
    """语音处理器主类 - 修复版本"""
    
    def __init__(self):
        self.recognizers: Dict[str, SpeechRecognizer] = {}
        self.synthesizers: Dict[str, SpeechSynthesizer] = {}
        self.default_recognizer = None
        self.default_synthesizer = None
        self.is_initialized = False
        
        # 从环境变量或配置文件读取配置
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        return {
            # CosyVoice 配置
            'cosyvoice': {
                'model_dir': os.getenv('COSYVOICE_MODEL_DIR', '/home/rh/Program/MCP_Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B'),
                'cosyvoice_path': os.getenv('COSYVOICE_PATH', 'tools/CosyVoice'),
                'reference_audio': os.getenv('COSYVOICE_REF_AUDIO', None),
                'reference_text': os.getenv('COSYVOICE_REF_TEXT', '参考音频文本'),
                'load_jit': os.getenv('COSYVOICE_LOAD_JIT', 'false').lower() == 'true',
                'load_trt': os.getenv('COSYVOICE_LOAD_TRT', 'false').lower() == 'true',
                'fp16': os.getenv('COSYVOICE_FP16', 'false').lower() == 'true'
            },
            # SenseVoice 配置
            'sensevoice': {
                'model': os.getenv('SENSEVOICE_MODEL', 'iic/SenseVoiceSmall'),
                'max_single_segment_time': int(os.getenv('SENSEVOICE_MAX_SEGMENT_TIME', '60000')),
                'batch_size_s': int(os.getenv('SENSEVOICE_BATCH_SIZE_S', '300')),
                'batch_size_threshold_s': int(os.getenv('SENSEVOICE_BATCH_THRESHOLD_S', '60')),
                'ncpu': int(os.getenv('SENSEVOICE_NCPU', '4')),
                'batch_size': int(os.getenv('SENSEVOICE_BATCH_SIZE', '1'))
            },
            # 通用配置
            'device': os.getenv('SPEECH_DEVICE', 'cpu')
        }

    async def initialize(self):
        """初始化语音处理器"""
        logger.info("🔧 初始化语音处理器 - 修复版本")
        
        # 尝试初始化可用的识别器
        await self._try_initialize_recognizers()
        
        # 尝试初始化可用的合成器
        await self._try_initialize_synthesizers()
        
        self.is_initialized = True
        
        available_recognizers = list(self.recognizers.keys())
        available_synthesizers = list(self.synthesizers.keys())
        
        logger.info(f"✅ 语音处理器初始化完成")
        logger.info(f"  - 可用识别器: {available_recognizers}")
        logger.info(f"  - 默认识别器: {self.default_recognizer}")
        logger.info(f"  - 可用合成器: {available_synthesizers}")
        logger.info(f"  - 默认合成器: {self.default_synthesizer}")
        
        if not available_recognizers and not available_synthesizers:
            logger.warning("⚠️ 没有可用的语音处理引擎，将使用模拟模式")

    async def _try_initialize_recognizers(self):
        """尝试初始化语音识别器"""
        
        # 优先尝试 SenseVoice
        try:
            config = self.config['sensevoice'].copy()
            config['device'] = self.config['device']
            
            recognizer = SenseVoiceRecognizer(config)
            if await recognizer.initialize():
                self.recognizers['sensevoice'] = recognizer
                if self.default_recognizer is None:
                    self.default_recognizer = 'sensevoice'
                logger.info("✅ SenseVoice 识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ SenseVoice 识别器初始化失败: {str(e)}")
    
        # 如果没有可用的识别器，添加模拟识别器
        if not self.recognizers:
            recognizer = SenseVoiceRecognizer({'device': 'cpu'})  # 模拟模式
            await recognizer.initialize()
            self.recognizers['mock'] = recognizer
            self.default_recognizer = 'mock'
            logger.info("✅ 模拟识别器已启用")

    async def _try_initialize_synthesizers(self):
        """尝试初始化语音合成器"""
        
        # 只初始化 CosyVoice
        try:
            config = self.config['cosyvoice'].copy()
            config['device'] = self.config['device']
            
            # 检查模型目录是否存在
            if Path(config['model_dir']).exists():
                synthesizer = CosyVoiceSynthesizer(config)
                if await synthesizer.initialize():
                    self.synthesizers['cosyvoice'] = synthesizer
                    self.default_synthesizer = 'cosyvoice'
                    logger.info("✅ CosyVoice 合成器初始化成功")
                else:
                    logger.error("❌ CosyVoice 合成器初始化失败")
                    raise RuntimeError("CosyVoice 合成器初始化失败")
            else:
                logger.error(f"❌ CosyVoice 模型目录不存在: {config['model_dir']}")
                raise FileNotFoundError(f"CosyVoice 模型目录不存在: {config['model_dir']}")
        except Exception as e:
            logger.error(f"❌ CosyVoice 合成器初始化失败: {str(e)}")
            raise

    async def recognize(self, 
                       audio_data: bytes,
                       language: str = "zh-CN",
                       model_name: Optional[str] = None,
                       request_id: Optional[str] = None,
                       **kwargs) -> SpeechRecognitionResponse:
        """语音识别 - 优化版"""
        if not self.is_initialized:
            raise RuntimeError("语音处理器未初始化")
        
        # 选择识别器
        recognizer_name = model_name or self.default_recognizer
        if recognizer_name not in self.recognizers:
            # 回退到默认识别器
            recognizer_name = self.default_recognizer
            logger.warning(f"⚠️ 指定的识别器不可用，使用默认识别器: {recognizer_name}")
        
        recognizer = self.recognizers[recognizer_name]
        
        try:
            logger.info(f"🎤 开始语音识别 - 模型: {recognizer_name}, 语言: {language}")
            
            result = await recognizer.recognize(audio_data, language, **kwargs)
            
            logger.info(f"✅ 语音识别完成 - 文本长度: {len(result['text'])}")
            
            return SpeechRecognitionResponse(
                success=True,
                text=result["text"],
                language=result["language"],
                confidence=result["confidence"],
                processing_time=result["processing_time"],
                model_used=result["model_used"],
                request_id=request_id or generate_response_id(),
                timestamp=datetime.utcnow(),
                segments=result.get("segments", [])
            )
            
        except Exception as e:
            logger.error(f"❌ 语音识别失败: {str(e)}")
            # 返回错误响应而不是抛出异常
            return SpeechRecognitionResponse(
                success=False,
                text="",
                language=language,
                confidence=0.0,
                processing_time=0.0,
                model_used=recognizer_name,
                request_id=request_id or generate_response_id(),
                timestamp=datetime.utcnow(),
                message=f"识别失败: {str(e)}"
            )

    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        tts_model: Optional[str] = None,
                        request_id: Optional[str] = None,
                        **kwargs) -> SpeechSynthesisResponse:
        """语音合成 - 修复版本，支持音色一致性"""
        if not self.is_initialized:
            raise RuntimeError("语音处理器未初始化")
        
        # 强制使用CosyVoice合成器，忽略其他设置
        synthesizer_name = 'cosyvoice'
        if synthesizer_name not in self.synthesizers:
            raise ValueError(f"CosyVoice语音合成器不可用，请确保CosyVoice2-0.5b模型已安装")
        
        synthesizer = self.synthesizers[synthesizer_name]
        
        try:
            logger.info(f"🔊 开始语音合成 - 模型: {synthesizer_name}, 文本长度: {len(text)}")
            
            result = await synthesizer.synthesize(
                text=text,
                voice=voice,
                language=language,
                speed=speed,
                pitch=pitch,
                **kwargs
            )
            
            logger.info(f"✅ 语音合成完成 - 音频时长: {result['duration']:.2f}s")
            
            return SpeechSynthesisResponse(
                success=True,
                audio_data=result["audio_data"],
                format=result["format"],
                duration=result["duration"],
                processing_time=result["processing_time"],
                model_used=result["model_used"],
                request_id=request_id or generate_response_id(),
                timestamp=datetime.utcnow(),
                synthesis_mode=result.get("synthesis_mode", "default"),
                sample_rate=result.get("sample_rate", 16000)
            )
            
        except Exception as e:
            logger.error(f"❌ 语音合成失败: {str(e)}")
            raise

    async def add_speaker(self, 
                         reference_audio_path: str, 
                         reference_text: str, 
                         speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """添加新的speaker"""
        if 'cosyvoice' not in self.synthesizers:
            raise ValueError("CosyVoice合成器不可用")
        
        synthesizer = self.synthesizers['cosyvoice']
        if hasattr(synthesizer, '_add_speaker'):
            try:
                new_speaker_id = await synthesizer._add_speaker(reference_text, reference_audio_path, speaker_id)
                return {
                    'success': True,
                    'speaker_id': new_speaker_id,
                    'message': f'成功添加speaker: {new_speaker_id}'
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f'添加speaker失败: {str(e)}'
                }
        else:
            return {
                'success': False,
                'error': 'CosyVoice合成器不支持添加speaker',
                'message': 'CosyVoice合成器不支持添加speaker'
            }

    async def get_speaker_list(self) -> Dict[str, Any]:
        """获取可用的speaker列表"""
        if 'cosyvoice' not in self.synthesizers:
            return {'speakers': [], 'default_speaker': None, 'count': 0}
        
        synthesizer = self.synthesizers['cosyvoice']
        if hasattr(synthesizer, 'get_speaker_list'):
            return await synthesizer.get_speaker_list()
        else:
            return {'speakers': [], 'default_speaker': None, 'count': 0}

    async def remove_speaker(self, speaker_id: str) -> Dict[str, Any]:
        """删除指定的speaker"""
        if 'cosyvoice' not in self.synthesizers:
            return {'success': False, 'error': 'CosyVoice合成器不可用'}
        
        synthesizer = self.synthesizers['cosyvoice']
        if hasattr(synthesizer, 'remove_speaker'):
            success = await synthesizer.remove_speaker(speaker_id)
            return {
                'success': success,
                'message': f'Speaker {speaker_id} 删除{"成功" if success else "失败"}'
            }
        else:
            return {'success': False, 'error': 'CosyVoice合成器不支持删除speaker'}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            return {
                "healthy": self.is_initialized and (len(self.recognizers) > 0 or len(self.synthesizers) > 0),
                "recognizers": {
                    "available": list(self.recognizers.keys()),
                    "default": self.default_recognizer,
                    "count": len(self.recognizers)
                },
                "synthesizers": {
                    "available": list(self.synthesizers.keys()),
                    "default": self.default_synthesizer,
                    "count": len(self.synthesizers)
                },
                "config": {
                    "device": self.config['device'],
                    "cosyvoice_model_dir": self.config['cosyvoice']['model_dir'],
                    "sensevoice_model": self.config['sensevoice']['model']
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def get_available_voices(self, synthesizer_name: Optional[str] = None) -> Dict[str, List[str]]:
        """获取可用的声音列表"""
        voices = {}
        
        target_synthesizers = [synthesizer_name] if synthesizer_name else self.synthesizers.keys()
        
        for name in target_synthesizers:
            if name in self.synthesizers:
                synthesizer = self.synthesizers[name]
                if hasattr(synthesizer, 'get_available_voices'):
                    voices[name] = await synthesizer.get_available_voices()
                else:
                    voices[name] = ["default"]
        
        return voices

    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理语音处理器资源")
        
        # 清理识别器
        for recognizer in self.recognizers.values():
            if hasattr(recognizer, 'cleanup'):
                try:
                    await recognizer.cleanup()
                except Exception as e:
                    logger.warning(f"清理识别器时出错: {str(e)}")
        
        # 清理合成器
        for synthesizer in self.synthesizers.values():
            if hasattr(synthesizer, 'cleanup'):
                try:
                    await synthesizer.cleanup()
                except Exception as e:
                    logger.warning(f"清理合成器时出错: {str(e)}")
        
        self.recognizers.clear()
        self.synthesizers.clear()
        self.default_recognizer = None
        self.default_synthesizer = None
        self.is_initialized = False
        
        logger.info("✅ 语音处理器资源清理完成")

# 全局实例
speech_processor = SpeechProcessor()

# 便捷函数
async def initialize_speech_processor():
    """初始化全局语音处理器"""
    await speech_processor.initialize()

async def recognize_speech(audio_data: bytes, **kwargs) -> SpeechRecognitionResponse:
    """便捷的语音识别函数"""
    return await speech_processor.recognize(audio_data, **kwargs)

async def synthesize_speech(text: str, **kwargs) -> SpeechSynthesisResponse:
    """便捷的语音合成函数"""
    return await speech_processor.synthesize(text, **kwargs)

async def add_speaker(reference_audio_path: str, reference_text: str, speaker_id: Optional[str] = None) -> Dict[str, Any]:
    """便捷的添加speaker函数"""
    return await speech_processor.add_speaker(reference_audio_path, reference_text, speaker_id)

async def get_speaker_list() -> Dict[str, Any]:
    """便捷的获取speaker列表函数"""
    return await speech_processor.get_speaker_list()

async def remove_speaker(speaker_id: str) -> Dict[str, Any]:
    """便捷的删除speaker函数"""
    return await speech_processor.remove_speaker(speaker_id)