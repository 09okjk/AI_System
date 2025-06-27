"""
语音处理器 - 优化版本
负责语音识别和语音合成功能
基于 SenseVoice 和 CosyVoice 官方实现优化
"""

import asyncio
import base64
import tempfile
import time
import uuid
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np

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
    """CosyVoice 语音合成器 - 基于官方实现优化，支持speaker缓存"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None  # Initialize model attribute
        
        # 参考音频设置 - 早期初始化以支持测试
        self.reference_audio_path = config.get('reference_audio', None)
        self.reference_text = config.get('reference_text', '参考音频文本')
        
        # Speaker缓存相关
        self.speaker_cache = {}  # speaker_id -> speaker_info
        self.speaker_cache_dir = config.get('speaker_cache_dir', '/tmp/ai_system_speakers')
        self.default_speaker_id = 'default_speaker'
        
        # 确保speaker缓存目录存在
        Path(self.speaker_cache_dir).mkdir(parents=True, exist_ok=True)
    
    async def _setup(self):
        """设置 CosyVoice - 根据官方最佳实践"""
        try:
            # 参考音频设置 - 提前初始化
            self.reference_audio_path = self.config.get('reference_audio', None)
            self.reference_text = self.config.get('reference_text', '参考音频文本')
            
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
            
            # 初始化speaker缓存
            await self._initialize_speaker_cache()
            
            logger.info(f"✅ CosyVoice 模型初始化成功 - 模型路径: {model_dir}")
            logger.info(f"📁 Speaker缓存目录: {self.speaker_cache_dir}")
            
        except ImportError as e:
            logger.warning(f"⚠️ CosyVoice 未正确安装: {str(e)}")
            self.model = None
        except Exception as e:
            logger.error(f"❌ CosyVoice 初始化失败: {str(e)}")
            raise
    
    async def _initialize_speaker_cache(self):
        """初始化speaker缓存系统"""
        try:
            # 如果有默认参考音频，预加载为默认speaker
            if self.reference_audio_path and Path(self.reference_audio_path).exists():
                logger.info("🎤 预加载默认speaker...")
                await self._add_speaker_to_cache(
                    speaker_id=self.default_speaker_id,
                    reference_audio_path=self.reference_audio_path,
                    reference_text=self.reference_text
                )
                logger.info(f"✅ 默认speaker已加载: {self.default_speaker_id}")
            
            # 加载已保存的speaker缓存
            await self._load_speaker_cache_from_disk()
            
        except Exception as e:
            logger.warning(f"⚠️ Speaker缓存初始化失败: {str(e)}")
    
    async def _add_speaker_to_cache(self, speaker_id: str, reference_audio_path: str, reference_text: str):
        """添加speaker到缓存
        
        Args:
            speaker_id: speaker标识符
            reference_audio_path: 参考音频路径
            reference_text: 参考音频对应的文本
        """
        try:
            if self.model is None:
                logger.warning("模型未初始化，跳过speaker缓存")
                return
            
            # 检查音频文件格式并转换
            if Path(reference_audio_path).suffix.lower() != '.wav':
                logger.info(f"转换参考音频格式: {reference_audio_path}")
                reference_audio_path = convert_audio_to_wav(
                    reference_audio_path, 
                    sample_rate=self.model.sample_rate
                )
            
            # 加载音频
            reference_audio = self.load_wav(reference_audio_path, self.model.sample_rate)
            
            # 根据CosyVoice2官方最佳实践，使用add_zero_shot_spk保存speaker信息
            # 注意：这是基于官方文档的实现，如果实际API不同，需要调整
            if hasattr(self.model, 'add_zero_shot_spk'):
                speaker_info = self.model.add_zero_shot_spk(reference_audio, reference_text)
                logger.info(f"✅ 使用add_zero_shot_spk保存speaker: {speaker_id}")
            else:
                # 如果模型没有add_zero_shot_spk方法，我们保存原始数据
                speaker_info = {
                    'reference_audio': reference_audio,
                    'reference_text': reference_text,
                    'audio_path': reference_audio_path,
                    'sample_rate': self.model.sample_rate
                }
                logger.info(f"✅ 保存speaker原始信息: {speaker_id}")
            
            # 保存到内存缓存
            self.speaker_cache[speaker_id] = {
                'info': speaker_info,
                'reference_text': reference_text,
                'audio_path': reference_audio_path,
                'created_at': time.time()
            }
            
            # 持久化保存
            await self._save_speaker_to_disk(speaker_id)
            
            logger.info(f"🎤 Speaker已添加到缓存: {speaker_id}")
            
        except Exception as e:
            logger.error(f"❌ 添加speaker到缓存失败: {str(e)}")
            raise
    
    async def _get_speaker_from_cache(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """从缓存获取speaker信息"""
        return self.speaker_cache.get(speaker_id)
    
    async def _save_speaker_to_disk(self, speaker_id: str):
        """保存speaker信息到磁盘"""
        try:
            if speaker_id not in self.speaker_cache:
                return
            
            cache_file = Path(self.speaker_cache_dir) / f"{speaker_id}.json"
            speaker_data = self.speaker_cache[speaker_id].copy()
            
            # 不保存音频tensor，只保存元数据
            speaker_data['info'] = {
                'audio_path': speaker_data['audio_path'],
                'reference_text': speaker_data['reference_text'],
                'sample_rate': getattr(self.model, 'sample_rate', 22050),
                'created_at': speaker_data['created_at']
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(speaker_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 Speaker信息已保存到磁盘: {cache_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ 保存speaker到磁盘失败: {str(e)}")
    
    async def _load_speaker_cache_from_disk(self):
        """从磁盘加载speaker缓存"""
        try:
            cache_dir = Path(self.speaker_cache_dir)
            if not cache_dir.exists():
                return
            
            for cache_file in cache_dir.glob("*.json"):
                try:
                    speaker_id = cache_file.stem
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        speaker_data = json.load(f)
                    
                    # 验证音频文件是否仍然存在
                    audio_path = speaker_data['info']['audio_path']
                    if Path(audio_path).exists():
                        # 重新加载speaker到缓存
                        await self._add_speaker_to_cache(
                            speaker_id=speaker_id,
                            reference_audio_path=audio_path,
                            reference_text=speaker_data['reference_text']
                        )
                        logger.debug(f"📂 从磁盘加载speaker: {speaker_id}")
                    else:
                        logger.warning(f"⚠️ Speaker音频文件不存在，跳过: {audio_path}")
                        # 删除无效的缓存文件
                        cache_file.unlink(missing_ok=True)
                        
                except Exception as e:
                    logger.warning(f"⚠️ 加载speaker缓存文件失败 {cache_file}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"⚠️ 从磁盘加载speaker缓存失败: {str(e)}")
    
    def list_cached_speakers(self) -> List[str]:
        """列出所有缓存的speaker"""
        return list(self.speaker_cache.keys())
    
    async def synthesize(self, 
                        text: str, 
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        synthesis_mode: str = "instruct",
                        speaker_id: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """CosyVoice 语音合成 - 支持多种合成模式和speaker缓存"""
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
                    "synthesis_mode": synthesis_mode,
                    "speaker_id": speaker_id or "mock_speaker"
                }
            
            # 获取或创建speaker
            speaker_info = await self._get_or_create_speaker(kwargs, speaker_id)
            if speaker_info:
                kwargs['speaker_info'] = speaker_info
                kwargs['speaker_id'] = speaker_info.get('speaker_id', speaker_id)
            
            # 根据合成模式选择不同的方法
            if synthesis_mode == "zero_shot":
                result = await self._zero_shot_synthesis(text, kwargs)
            elif synthesis_mode == "cross_lingual":
                result = await self._cross_lingual_synthesis(text, kwargs)
            elif synthesis_mode == "instruct":
                result = await self._instruct_synthesis(text, kwargs)
            else:
                # 默认使用零样本合成
                result = await self._zero_shot_synthesis(text, kwargs)
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["synthesis_mode"] = synthesis_mode
            result["speaker_id"] = kwargs.get('speaker_id', speaker_id or 'default')
            
            log_speech_operation(
                logger, "synthesis", "cosyvoice", 
                len(text), len(result["audio_data"]), processing_time, 
                True, language
            )
            
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
    
    async def _get_or_create_speaker(self, kwargs: Dict[str, Any], speaker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取或创建speaker信息"""
        try:
            # 如果指定了speaker_id，优先使用缓存的speaker
            if speaker_id:
                cached_speaker = await self._get_speaker_from_cache(speaker_id)
                if cached_speaker:
                    logger.info(f"🎤 使用缓存的speaker: {speaker_id}")
                    return cached_speaker
                else:
                    logger.warning(f"⚠️ 指定的speaker不存在于缓存中: {speaker_id}")
            
            # 检查是否提供了新的参考音频
            reference_audio_path = kwargs.get('reference_audio')
            reference_text = kwargs.get('reference_text')
            
            if reference_audio_path and reference_text:
                # 生成speaker_id（如果未提供）
                if not speaker_id:
                    import hashlib
                    # 基于音频路径和文本生成唯一ID
                    content = f"{reference_audio_path}_{reference_text}"
                    speaker_id = hashlib.md5(content.encode()).hexdigest()[:12]
                
                # 检查该speaker是否已经缓存
                cached_speaker = await self._get_speaker_from_cache(speaker_id)
                if cached_speaker:
                    logger.info(f"🎤 使用已缓存的speaker: {speaker_id}")
                    return cached_speaker
                
                # 创建新的speaker
                logger.info(f"🎤 创建新的speaker: {speaker_id}")
                await self._add_speaker_to_cache(speaker_id, reference_audio_path, reference_text)
                return await self._get_speaker_from_cache(speaker_id)
            
            # 使用默认speaker
            if self.default_speaker_id in self.speaker_cache:
                logger.info(f"🎤 使用默认speaker: {self.default_speaker_id}")
                return self.speaker_cache[self.default_speaker_id]
            
            # 如果没有任何speaker，返回None（使用原始方式）
            logger.warning("⚠️ 没有可用的speaker，将使用传统方式处理参考音频")
            return None
            
        except Exception as e:
            logger.error(f"❌ 获取或创建speaker失败: {str(e)}")
            return None
    
    async def _zero_shot_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """零样本语音合成 - 支持speaker缓存"""
        # 优先使用缓存的speaker信息
        speaker_info = kwargs.get('speaker_info')
        
        if speaker_info and 'info' in speaker_info:
            # 使用缓存的speaker
            logger.info(f"🎤 使用缓存speaker进行零样本合成")
            cached_info = speaker_info['info']
            
            if hasattr(self.model, 'inference_zero_shot_with_spk') and 'reference_audio' in cached_info:
                # 如果模型支持预处理的speaker信息
                output_audio = None
                stream = kwargs.get('stream', False)
                
                for i, result in enumerate(self.model.inference_zero_shot_with_spk(
                    text, cached_info, stream=stream
                )):
                    output_audio = result['tts_speech']
                    if not stream:
                        break
                        
                if output_audio is None:
                    raise Exception("零样本合成失败，未生成音频")
                    
                return await self._process_output_audio(output_audio)
            
            else:
                # 使用原始音频信息
                reference_audio = cached_info.get('reference_audio')
                reference_text = speaker_info['reference_text']
                
                if reference_audio is not None:
                    output_audio = None
                    stream = kwargs.get('stream', False)
                    
                    for i, result in enumerate(self.model.inference_zero_shot(
                        text, reference_text, reference_audio, stream=stream
                    )):
                        output_audio = result['tts_speech']
                        if not stream:
                            break
                    
                    if output_audio is None:
                        raise Exception("零样本合成失败，未生成音频")
                    
                    return await self._process_output_audio(output_audio)
        
        # 回退到传统方式
        logger.info("🔄 回退到传统零样本合成方式")
        return await self._traditional_zero_shot_synthesis(text, kwargs)
    
    async def _traditional_zero_shot_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """传统零样本语音合成（原始实现）"""
        # 获取参考音频
        reference_audio_path = kwargs.get('reference_audio', self.reference_audio_path)
        reference_text = kwargs.get('reference_text', self.reference_text)
        
        if not reference_audio_path or not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {reference_audio_path}")
        
        # 检查是否需要转换音频格式
        reference_audio_ext = Path(reference_audio_path).suffix.lower()
        if reference_audio_ext != '.wav':
            logger.info(f"参考音频非WAV格式 ({reference_audio_ext})，进行格式转换")
            try:
                reference_audio_path = convert_audio_to_wav(reference_audio_path, sample_rate=self.model.sample_rate)
            except Exception as e:
                logger.error(f"参考音频转换失败: {str(e)}")
                raise ValueError(f"参考音频格式转换失败: {str(e)}")
        
        # 加载参考音频
        reference_audio = self.load_wav(reference_audio_path, self.model.sample_rate)
        
        # 执行合成
        output_audio = None
        stream = kwargs.get('stream', False)
        
        for i, result in enumerate(self.model.inference_zero_shot(
            text, reference_text, reference_audio, stream=stream
        )):
            output_audio = result['tts_speech']
            if not stream:  # 非流式模式只取第一个结果
                break
        
        if output_audio is None:
            raise Exception("零样本合成失败，未生成音频")
        
        return await self._process_output_audio(output_audio)
    
    async def _cross_lingual_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """跨语言语音合成 - 支持speaker缓存"""
        # 优先使用缓存的speaker信息
        speaker_info = kwargs.get('speaker_info')
        
        if speaker_info and 'info' in speaker_info:
            logger.info(f"🎤 使用缓存speaker进行跨语言合成")
            cached_info = speaker_info['info']
            
            # 使用原始音频信息
            reference_audio = cached_info.get('reference_audio')
            
            if reference_audio is not None:
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
        
        # 回退到传统方式
        logger.info("🔄 回退到传统跨语言合成方式")
        return await self._traditional_cross_lingual_synthesis(text, kwargs)
    
    async def _traditional_cross_lingual_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """传统跨语言语音合成（原始实现）"""
        reference_audio_path = kwargs.get('reference_audio', self.reference_audio_path)
        
        if not reference_audio_path or not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {reference_audio_path}")
        
        # 检查是否需要转换音频格式
        reference_audio_ext = Path(reference_audio_path).suffix.lower()
        if reference_audio_ext != '.wav':
            logger.info(f"参考音频非WAV格式 ({reference_audio_ext})，进行格式转换")
            try:
                reference_audio_path = convert_audio_to_wav(reference_audio_path, sample_rate=self.model.sample_rate)
            except Exception as e:
                logger.error(f"参考音频转换失败: {str(e)}")
                raise ValueError(f"参考音频格式转换失败: {str(e)}")
        
        # 加载参考音频
        reference_audio = self.load_wav(reference_audio_path, self.model.sample_rate)
        
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
    
    async def _instruct_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """指令式语音合成 - 支持speaker缓存"""
        # 优先使用缓存的speaker信息
        speaker_info = kwargs.get('speaker_info')
        instruction = kwargs.get('instruction', '用温和的中文女声朗读')
        
        if speaker_info and 'info' in speaker_info:
            logger.info(f"🎤 使用缓存speaker进行指令式合成")
            cached_info = speaker_info['info']
            
            # 使用原始音频信息
            reference_audio = cached_info.get('reference_audio')
            
            if reference_audio is not None:
                output_audio = None
                stream = kwargs.get('stream', False)
                
                # 执行指令式合成 - 使用正确的方法名 inference_instruct2
                for i, result in enumerate(self.model.inference_instruct2(
                    text, instruction, reference_audio, stream=stream
                )):
                    output_audio = result['tts_speech']
                    if not stream:
                        break
                
                if output_audio is None:
                    raise Exception("指令式合成失败，未生成音频")
                
                return await self._process_output_audio(output_audio)
        
        # 回退到传统方式
        logger.info("🔄 回退到传统指令式合成方式")
        return await self._traditional_instruct_synthesis(text, kwargs)
    
    async def _traditional_instruct_synthesis(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """传统指令式语音合成（原始实现）"""
        reference_audio_path = kwargs.get('reference_audio', self.reference_audio_path)
        instruction = kwargs.get('instruction', '用温和的中文女声朗读')
        
        if not reference_audio_path or not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {reference_audio_path}")
        
        # 检查是否需要转换音频格式
        reference_audio_ext = Path(reference_audio_path).suffix.lower()
        if reference_audio_ext != '.wav':
            logger.info(f"参考音频非WAV格式 ({reference_audio_ext})，进行格式转换")
            try:
                reference_audio_path = convert_audio_to_wav(reference_audio_path, sample_rate=self.model.sample_rate)
            except Exception as e:
                logger.error(f"参考音频转换失败: {str(e)}")
                raise ValueError(f"参考音频格式转换失败: {str(e)}")
        
        # 加载参考音频
        reference_audio = self.load_wav(reference_audio_path, self.model.sample_rate)
        
        # 执行指令式合成 - 使用正确的方法名 inference_instruct2
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
    
    async def _process_output_audio(self, output_audio) -> Dict[str, Any]:
        """处理输出音频"""
        import io
        
        # 转换为字节数据
        buffer = io.BytesIO()
        
        try:
            # 明确指定音频格式和位深度
            self.torchaudio.save(
                buffer, 
                output_audio, 
                self.model.sample_rate, 
                format='wav',
                bits_per_sample=16,
                encoding='PCM_S'
            )
            buffer.seek(0)
            audio_data = buffer.read()
            
            # 编码为base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 计算时长
            duration = output_audio.shape[1] / self.model.sample_rate
            
            return {
                "audio_data": audio_base64,
                "format": AudioFormat.WAV,
                "duration": duration,
                "model_used": "cosyvoice",
                "sample_rate": self.model.sample_rate
            }
        except Exception as e:
            # 提供更详细的错误信息
            logger.error(f"❌ 音频处理失败: {str(e)}")
            raise Exception(f"音频处理失败: {str(e)}")
    
    # Public speaker management methods
    async def add_speaker(self, speaker_id: str, reference_audio_path: str, reference_text: str) -> bool:
        """添加新的speaker到缓存
        
        Args:
            speaker_id: speaker标识符
            reference_audio_path: 参考音频路径
            reference_text: 参考音频对应的文本
            
        Returns:
            bool: 添加是否成功
        """
        try:
            await self._add_speaker_to_cache(speaker_id, reference_audio_path, reference_text)
            logger.info(f"✅ Speaker已添加: {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 添加speaker失败: {str(e)}")
            return False
    
    async def remove_speaker(self, speaker_id: str) -> bool:
        """从缓存中移除speaker
        
        Args:
            speaker_id: speaker标识符
            
        Returns:
            bool: 移除是否成功
        """
        try:
            if speaker_id in self.speaker_cache:
                del self.speaker_cache[speaker_id]
                
                # 删除磁盘文件
                cache_file = Path(self.speaker_cache_dir) / f"{speaker_id}.json"
                cache_file.unlink(missing_ok=True)
                
                logger.info(f"🗑️ Speaker已移除: {speaker_id}")
                return True
            else:
                logger.warning(f"⚠️ Speaker不存在: {speaker_id}")
                return False
        except Exception as e:
            logger.error(f"❌ 移除speaker失败: {str(e)}")
            return False
    
    def get_speaker_info(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """获取speaker信息
        
        Args:
            speaker_id: speaker标识符
            
        Returns:
            Dict: speaker信息，如果不存在返回None
        """
        speaker_data = self.speaker_cache.get(speaker_id)
        if speaker_data:
            return {
                'speaker_id': speaker_id,
                'reference_text': speaker_data['reference_text'],
                'audio_path': speaker_data['audio_path'],
                'created_at': speaker_data['created_at']
            }
        return None
    
    async def validate_speaker_consistency(self, speaker_id: str, test_texts: List[str]) -> Dict[str, Any]:
        """验证speaker的音色一致性
        
        Args:
            speaker_id: speaker标识符
            test_texts: 测试文本列表
            
        Returns:
            Dict: 验证结果
        """
        if speaker_id not in self.speaker_cache:
            return {"error": f"Speaker不存在: {speaker_id}"}
        
        try:
            results = []
            for i, text in enumerate(test_texts):
                result = await self.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    synthesis_mode="zero_shot"
                )
                results.append({
                    "text": text,
                    "audio_data": result["audio_data"],
                    "duration": result["duration"],
                    "processing_time": result["processing_time"]
                })
            
            return {
                "speaker_id": speaker_id,
                "test_count": len(test_texts),
                "results": results,
                "success": True
            }
            
        except Exception as e:
            return {
                "speaker_id": speaker_id,
                "error": str(e),
                "success": False
            }

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
    """语音处理器主类 - 优化版本"""
    
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
                'speaker_cache_dir': os.getenv('COSYVOICE_SPEAKER_CACHE_DIR', '/tmp/ai_system_speakers'),
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
        logger.info("🔧 初始化语音处理器 - 优化版本")
        
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
        """语音合成 - 优化版"""
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