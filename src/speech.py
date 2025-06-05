"""
语音处理器
负责语音识别和语音合成功能
"""

import asyncio
import base64
import tempfile
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import os

from .logger import get_logger, log_speech_operation
from .models import SpeechRecognitionResponse, SpeechSynthesisResponse, AudioFormat
from .utils import generate_response_id

logger = get_logger(__name__)

class SpeechRecognizer:
    """语音识别器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化识别器"""
        try:
            await self._setup()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ 语音识别器初始化失败: {str(e)}")
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

class SensVoiceRecognizer(SpeechRecognizer):
    """SensVoice 语音识别器"""
    
    async def _setup(self):
        """设置 SensVoice"""
        try:
            # 这里集成您之前创建的 SensVoice MCP 客户端
            # 或者直接导入 funasr
            from funasr import AutoModel
            
            # 初始化模型
            self.model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device=self.config.get('device', 'cpu')
            )
            
            logger.info("✅ SensVoice 模型初始化成功")
            
        except ImportError:
            logger.warning("⚠️ FunASR 未安装，使用模拟识别器")
            self.model = None
        except Exception as e:
            logger.error(f"❌ SensVoice 初始化失败: {str(e)}")
            raise
    
    async def recognize(self, 
                       audio_data: bytes, 
                       language: str = "zh-CN",
                       **kwargs) -> Dict[str, Any]:
        """SensVoice 语音识别"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # 模拟识别结果
                await asyncio.sleep(0.5)  # 模拟处理时间
                
                processing_time = time.time() - start_time
                
                return {
                    "text": "这是一个模拟的语音识别结果",
                    "language": language,
                    "confidence": 0.95,
                    "processing_time": processing_time,
                    "model_used": "mock_sensvoice"
                }
            
            # 保存音频到临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # 执行识别
                result = self.model.generate(
                    input=temp_audio_path,
                    **kwargs
                )
                
                processing_time = time.time() - start_time
                
                # 提取识别结果
                if result and len(result) > 0:
                    text = result[0].get('text', '')
                    confidence = result[0].get('confidence', 0.0)
                else:
                    text = ""
                    confidence = 0.0
                
                log_speech_operation(
                    logger, "recognition", "sensvoice", 
                    len(audio_data), len(text), processing_time, 
                    True, language
                )
                
                return {
                    "text": text,
                    "language": language,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "model_used": "sensvoice"
                }
                
            finally:
                # 清理临时文件
                Path(temp_audio_path).unlink(missing_ok=True)
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "recognition", "sensvoice", 
                len(audio_data), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"语音识别失败: {error_msg}")

class WhisperRecognizer(SpeechRecognizer):
    """Whisper 语音识别器"""
    
    async def _setup(self):
        """设置 Whisper"""
        try:
            import whisper
            
            model_size = self.config.get('model_size', 'base')
            self.model = whisper.load_model(model_size)
            
            logger.info(f"✅ Whisper 模型初始化成功: {model_size}")
            
        except ImportError:
            logger.warning("⚠️ Whisper 未安装，使用模拟识别器")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Whisper 初始化失败: {str(e)}")
            raise
    
    async def recognize(self, 
                       audio_data: bytes, 
                       language: str = "zh",
                       **kwargs) -> Dict[str, Any]:
        """Whisper 语音识别"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # 模拟识别结果
                await asyncio.sleep(1.0)
                
                processing_time = time.time() - start_time
                
                return {
                    "text": "这是一个模拟的 Whisper 识别结果",
                    "language": language,
                    "confidence": 0.90,
                    "processing_time": processing_time,
                    "model_used": "mock_whisper"
                }
            
            # 保存音频到临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            try:
                # 执行识别
                result = self.model.transcribe(
                    temp_audio_path,
                    language=language if language != "zh-CN" else "zh",
                    **kwargs
                )
                
                processing_time = time.time() - start_time
                
                text = result.get('text', '')
                
                # Whisper 不直接提供置信度，使用平均概率
                segments = result.get('segments', [])
                confidence = 0.0
                if segments:
                    confidences = [seg.get('avg_logprob', 0.0) for seg in segments]
                    confidence = sum(confidences) / len(confidences)
                    confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))  # 转换为0-1范围
                
                log_speech_operation(
                    logger, "recognition", "whisper", 
                    len(audio_data), len(text), processing_time, 
                    True, language
                )
                
                return {
                    "text": text,
                    "language": language,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "model_used": "whisper"
                }
                
            finally:
                # 清理临时文件
                Path(temp_audio_path).unlink(missing_ok=True)
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "recognition", "whisper", 
                len(audio_data), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"Whisper 识别失败: {error_msg}")

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
                       pitch: float = 1.0,
                       **kwargs) -> Dict[str, Any]:
        """模拟语音合成"""
        start_time = time.time()
        
        # 生成模拟的静音音频
        await asyncio.sleep(0.2)  # 模拟处理时间
        
        # 生成2秒的静音音频
        sample_rate = 16000
        duration = min(len(text) / 5, 10)  # 最长10秒
        duration = max(duration, 1.0)  # 最短1秒
        
        # 创建静音音频
        samples = int(duration * sample_rate)
        audio_data = bytes(samples * 2)  # 16-bit PCM
        
        processing_time = time.time() - start_time
        
        return {
            "audio_data": audio_data,
            "format": AudioFormat.PCM_16,
            "duration": duration,
            "processing_time": processing_time,
            "model_used": "mock_synthesizer"
        }

class CosyVoiceSynthesizer(SpeechSynthesizer):
    """CosyVoice 语音合成器"""
    
    async def _setup(self):
        """设置 CosyVoice"""
        try:
            # 这里集成您之前创建的 CosyVoice MCP 客户端
            # 或者直接导入 CosyVoice
            import sys
            sys.path.append('third_party/Matcha-TTS')
            
            from cosyvoice.cli.cosyvoice import CosyVoice2
            from cosyvoice.utils.file_utils import load_wav
            import torchaudio
            
            model_dir = self.config.get('model_dir', 'pretrained_models/CosyVoice2-0.5B')
            
            self.model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
            self.load_wav = load_wav
            self.torchaudio = torchaudio
            
            logger.info("✅ CosyVoice 模型初始化成功")
            
        except ImportError:
            logger.warning("⚠️ CosyVoice 未安装，使用模拟合成器")
            self.model = None
        except Exception as e:
            logger.error(f"❌ CosyVoice 初始化失败: {str(e)}")
            raise
    
    async def synthesize(self, 
                        text: str, 
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        **kwargs) -> Dict[str, Any]:
        """CosyVoice 语音合成"""
        start_time = time.time()
        
        try:
            if self.model is None:
                # 模拟合成结果
                await asyncio.sleep(1.5)
                
                processing_time = time.time() - start_time
                
                # 创建模拟音频数据（静音）
                sample_rate = 22050
                duration = 2.0  # 2秒
                audio_samples = int(sample_rate * duration)
                mock_audio = b'\x00' * (audio_samples * 2)  # 16位PCM
                
                audio_base64 = base64.b64encode(mock_audio).decode('utf-8')
                
                return {
                    "audio_data": audio_base64,
                    "format": AudioFormat.WAV,
                    "duration": duration,
                    "processing_time": processing_time,
                    "model_used": "mock_cosyvoice"
                }
            
            # 使用预设的参考音频（需要提供）
            reference_audio_path = self.config.get('reference_audio', 'reference.wav')
            reference_text = self.config.get('reference_text', '参考音频文本')
            
            if not Path(reference_audio_path).exists():
                # 如果没有参考音频，使用 SFT 模式（如果支持）
                logger.warning("⚠️ 未找到参考音频，使用默认合成模式")
                
                # 模拟输出
                processing_time = time.time() - start_time
                mock_audio = b'\x00' * (22050 * 2 * 2)  # 2秒静音
                audio_base64 = base64.b64encode(mock_audio).decode('utf-8')
                
                return {
                    "audio_data": audio_base64,
                    "format": AudioFormat.WAV,
                    "duration": 2.0,
                    "processing_time": processing_time,
                    "model_used": "cosyvoice_sft"
                }
            
            # 加载参考音频
            reference_audio = self.load_wav(reference_audio_path, 16000)
            
            # 执行合成
            output_audio = None
            for i, result in enumerate(self.model.inference_zero_shot(
                text, reference_text, reference_audio, stream=False
            )):
                output_audio = result['tts_speech']
                break  # 只取第一个结果
            
            if output_audio is None:
                raise Exception("合成失败，未生成音频")
            
            processing_time = time.time() - start_time
            
            # 转换为字节数据
            import io
            buffer = io.BytesIO()
            self.torchaudio.save(buffer, output_audio, self.model.sample_rate, format='wav')
            buffer.seek(0)
            audio_data = buffer.read()
            
            # 编码为base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 计算时长
            duration = output_audio.shape[1] / self.model.sample_rate
            
            log_speech_operation(
                logger, "synthesis", "cosyvoice", 
                len(text), len(audio_data), processing_time, 
                True, language
            )
            
            return {
                "audio_data": audio_base64,
                "format": AudioFormat.WAV,
                "duration": duration,
                "processing_time": processing_time,
                "model_used": "cosyvoice"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "synthesis", "cosyvoice", 
                len(text), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"CosyVoice 合成失败: {error_msg}")

class EdgeTTSSynthesizer(SpeechSynthesizer):
    """Edge TTS 语音合成器"""
    
    async def _setup(self):
        """设置 Edge TTS"""
        try:
            import edge_tts
            self.edge_tts = edge_tts
            
            logger.info("✅ Edge TTS 初始化成功")
            
        except ImportError:
            logger.warning("⚠️ Edge TTS 未安装，使用模拟合成器")
            self.edge_tts = None
        except Exception as e:
            logger.error(f"❌ Edge TTS 初始化失败: {str(e)}")
            raise
    
    async def synthesize(self, 
                        text: str, 
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        **kwargs) -> Dict[str, Any]:
        """Edge TTS 语音合成"""
        start_time = time.time()
        
        try:
            if self.edge_tts is None:
                # 模拟合成结果
                await asyncio.sleep(0.8)
                
                processing_time = time.time() - start_time
                
                # 创建模拟音频数据
                mock_audio = b'\x00' * (16000 * 2 * 2)  # 2秒，16kHz，16位
                audio_base64 = base64.b64encode(mock_audio).decode('utf-8')
                
                return {
                    "audio_data": audio_base64,
                    "format": AudioFormat.WAV,
                    "duration": 2.0,
                    "processing_time": processing_time,
                    "model_used": "mock_edge_tts"
                }
            
            # 选择声音
            if not voice:
                voice = self._get_default_voice(language)
            
            # 创建TTS实例
            tts = self.edge_tts.Communicate(text, voice)
            
            # 生成音频
            audio_data = b""
            async for chunk in tts.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            processing_time = time.time() - start_time
            
            # 编码为base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 估算时长（简单估算，实际可能需要解析音频文件）
            duration = len(text) * 0.1  # 简单估算：每个字符0.1秒
            
            log_speech_operation(
                logger, "synthesis", "edge_tts", 
                len(text), len(audio_data), processing_time, 
                True, language
            )
            
            return {
                "audio_data": audio_base64,
                "format": AudioFormat.MP3,  # Edge TTS 默认输出MP3
                "duration": duration,
                "processing_time": processing_time,
                "model_used": "edge_tts"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            log_speech_operation(
                logger, "synthesis", "edge_tts", 
                len(text), 0, processing_time, 
                False, language, error_msg
            )
            
            raise Exception(f"Edge TTS 合成失败: {error_msg}")
    
    def _get_default_voice(self, language: str) -> str:
        """获取默认声音"""
        voice_map = {
            "zh-CN": "zh-CN-XiaoxiaoNeural",
            "zh-TW": "zh-TW-HsiaoyuNeural", 
            "en-US": "en-US-AriaNeural",
            "ja-JP": "ja-JP-NanamiNeural",
            "ko-KR": "ko-KR-SunHiNeural"
        }
        return voice_map.get(language, "zh-CN-XiaoxiaoNeural")

class SpeechProcessor:
    """语音处理器主类"""
    
    def __init__(self):
        self.recognizers: Dict[str, Any] = {}
        self.synthesizers: Dict[str, Any] = {}
        self.default_recognizer = None
        self.default_synthesizer = None
        self.is_initialized = False
        
        # 从环境变量或配置文件读取模型路径
        self.cosyvoice_model_dir = os.getenv(
            'COSYVOICE_MODEL_DIR', 
            'pretrained_models/CosyVoice2-0.5B'
        )
        self.whisper_model_size = os.getenv('WHISPER_MODEL_SIZE', 'base')
        self.device = os.getenv('SPEECH_DEVICE', 'cpu')

    async def initialize(self):
        """初始化语音处理器"""
        logger.info("🔧 初始化语音处理器")
        
        # 尝试初始化可用的识别器
        await self._try_initialize_recognizers()
        
        # 尝试初始化可用的合成器
        await self._try_initialize_synthesizers()
        
        self.is_initialized = True
        
        available_recognizers = list(self.recognizers.keys())
        available_synthesizers = list(self.synthesizers.keys())
        
        logger.info(f"✅ 语音处理器初始化完成")
        logger.info(f"  - 可用识别器: {available_recognizers}")
        logger.info(f"  - 可用合成器: {available_synthesizers}")
        
        if not available_recognizers and not available_synthesizers:
            logger.warning("⚠️ 没有可用的语音处理引擎，将使用模拟模式")
    

    async def _try_initialize_recognizers(self):
        """尝试初始化语音识别器"""
        
        # 尝试 SensVoice/FunASR
        try:
            from funasr import AutoModel
            
            model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device=self.device
            )
            
            self.recognizers['sensvoice'] = {
                'model': model,
                'type': 'sensvoice'
            }
            
            if self.default_recognizer is None:
                self.default_recognizer = 'sensvoice'
            
            logger.info("✅ SensVoice 识别器初始化成功")
            
        except ImportError:
            logger.info("ℹ️ FunASR 未安装，跳过 SensVoice 识别器")
        except Exception as e:
            logger.warning(f"⚠️ SensVoice 识别器初始化失败: {str(e)}")
        
        # 尝试 Whisper
        try:
            import whisper
            
            model = whisper.load_model(self.whisper_model_size)
            
            self.recognizers['whisper'] = {
                'model': model,
                'type': 'whisper'
            }
            
            if self.default_recognizer is None:
                self.default_recognizer = 'whisper'
            
            logger.info(f"✅ Whisper 识别器初始化成功 (模型: {self.whisper_model_size})")
            
        except ImportError:
            logger.info("ℹ️ Whisper 未安装，跳过 Whisper 识别器")
        except Exception as e:
            logger.warning(f"⚠️ Whisper 识别器初始化失败: {str(e)}")
        
        # 如果没有可用的识别器，添加模拟识别器
        if not self.recognizers:
            self.recognizers['mock'] = {
                'model': None,
                'type': 'mock'
            }
            self.default_recognizer = 'mock'
            logger.info("✅ 模拟识别器已启用")
    

    async def _try_initialize_synthesizers(self):
        """尝试初始化语音合成器"""
        
        # 尝试 CosyVoice
        try:
            # 检查模型目录是否存在
            if Path(self.cosyvoice_model_dir).exists():
                import sys
                sys.path.append('third_party/Matcha-TTS')
                
                from cosyvoice.cli.cosyvoice import CosyVoice2
                from cosyvoice.utils.file_utils import load_wav
                import torchaudio
                
                model = CosyVoice2(self.cosyvoice_model_dir, load_jit=False, load_trt=False, fp16=False)
                
                self.synthesizers['cosyvoice'] = CosyVoiceSynthesizer({'model_dir': self.cosyvoice_model_dir})
                await self.synthesizers['cosyvoice'].initialize()
                
                if self.default_synthesizer is None:
                    self.default_synthesizer = 'cosyvoice'
                
                logger.info("✅ CosyVoice 合成器初始化成功")
            else:
                logger.info(f"ℹ️ CosyVoice 模型目录不存在: {self.cosyvoice_model_dir}")
                
        except ImportError:
            logger.info("ℹ️ CosyVoice 依赖未安装，跳过 CosyVoice 合成器")
        except Exception as e:
            logger.warning(f"⚠️ CosyVoice 合成器初始化失败: {str(e)}")
        
        # 尝试 Edge TTS
        try:
            import edge_tts
            
            self.synthesizers['edge_tts'] = EdgeTTSSynthesizer({})
            await self.synthesizers['edge_tts'].initialize()
            
            if self.default_synthesizer is None:
                self.default_synthesizer = 'edge_tts'
            
            logger.info("✅ Edge TTS 合成器初始化成功")
            
        except ImportError:
            logger.info("ℹ️ Edge TTS 未安装，跳过 Edge TTS 合成器")
        except Exception as e:
            logger.warning(f"⚠️ Edge TTS 合成器初始化失败: {str(e)}")
        
        # 如果没有可用的合成器，添加模拟合成器
        if not self.synthesizers:
            self.synthesizers['mock'] = MockSynthesizer({})
            await self.synthesizers['mock'].initialize()
            self.default_synthesizer = 'mock'
            logger.info("✅ 模拟合成器已启用")
    

    async def recognize(self, 
                       audio_data: bytes,
                       language: str = "zh-CN",
                       model_name: Optional[str] = None,
                       request_id: Optional[str] = None) -> SpeechRecognitionResponse:
        """语音识别 - 改进版"""
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
            
            result = await self._perform_recognition(recognizer, audio_data, language)
            
            logger.info(f"✅ 语音识别完成 - 文本长度: {len(result['text'])}")
            
            return SpeechRecognitionResponse(
                success=True,
                text=result["text"],
                language=result["language"],
                confidence=result["confidence"],
                processing_time=result["processing_time"],
                model_used=result["model_used"],
                request_id=request_id or generate_response_id(),
                timestamp=datetime.utcnow()
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
    
    async def _perform_recognition(self, recognizer: Dict[str, Any], audio_data: bytes, language: str) -> Dict[str, Any]:
        """执行具体的语音识别"""
        start_time = time.time()
        
        if recognizer['type'] == 'mock':
            # 模拟识别
            await asyncio.sleep(0.5)
            processing_time = time.time() - start_time
            
            return {
                "text": "这是一个模拟的语音识别结果",
                "language": language,
                "confidence": 0.95,
                "processing_time": processing_time,
                "model_used": "mock_recognizer"
            }
        
        elif recognizer['type'] == 'sensvoice':
            return await self._sensvoice_recognize(recognizer, audio_data, language, start_time)
        
        elif recognizer['type'] == 'whisper':
            return await self._whisper_recognize(recognizer, audio_data, language, start_time)
        
        else:
            raise ValueError(f"未知的识别器类型: {recognizer['type']}")
    
    async def _sensvoice_recognize(self, recognizer: Dict[str, Any], audio_data: bytes, language: str, start_time: float) -> Dict[str, Any]:
        """SensVoice 识别实现"""
        model = recognizer['model']
        
        # 保存音频到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_audio_path = temp_file.name
        
        try:
            # 执行识别
            result = model.generate(input=temp_audio_path)
            
            processing_time = time.time() - start_time
            
            # 提取识别结果
            if result and len(result) > 0:
                text = result[0].get('text', '')
                confidence = result[0].get('confidence', 0.0)
            else:
                text = ""
                confidence = 0.0
            
            log_speech_operation(
                logger, "recognition", "sensvoice", 
                len(audio_data), len(text), processing_time, 
                True, language
            )
            
            return {
                "text": text,
                "language": language,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_used": "sensvoice"
            }
            
        finally:
            # 清理临时文件
            Path(temp_audio_path).unlink(missing_ok=True)
    
    async def _whisper_recognize(self, recognizer: Dict[str, Any], audio_data: bytes, language: str, start_time: float) -> Dict[str, Any]:
        """Whisper 识别实现"""
        model = recognizer['model']
        
        # 保存音频到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_audio_path = temp_file.name
        
        try:
            # 执行识别
            result = model.transcribe(
                temp_audio_path,
                language=language if language != "zh-CN" else "zh"
            )
            
            processing_time = time.time() - start_time
            
            text = result.get('text', '')
            
            # Whisper 不直接提供置信度，使用平均概率估算
            segments = result.get('segments', [])
            confidence = 0.0
            if segments:
                confidences = [seg.get('avg_logprob', 0.0) for seg in segments]
                confidence = sum(confidences) / len(confidences)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            log_speech_operation(
                logger, "recognition", "whisper", 
                len(audio_data), len(text), processing_time, 
                True, language
            )
            
            return {
                "text": text,
                "language": language,
                "confidence": confidence,
                "processing_time": processing_time,
                "model_used": "whisper"
            }
            
        finally:
            # 清理临时文件
            Path(temp_audio_path).unlink(missing_ok=True)
    
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        language: str = "zh-CN",
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        tts_model: Optional[str] = None,
                        request_id: Optional[str] = None) -> SpeechSynthesisResponse:
        """语音合成"""
        if not self.is_initialized:
            raise RuntimeError("语音处理器未初始化")
        
        # 选择合成器
        synthesizer_name = tts_model or self.default_synthesizer
        if synthesizer_name not in self.synthesizers:
            raise ValueError(f"语音合成器不可用: {synthesizer_name}")
        
        synthesizer = self.synthesizers[synthesizer_name]
        
        try:
            logger.info(f"🔊 开始语音合成 - 模型: {synthesizer_name}, 文本长度: {len(text)}")
            
            # 如果synthesizer是SpeechSynthesizer的实例，直接调用其synthesize方法
            if isinstance(synthesizer, SpeechSynthesizer):
                result = await synthesizer.synthesize(
                    text=text,
                    voice=voice,
                    language=language,
                    speed=speed,
                    pitch=pitch
                )
            # 否则，按照原来的逻辑处理字典类型的synthesizer
            else:
                if synthesizer['type'] == 'cosyvoice':
                    result = await self._cosyvoice_synthesize(synthesizer, text, voice, language, speed, pitch)
                elif synthesizer['type'] == 'edge_tts':
                    result = await self._edge_tts_synthesize(synthesizer, text, voice, language, speed, pitch)
                elif synthesizer['type'] == 'mock':
                    # 创建一个临时的MockSynthesizer来处理
                    mock_synth = MockSynthesizer({})
                    await mock_synth.initialize()
                    result = await mock_synth.synthesize(
                        text=text, 
                        voice=voice, 
                        language=language,
                        speed=speed,
                        pitch=pitch
                    )
                else:
                    raise ValueError(f"未知的合成器类型: {synthesizer['type']}")
            
            logger.info(f"✅ 语音合成完成 - 音频时长: {result['duration']:.2f}s")
            
            return SpeechSynthesisResponse(
                success=True,
                audio_data=result["audio_data"],
                format=result["format"],
                duration=result["duration"],
                processing_time=result["processing_time"],
                model_used=result["model_used"],
                request_id=request_id or generate_response_id(),
                timestamp=datetime.utcnow()
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
                    "default": self.default_recognizer
                },
                "synthesizers": {
                    "available": list(self.synthesizers.keys()),
                    "default": self.default_synthesizer
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理语音处理器资源")
        
        self.recognizers.clear()
        self.synthesizers.clear()
        self.default_recognizer = None
        self.default_synthesizer = None
        self.is_initialized = False
        
        logger.info("✅ 语音处理器资源清理完成")