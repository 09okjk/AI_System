"""
PPT处理模块 - PPT文件转换为数据文档
"""

import os
import io
import base64
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from pptx import Presentation
from pdf2image import convert_from_path
import time
import uuid
import subprocess
import platform
from openai import OpenAI
from PIL import Image

from .models import DataDocumentCreate, DataItemContent
from .logger import get_logger

class PPTProcessor:
    """PPT处理器类"""
    
    prompt_template = """
    讲解内容干净利落，不做过多的其他关系不大的讲解，只聚焦于当前页的内容进行讲解。
    以老师的口吻进行讲解，不需要开场白和结束语，直接进行讲解。
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.image_model = os.getenv("DASHSCOPE_IMAGE_MODEL", "qwen-vl-plus")
        
    async def process_ppt(self, file_content: bytes, file_name: str, prompt: str = prompt_template ) -> Dict[str, Any]:
        """
        处理PPT文件并生成数据文档内容
        
        Args:
            file_content: PPT文件内容
            file_name: 文件名
            prompt: 图片理解提示词
        
        Returns:
            数据文档创建对象
        """
        self.logger.info(f"开始处理PPT文件: {file_name}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        ppt_path = os.path.join(temp_dir, file_name)
        
        try:
            # 保存上传的文件
            with open(ppt_path, 'wb') as f:
                f.write(file_content)
            
            # 将PPT转换为图片
            images = await self._convert_ppt_to_images(ppt_path)
            
            # 处理每一张图片
            data_items = []
            for i, img_data in enumerate(images):
                # 调用图片理解API
                description = await self._analyze_image(img_data, prompt)
                
                # 创建数据项
                item = DataItemContent(
                    sequence=i+1,
                    text=description,
                    image=img_data,
                    image_filename=f"slide_{i+1}.png",
                    image_mimetype="image/png"
                )
                data_items.append(item)
            
            # 创建数据文档对象
            doc_name = os.path.splitext(file_name)[0]
            document = DataDocumentCreate(
                name=f"{doc_name} - 自动转换 - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description=f"由PPT文件 '{file_name}' 自动转换生成的数据文档",
                data_list=data_items,
                tags=["自动转换", "PPT导入"],
                metadata={
                    "source": "ppt_import",
                    "original_filename": file_name,
                    "slides_count": len(images),
                    "import_time": datetime.utcnow().isoformat(),
                    "prompt_used": prompt
                }
            )
            
            self.logger.info(f"PPT处理完成: {file_name}, 共{len(images)}张幻灯片")
            return document
            
        except Exception as e:
            self.logger.error(f"处理PPT文件失败: {e}")
            raise
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir)
    
    async def _convert_ppt_to_images(self, ppt_path: str) -> List[str]:
        """
        将PPT文件转换为图片列表
        
        Args:
            ppt_path: PPT文件路径
        
        Returns:
            图片Base64编码列表
        """
        images = []
        file_ext = os.path.splitext(ppt_path)[1].lower()
        
        try:
            if file_ext == '.pptx' or file_ext == '.ppt':
                # 先将PPT转换为PDF
                pdf_path = ppt_path + '.pdf'
                await self._convert_ppt_to_pdf(ppt_path, pdf_path)
                
                # 将PDF转换为图片
                pdf_images = convert_from_path(pdf_path, dpi=200)
                
                for img in pdf_images:
                    # 确保图片是RGB模式（防止透明通道导致WebP转换）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # 保存到内存中，强制使用PNG格式
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG', optimize=True)
                    img_byte_arr.seek(0)
                    
                    # 验证保存的图片格式是否正确
                    img_data = img_byte_arr.getvalue()
                    # PNG文件的前8个字节是固定的签名: 89 50 4E 47 0D 0A 1A 0A
                    if not img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                        self.logger.warning(f"检测到非PNG格式图片! 前10个字节: {img_data[:10].hex()}")
                        self.logger.info("尝试强制转换为PNG...")
                        # 重新加载并保存
                        temp_img = Image.open(io.BytesIO(img_data))
                        img_byte_arr = io.BytesIO()
                        temp_img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        img_data = img_byte_arr.getvalue()
                        
                        # 再次验证
                        if not img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                            self.logger.error(f"强制转换失败! 图片格式仍然不是PNG! 前10个字节: {img_data[:10].hex()}")
                    else:
                        self.logger.info("图片格式验证成功: PNG格式")
                    
                    # 转换为Base64
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    images.append(img_base64)
                
                # 删除临时PDF文件
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            return images
            
        except Exception as e:
            self.logger.error(f"PPT转换为图片失败: {e}")
            raise
    
    async def _convert_ppt_to_pdf(self, ppt_path: str, pdf_path: str):
        """
        将PPT转换为PDF（适用于Ubuntu系统）
        
        Args:
            ppt_path: PPT文件路径
            pdf_path: 输出PDF路径
        """
        try:
            # 方法1：使用LibreOffice (推荐)
            self.logger.info("尝试使用LibreOffice转换PPT...")
            result = subprocess.run([
                'libreoffice', 
                '--headless', 
                '--convert-to', 'pdf',
                '--outdir', os.path.dirname(pdf_path),
                ppt_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # LibreOffice 会创建一个与原文件同名但扩展名为.pdf的文件
                original_pdf = os.path.join(os.path.dirname(pdf_path), 
                                          os.path.splitext(os.path.basename(ppt_path))[0] + '.pdf')
                if os.path.exists(original_pdf):
                    shutil.move(original_pdf, pdf_path)
                    self.logger.info("✅ 使用LibreOffice转换成功")
                    return
            
            self.logger.warning(f"LibreOffice转换失败: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("LibreOffice转换超时")
        except FileNotFoundError:
            self.logger.warning("LibreOffice未安装，尝试其他方法...")
        except Exception as e:
            self.logger.warning(f"LibreOffice转换出错: {e}")
        
        try:
            # 方法2：使用unoconv
            self.logger.info("尝试使用unoconv转换PPT...")
            result = subprocess.run([
                'unoconv', '-f', 'pdf', '-o', pdf_path, ppt_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(pdf_path):
                self.logger.info("✅ 使用unoconv转换成功")
                return
            
            self.logger.warning(f"unoconv转换失败: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("unoconv转换超时")
        except FileNotFoundError:
            self.logger.warning("unoconv未安装")
        except Exception as e:
            self.logger.warning(f"unoconv转换出错: {e}")
        
        # 如果是Windows系统，尝试使用COM (仅作为最后的备用方案)
        if platform.system() == "Windows":
            try:
                self.logger.info("Windows系统，尝试使用COM转换...")
                import comtypes.client
                powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
                powerpoint.Visible = False
                slides = powerpoint.Presentations.Open(ppt_path)
                slides.SaveAs(pdf_path, 32)  # 32 表示PDF格式
                slides.Close()
                powerpoint.Quit()
                self.logger.info("✅ 使用COM转换成功")
                return
            except Exception as e:
                self.logger.warning(f"COM转换失败: {e}")
        
        # 所有方法都失败
        raise RuntimeError(
            "PPT转PDF失败。请确保系统已安装以下工具之一：\n"
            "1. LibreOffice: sudo apt install libreoffice\n"
            "2. unoconv: sudo apt install unoconv\n"
            "或者检查PPT文件是否损坏。"
        )
    
    async def _analyze_image(self, image_base64: str, prompt: str) -> str:
        """
        使用阿里云图片理解模型分析图片
        
        Args:
            image_base64: 图片的Base64编码
            prompt: 提示词
            
        Returns:
            图片描述文本
        """
        try:
            # 检查图片格式 - 如果不是以PNG头开始，可能是WebP
            image_data = base64.b64decode(image_base64)
            if not image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                self.logger.warning(f"发送到API的图片不是PNG格式! 前10个字节: {image_data[:10].hex()}")
                
                # 强制转换为PNG
                self.logger.info("尝试将图片转换为PNG...")
                img = Image.open(io.BytesIO(image_data))
                output = io.BytesIO()
                img.save(output, format='PNG')
                output.seek(0)
                png_data = output.getvalue()
                image_base64 = base64.b64encode(png_data).decode('utf-8')
                
                # 验证转换后的格式
                converted_data = base64.b64decode(image_base64)
                if not converted_data.startswith(b'\x89PNG\r\n\x1a\n'):
                    self.logger.error("图片转换失败，仍然不是PNG格式!")
                else:
                    self.logger.info("图片成功转换为PNG格式")
            
            # 对图片进行分析
            completion = self.client.chat.completions.create(
                model=self.image_model,
                messages=[{"role": "user","content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                        ]}]
            )
            
            # 提取回答
            description = completion.choices[0].message.content
            return description
            
        except Exception as e:
            self.logger.error(f"图片分析失败: {e}")
            # 失败时返回默认值
            return "图片分析失败，无法获取描述。"

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {}
        try:
            # 检查PPT处理依赖
            ppt_processor = PPTProcessor()
            ppt_status = {"healthy": True}
    
            # 检查关键依赖
            import importlib
            dependencies = ["pptx", "pdf2image"]
            missing = []
            for dep in dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing.append(dep)
    
            # 检查系统转换工具
            conversion_tools = []
            try:
                subprocess.run(['libreoffice', '--version'], 
                             capture_output=True, check=True)
                conversion_tools.append("LibreOffice")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            try:
                subprocess.run(['unoconv', '--version'], 
                             capture_output=True, check=True)
                conversion_tools.append("unoconv")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
    
            if not conversion_tools:
                ppt_status["healthy"] = False
                ppt_status["message"] = "未找到PPT转换工具 (LibreOffice/unoconv)"
            else:
                ppt_status["conversion_tools"] = conversion_tools
    
            if missing:
                ppt_status["healthy"] = False
                ppt_status["message"] = f"缺少Python依赖: {', '.join(missing)}"
    
            health_status["ppt_processor"] = ppt_status
        except Exception as e:
            health_status["ppt_processor"] = {
                "healthy": False,
                "message": f"PPT处理器检查失败: {str(e)}"
            }        
        return health_status
    
    async def initialize(self):
        """初始化PPT处理器"""
        try:
            # 检查必要依赖
            import pptx
            import pdf2image
            
            # 检查系统转换工具
            tools_available = []
            try:
                subprocess.run(['libreoffice', '--version'], 
                             capture_output=True, check=True, timeout=10)
                tools_available.append("LibreOffice")
            except:
                pass
            
            try:
                subprocess.run(['unoconv', '--version'], 
                             capture_output=True, check=True, timeout=10)
                tools_available.append("unoconv")
            except:
                pass
            
            if not tools_available:
                self.logger.warning("⚠️ 未检测到PPT转换工具，请安装LibreOffice或unoconv")
                self.logger.info("安装命令:")
                self.logger.info("  sudo apt update")
                self.logger.info("  sudo apt install libreoffice")
                self.logger.info("  # 或者")
                self.logger.info("  sudo apt install unoconv")
            else:
                self.logger.info(f"✅ 检测到转换工具: {', '.join(tools_available)}")
            
            # 检查模型连接
            if self.client:
                # 简单测试API连接
                test_response = self.client.models.list()
                if test_response:
                    self.logger.info("✅ 图像分析模型连接正常")
            
            self.logger.info("✅ PPT处理器初始化完成")
            return True
        except ImportError as e:
            self.logger.error(f"❌ PPT处理器初始化失败: 缺少依赖 - {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ PPT处理器初始化失败: {e}")
            return False
            
    async def cleanup(self):
        """清理PPT处理器资源"""
        try:
            # 如果有任何需要清理的资源，在这里处理
            # 例如临时文件或连接等
            self.logger.info("✅ PPT处理器资源已清理")
        except Exception as e:
            self.logger.error(f"❌ PPT处理器清理失败: {e}")