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
from openai import OpenAI

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
                # 先将PPT转换为PDF（可以使用其他方法）
                pdf_path = ppt_path + '.pdf'
                self._convert_ppt_to_pdf(ppt_path, pdf_path)
                
                # 将PDF转换为图片
                pdf_images = convert_from_path(pdf_path, dpi=200)
                
                for img in pdf_images:
                    # 保存到内存中
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # 转换为Base64
                    img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                    images.append(img_base64)
                
                # 删除临时PDF文件
                os.remove(pdf_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            return images
            
        except Exception as e:
            self.logger.error(f"PPT转换为图片失败: {e}")
            raise
    
    def _convert_ppt_to_pdf(self, ppt_path: str, pdf_path: str):
        """
        将PPT转换为PDF（需要系统安装LibreOffice或unoconv）
        
        Args:
            ppt_path: PPT文件路径
            pdf_path: 输出PDF路径
        """
        try:
            # 使用unoconv或其他工具（这里需要系统安装该工具）
            import subprocess
            subprocess.call(['unoconv', '-f', 'pdf', '-o', pdf_path, ppt_path])
        except Exception:
            # 备用方案：使用comtypes(仅限Windows且需要安装Microsoft PowerPoint)
            import comtypes.client
            powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
            powerpoint.Visible = True
            slides = powerpoint.Presentations.Open(ppt_path)
            slides.SaveAs(pdf_path, 32)  # 32 表示PDF格式
            slides.Close()
            powerpoint.Quit()
    
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
    
            # 可选：检查关键依赖
            import importlib
            dependencies = ["pptx", "pdf2image"]
            missing = []
            for dep in dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing.append(dep)
    
            if missing:
                ppt_status["healthy"] = False
                ppt_status["message"] = f"缺少依赖: {', '.join(missing)}"
    
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