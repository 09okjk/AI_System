"""
MongoDB 数据管理接口模块
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import base64
import magic
from src.models import (
    DataDocumentCreate, DataDocumentUpdate, DataDocumentResponse,
    DataDocumentListResponse, DataDocumentQuery, DataDocumentSearchResponse,
    DataStatisticsResponse, DataItemContent
)
from src.utils import generate_response_id

router = APIRouter()

# 全局变量引用
def get_managers():
    """获取全局管理器实例"""
    from main import mongodb_manager, logger
    return {
        'mongodb_manager': mongodb_manager,
        'logger': logger
    }

# ==================== 数据文档管理接口 ====================

@router.post("/api/data/documents", response_model=DataDocumentResponse)
async def create_document(document: DataDocumentCreate):
    """创建数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📄 创建数据文档 [请求ID: {request_id}] - 名称: {document.name}")
    
    try:
        result = await managers['mongodb_manager'].create_document(document)
        logger.info(f"✅ 数据文档创建成功 [请求ID: {request_id}]: {result.id}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ 数据文档创建失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 数据文档创建异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/data/documents/{document_id}", response_model=DataDocumentResponse)
async def get_document(document_id: str):
    """获取数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📄 获取数据文档 [请求ID: {request_id}] - ID: {document_id}")
    
    try:
        result = await managers['mongodb_manager'].get_document(document_id)
        if not result:
            raise HTTPException(status_code=404, detail="数据文档不存在")
        
        logger.info(f"✅ 数据文档获取成功 [请求ID: {request_id}]")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 数据文档获取异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/data/documents/{document_id}", response_model=DataDocumentResponse)
async def update_document(document_id: str, update_data: DataDocumentUpdate):
    """更新数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📄 更新数据文档 [请求ID: {request_id}] - ID: {document_id}")
    
    try:
        result = await managers['mongodb_manager'].update_document(document_id, update_data)
        if not result:
            raise HTTPException(status_code=404, detail="数据文档不存在")
        
        logger.info(f"✅ 数据文档更新成功 [请求ID: {request_id}]")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ 数据文档更新失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 数据文档更新异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/data/documents/{document_id}")
async def delete_document(document_id: str):
    """删除数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📄 删除数据文档 [请求ID: {request_id}] - ID: {document_id}")
    
    try:
        success = await managers['mongodb_manager'].delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="数据文档不存在")
        
        logger.info(f"✅ 数据文档删除成功 [请求ID: {request_id}]")
        return {"message": f"数据文档 {document_id} 已删除", "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 数据文档删除异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/data/documents", response_model=DataDocumentListResponse)
async def list_documents(
    name: Optional[str] = None,
    tags: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    sort_by: str = "created_at",
    sort_order: int = -1
):
    """列出数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📄 列出数据文档 [请求ID: {request_id}] - 页码: {page}")
    
    try:
        # 解析标签
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        query = DataDocumentQuery(
            name=name,
            tags=tag_list,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        result = await managers['mongodb_manager'].list_documents(query)
        logger.info(f"✅ 数据文档列出成功 [请求ID: {request_id}] - 总数: {result.total}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 数据文档列出异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/data/documents/search", response_model=DataDocumentSearchResponse)
async def search_documents(q: str, limit: int = 10):
    """搜索数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"🔍 搜索数据文档 [请求ID: {request_id}] - 关键词: {q}")
    
    try:
        result = await managers['mongodb_manager'].search_documents(q, limit)
        logger.info(f"✅ 数据文档搜索成功 [请求ID: {request_id}] - 结果: {result.total_matches}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 数据文档搜索异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/data/statistics", response_model=DataStatisticsResponse)
async def get_statistics():
    """获取数据统计"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📊 获取数据统计 [请求ID: {request_id}]")
    
    try:
        result = await managers['mongodb_manager'].get_statistics()
        logger.info(f"✅ 数据统计获取成功 [请求ID: {request_id}]")
        return result
        
    except Exception as e:
        logger.error(f"❌ 数据统计获取异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 数据项管理接口 ====================

@router.post("/api/data/documents/{document_id}/items")
async def add_data_item(document_id: str, item: DataItemContent):
    """向文档添加数据项"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📝 添加数据项 [请求ID: {request_id}] - 文档ID: {document_id}")
    
    try:
        success = await managers['mongodb_manager'].add_data_item(document_id, item)
        if not success:
            raise HTTPException(status_code=404, detail="文档不存在或添加失败")
        
        logger.info(f"✅ 数据项添加成功 [请求ID: {request_id}]")
        return {"message": "数据项添加成功", "success": True}
        
    except ValueError as e:
        logger.error(f"❌ 数据项添加失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 数据项添加异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/data/documents/{document_id}/items/{sequence}")
async def update_data_item(document_id: str, sequence: int, item: DataItemContent):
    """更新文档中的数据项"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📝 更新数据项 [请求ID: {request_id}] - 文档ID: {document_id}, 序号: {sequence}")
    
    try:
        success = await managers['mongodb_manager'].update_data_item(document_id, sequence, item)
        if not success:
            raise HTTPException(status_code=404, detail="文档或数据项不存在")
        
        logger.info(f"✅ 数据项更新成功 [请求ID: {request_id}]")
        return {"message": "数据项更新成功", "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 数据项更新异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/data/documents/{document_id}/items/{sequence}")
async def delete_data_item(document_id: str, sequence: int):
    """删除文档中的数据项"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📝 删除数据项 [请求ID: {request_id}] - 文档ID: {document_id}, 序号: {sequence}")
    
    try:
        success = await managers['mongodb_manager'].delete_data_item(document_id, sequence)
        if not success:
            raise HTTPException(status_code=404, detail="文档或数据项不存在")
        
        logger.info(f"✅ 数据项删除成功 [请求ID: {request_id}]")
        return {"message": "数据项删除成功", "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 数据项删除异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 图片上传接口 ====================

@router.post("/api/data/upload-image")
async def upload_image(image: UploadFile = File(...)):
    """上传图片并返回base64编码"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"🖼️ 上传图片 [请求ID: {request_id}] - 文件: {image.filename}")
    
    try:
        # 验证文件类型
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")
        
        # 读取图片数据
        image_data = await image.read()
        
        # 限制文件大小 (5MB)
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片文件大小不能超过5MB")
        
        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        logger.info(f"✅ 图片上传成功 [请求ID: {request_id}] - 大小: {len(image_data)} bytes")
        
        return {
            "success": True,
            "message": "图片上传成功",
            "image_data": image_base64,
            "filename": image.filename,
            "mimetype": image.content_type,
            "size": len(image_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 图片上传异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PPT上传接口 ====================
from fastapi.background import BackgroundTasks

# ==================== PPT上传接口 ====================

@router.post("/api/data/ppt-import")
async def import_ppt_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(None)  # 允许用户自定义提示词
):
    """导入PPT文件并转换为数据文档"""
    managers = get_managers()
    logger = managers['logger']
    
    request_id = generate_response_id()
    logger.info(f"📊 导入PPT文件 [请求ID: {request_id}] - 文件: {file.filename}")
    
    try:
        # 检查文件类型
        if not file.filename.lower().endswith(('.ppt', '.pptx')):
            raise HTTPException(status_code=400, detail="只支持PPT和PPTX格式的文件")
            
        # 读取文件内容
        file_content = await file.read()
        
        # 限制文件大小 (50MB，因为需要处理图像)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过50MB")
        
        # 导入PPT处理器
        from src.ppt_processor import PPTProcessor
        ppt_processor = PPTProcessor()
        
        # 初始化处理器
        init_success = await ppt_processor.initialize()
        if not init_success:
            raise HTTPException(status_code=500, detail="PPT处理器初始化失败")
        
        # 使用用户提供的prompt，如果没提供则使用默认的
        if not prompt:
            prompt = ppt_processor.prompt_template
        
        logger.info(f"开始处理PPT文件，使用提示词: {prompt[:50]}...")
        
        # 处理PPT文件
        document = await ppt_processor.process_ppt(
            file_content, 
            file.filename,
            prompt
        )
        
        # 创建数据文档
        result = await managers['mongodb_manager'].create_document(document)
        
        logger.info(f"✅ PPT导入成功 [请求ID: {request_id}] - ID: {result.id}, 幻灯片数: {len(document.data_list)}")
        
        # 清理处理器资源
        await ppt_processor.cleanup()
        
        return {
            "success": True,
            "message": "PPT导入成功并转换为数据文档",
            "document_id": result.id,
            "document_name": result.name,
            "slides_count": len(document.data_list),
            "processing_method": "ai_image_analysis"
        }
            
    except ValueError as e:
        logger.error(f"❌ PPT导入失败 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PPT导入异常 [请求ID: {request_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PPT导入处理失败: {str(e)}")

# 添加PPT处理器健康检查接口
@router.get("/api/data/ppt-processor/health")
async def check_ppt_processor_health():
    """检查PPT处理器健康状态"""
    try:
        from src.ppt_processor import PPTProcessor
        ppt_processor = PPTProcessor()
        health_status = await ppt_processor.health_check()
        return health_status
    except Exception as e:
        return {
            "ppt_processor": {
                "healthy": False,
                "message": f"健康检查失败: {str(e)}"
            }
        }