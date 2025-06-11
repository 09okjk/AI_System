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