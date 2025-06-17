"""
MongoDB 数据管理模块
"""

import os
import io
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from bson import ObjectId
import asyncio

from .models import (
    DataDocumentCreate, DataDocumentUpdate, DataDocumentResponse,
    DataDocumentQuery, DataDocumentListResponse, DataItemContent,
    DataDocumentSearchResponse, DataStatisticsResponse
)
from .logger import get_logger

class MongoDBManager:
    """MongoDB 数据管理器"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection_name = "data_documents"
        self.logger = get_logger(__name__)
        
        # 配置
        self.host = os.getenv("MONGODB_HOST", "localhost")
        self.port = int(os.getenv("MONGODB_PORT", "27017"))
        self.database_name = os.getenv("MONGODB_DATABASE", "ai_system")
        self.username = os.getenv("MONGODB_USERNAME")
        self.password = os.getenv("MONGODB_PASSWORD")
        
    async def initialize(self):
        """初始化MongoDB连接"""
        try:
            # 构建连接字符串
            if self.username and self.password:
                connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
            else:
                connection_string = f"mongodb://{self.host}:{self.port}"
        
            self.logger.info(f"🔌 连接 MongoDB: {self.host}:{self.port}")
        
            # 创建客户端
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            
            # 选择数据库
            self.db = self.client[self.database_name]
            
            # 初始化GridFS
            self.fs = AsyncIOMotorGridFSBucket(self.db)
            
            # 测试连接
            await self.client.admin.command('ping')
            self.logger.info("✅ MongoDB 连接成功")
            
            # 创建索引
            await self._create_indexes()
            
            return True
            
        except ConnectionFailure as e:
            self.logger.error(f"❌ MongoDB 连接失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ MongoDB 初始化失败: {e}")
            raise
    
    async def _create_indexes(self):
        """创建索引"""
        try:
            collection = self.db[self.collection_name]
            
            # 创建索引
            await collection.create_index("name")
            await collection.create_index("tags")
            await collection.create_index("created_at")
            await collection.create_index("updated_at")
            await collection.create_index([("name", "text"), ("description", "text")])
            
            self.logger.info("✅ MongoDB 索引创建完成")
            
        except Exception as e:
            self.logger.error(f"❌ 创建索引失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        if self.client:
            self.client.close()
            self.logger.info("✅ MongoDB 连接已关闭")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self.client:
                return {"healthy": False, "error": "MongoDB 客户端未初始化"}
            
            # 测试连接
            await self.client.admin.command('ping')
            
            # 获取数据库状态
            stats = await self.db.command("dbStats")
            
            return {
                "healthy": True,
                "database": self.database_name,
                "collections": stats.get("collections", 0),
                "objects": stats.get("objects", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    # ==================== 数据文档管理 ====================
    
    async def create_document(self, document: DataDocumentCreate) -> DataDocumentResponse:
        """创建数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            # 检查名称是否重复
            existing = await collection.find_one({"name": document.name})
            if existing:
                raise ValueError(f"数据文档名称 '{document.name}' 已存在")
            
            # 准备文档数据
            doc_data = document.dict()
            
            # 处理图片数据，将大图片存储到GridFS
            modified_data_list = []
            for idx, item in enumerate(document.data_list):
                item_dict = item.dict()
                
                # 如果有图片数据，存储到GridFS
                if item.image:
                    try:
                        # 解码base64
                        image_data = base64.b64decode(item.image)
                        
                        # 生成文件名
                        filename = item.image_filename or f"image_{document.name}_{idx}.png"
                        
                        # 上传到GridFS
                        file_id = await self.fs.upload_from_stream(
                            filename,
                            io.BytesIO(image_data),
                            metadata={
                                "document_name": document.name,
                                "sequence": item.sequence,
                                "mimetype": item.image_mimetype or "image/png"
                            }
                        )
                        
                        # 替换image字段为GridFS文件ID引用
                        item_dict["image"] = None
                        item_dict["image_file_id"] = str(file_id)
                        self.logger.info(f"图片已存储到GridFS: {filename}, ID: {file_id}")
                    except Exception as e:
                        self.logger.error(f"存储图片到GridFS失败: {e}")
                        raise
                
                modified_data_list.append(item_dict)
            
            # 更新文档数据
            doc_data["data_list"] = modified_data_list
            doc_data.update({
                "_id": ObjectId(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "version": 1
            })
            
            # 插入文档
            result = await collection.insert_one(doc_data)
            
            # 获取插入的文档
            created_doc = await collection.find_one({"_id": result.inserted_id})
            
            # 转换为响应模型
            response_data = dict(created_doc)
            response_data["id"] = str(response_data.pop("_id"))
            
            self.logger.info(f"✅ 创建数据文档: {document.name}")
            return DataDocumentResponse(**response_data)
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"❌ 创建数据文档失败: {e}")
            raise
        
    async def get_document(self, document_id: str) -> DataDocumentResponse:
        """获取数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            # 查找文档
            document = await collection.find_one({"_id": ObjectId(document_id)})
            
            if not document:
                raise ValueError(f"未找到ID为 {document_id} 的数据文档")
            
            # 处理GridFS文件引用
            for item in document["data_list"]:
                if "image_file_id" in item and item["image_file_id"]:
                    try:
                        # 获取GridFS中的文件
                        grid_out = await self.fs.open_download_stream(ObjectId(item["image_file_id"]))
                        
                        # 读取文件内容
                        chunks = []
                        async for chunk in grid_out:
                            chunks.append(chunk)
                        
                        # 转换为base64
                        image_data = b''.join(chunks)
                        item["image"] = base64.b64encode(image_data).decode('utf-8')
                        
                        # 如果没有mimetype，从metadata中获取
                        if not item.get("image_mimetype") and hasattr(grid_out, "metadata") and grid_out.metadata:
                            item["image_mimetype"] = grid_out.metadata.get("mimetype", "image/png")
                    except Exception as e:
                        self.logger.error(f"从GridFS获取图片失败: {e}")
                        # 继续处理，但不包含图片
            
            # 转换为响应模型
            response_data = dict(document)
            response_data["id"] = str(response_data.pop("_id"))
            
            return DataDocumentResponse(**response_data)
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"❌ 获取数据文档失败: {e}")
            raise
    
    async def update_document(self, document_id: str, update_data: DataDocumentUpdate) -> Optional[DataDocumentResponse]:
        """更新数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            # 检查文档是否存在
            existing = await collection.find_one({"_id": ObjectId(document_id)})
            if not existing:
                return None
            
            # 如果更新名称，检查是否重复
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            if "name" in update_dict and update_dict["name"] != existing["name"]:
                name_exists = await collection.find_one({
                    "name": update_dict["name"],
                    "_id": {"$ne": ObjectId(document_id)}
                })
                if name_exists:
                    raise ValueError(f"数据文档名称 '{update_dict['name']}' 已存在")
            
            # 处理图片数据，将大图片存储到GridFS
            if "data_list" in update_dict and update_dict["data_list"]:
                modified_data_list = []
                for idx, item in enumerate(update_dict["data_list"]):
                    # 检查item是否为字典或Pydantic模型
                    if hasattr(item, 'dict'):
                        item_dict = item.dict()
                    else:
                        item_dict = item  # 如果已经是字典，则直接使用
                    
                    # 如果有图片数据，存储到GridFS
                    image_data = None
                    image_field = item_dict.get("image") if isinstance(item_dict, dict) else getattr(item, "image", None)
                    
                    if image_field:
                        try:
                            # 解码base64
                            image_data = base64.b64decode(image_field)
                            
                            # 生成文件名
                            doc_name = update_dict.get("name", existing["name"])
                            
                            # 获取文件名和序号
                            image_filename = item_dict.get("image_filename", "") if isinstance(item_dict, dict) else getattr(item, "image_filename", "")
                            sequence = item_dict.get("sequence", idx+1) if isinstance(item_dict, dict) else getattr(item, "sequence", idx+1)
                            mimetype = item_dict.get("image_mimetype", "image/png") if isinstance(item_dict, dict) else getattr(item, "image_mimetype", "image/png")
                            
                            # 使用提供的文件名或生成一个
                            filename = image_filename or f"image_{doc_name}_{idx}.png"
                            
                            # 上传到GridFS
                            file_id = await self.fs.upload_from_stream(
                                filename,
                                io.BytesIO(image_data),
                                metadata={
                                    "document_name": doc_name,
                                    "sequence": sequence,
                                    "mimetype": mimetype
                                }
                            )
                            
                            # 替换image字段为GridFS文件ID引用
                            item_dict["image"] = None
                            item_dict["image_file_id"] = str(file_id)
                            self.logger.info(f"图片已存储到GridFS: {filename}, ID: {file_id}")
                        except Exception as e:
                            self.logger.error(f"存储图片到GridFS失败: {e}")
                            raise
                    
                    modified_data_list.append(item_dict)
                
                # 更新data_list字段
                update_dict["data_list"] = modified_data_list
        
            # 更新文档
            update_dict.update({
                "updated_at": datetime.utcnow(),
                "version": existing["version"] + 1
            })
            
            result = await collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_dict}
            )
            
            if result.modified_count == 0:
                return None
            
            # 获取更新后的文档
            updated_doc = await collection.find_one({"_id": ObjectId(document_id)})
            
            # 转换为响应模型
            response_data = dict(updated_doc)
            response_data["id"] = str(response_data.pop("_id"))
            
            self.logger.info(f"✅ 更新数据文档: {document_id}")
            return DataDocumentResponse(**response_data)
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"❌ 更新数据文档失败: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """删除数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            result = await collection.delete_one({"_id": ObjectId(document_id)})
            
            if result.deleted_count > 0:
                self.logger.info(f"✅ 删除数据文档: {document_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 删除数据文档失败: {e}")
            raise
    
    async def list_documents(self, query: DataDocumentQuery) -> DataDocumentListResponse:
        """列出数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            # 构建查询条件
            filter_dict = {}
            
            if query.name:
                filter_dict["name"] = {"$regex": query.name, "$options": "i"}
            
            if query.tags:
                filter_dict["tags"] = {"$in": query.tags}
            
            # 计算总数
            total = await collection.count_documents(filter_dict)
            
            # 计算分页
            skip = (query.page - 1) * query.page_size
            total_pages = (total + query.page_size - 1) // query.page_size
            
            # 查询文档
            cursor = collection.find(filter_dict).sort(
                query.sort_by, query.sort_order
            ).skip(skip).limit(query.page_size)
            
            documents = []
            async for doc in cursor:
                response_data = dict(doc)
                response_data["id"] = str(response_data.pop("_id"))
                documents.append(DataDocumentResponse(**response_data))
            
            return DataDocumentListResponse(
                success=True,
                documents=documents,
                total=total,
                page=query.page,
                page_size=query.page_size,
                total_pages=total_pages
            )
            
        except Exception as e:
            self.logger.error(f"❌ 列出数据文档失败: {e}")
            raise
    
    async def search_documents(self, search_text: str, limit: int = 10) -> DataDocumentSearchResponse:
        """搜索数据文档"""
        try:
            collection = self.db[self.collection_name]
            
            start_time = datetime.utcnow()
            
            # 使用文本搜索
            cursor = collection.find(
                {"$text": {"$search": search_text}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            results = []
            async for doc in cursor:
                response_data = dict(doc)
                response_data["id"] = str(response_data.pop("_id"))
                # 移除score字段
                response_data.pop("score", None)
                results.append(DataDocumentResponse(**response_data))
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            return DataDocumentSearchResponse(
                success=True,
                results=results,
                total_matches=len(results),
                search_time=search_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ 搜索数据文档失败: {e}")
            raise
    
    async def get_statistics(self) -> DataStatisticsResponse:
        """获取数据统计"""
        try:
            collection = self.db[self.collection_name]
            
            # 聚合统计
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_documents": {"$sum": 1},
                        "total_items": {"$sum": {"$size": "$data_list"}},
                        "total_images": {
                            "$sum": {
                                "$size": {
                                    "$filter": {
                                        "input": "$data_list",
                                        "cond": {"$ne": ["$$this.image", None]}
                                    }
                                }
                            }
                        }
                    }
                }
            ]
            
            stats_cursor = collection.aggregate(pipeline)
            stats = await stats_cursor.to_list(length=1)
            
            if not stats:
                stats = [{
                    "total_documents": 0,
                    "total_items": 0,
                    "total_images": 0
                }]
            
            # 获取标签统计
            tag_pipeline = [
                {"$unwind": "$tags"},
                {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            tag_cursor = collection.aggregate(tag_pipeline)
            most_used_tags = []
            async for tag_stat in tag_cursor:
                most_used_tags.append({
                    "tag": tag_stat["_id"],
                    "count": tag_stat["count"]
                })
            
            # 获取最近活动
            recent_cursor = collection.find().sort("updated_at", -1).limit(5)
            recent_activity = []
            async for doc in recent_cursor:
                recent_activity.append({
                    "id": str(doc["_id"]),
                    "name": doc["name"],
                    "updated_at": doc["updated_at"],
                    "action": "updated"
                })
            
            # 获取存储大小
            db_stats = await self.db.command("dbStats")
            storage_size = f"{db_stats.get('storageSize', 0) / 1024 / 1024:.2f} MB"
            
            return DataStatisticsResponse(
                success=True,
                total_documents=stats[0]["total_documents"],
                total_items=stats[0]["total_items"],
                total_images=stats[0]["total_images"],
                storage_size=storage_size,
                most_used_tags=most_used_tags,
                recent_activity=recent_activity
            )
            
        except Exception as e:
            self.logger.error(f"❌ 获取数据统计失败: {e}")
            raise
    
    # ==================== 数据项管理 ====================
    
    async def add_data_item(self, document_id: str, item: DataItemContent) -> bool:
        """向文档添加数据项"""
        try:
            collection = self.db[self.collection_name]
            
            # 检查序号是否重复
            existing_doc = await collection.find_one({"_id": ObjectId(document_id)})
            if not existing_doc:
                raise ValueError("文档不存在")
            
            existing_sequences = [item["sequence"] for item in existing_doc.get("data_list", [])]
            if item.sequence in existing_sequences:
                raise ValueError(f"序号 {item.sequence} 已存在")
            
            # 添加数据项
            result = await collection.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$push": {"data_list": item.dict()},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$inc": {"version": 1}
                }
            )
            
            if result.modified_count > 0:
                self.logger.info(f"✅ 添加数据项到文档: {document_id}")
                return True
            return False
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"❌ 添加数据项失败: {e}")
            raise
    
    async def update_data_item(self, document_id: str, sequence: int, item: DataItemContent) -> bool:
        """更新文档中的数据项"""
        try:
            collection = self.db[self.collection_name]
            
            result = await collection.update_one(
                {"_id": ObjectId(document_id), "data_list.sequence": sequence},
                {
                    "$set": {
                        "data_list.$": item.dict(),
                        "updated_at": datetime.utcnow()
                    },
                    "$inc": {"version": 1}
                }
            )
            
            if result.modified_count > 0:
                self.logger.info(f"✅ 更新数据项: {document_id}, 序号: {sequence}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 更新数据项失败: {e}")
            raise
    
    async def delete_data_item(self, document_id: str, sequence: int) -> bool:
        """删除文档中的数据项"""
        try:
            collection = self.db[self.collection_name]
            
            result = await collection.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$pull": {"data_list": {"sequence": sequence}},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$inc": {"version": 1}
                }
            )
            
            if result.modified_count > 0:
                self.logger.info(f"✅ 删除数据项: {document_id}, 序号: {sequence}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 删除数据项失败: {e}")
            raise