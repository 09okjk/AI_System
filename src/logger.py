"""
日志配置和管理模块
提供结构化日志记录功能
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from contextvars import ContextVar

# 请求ID上下文变量
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 基础日志信息
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加请求ID（如果存在）
        request_id = request_id_var.get('')
        if request_id:
            log_entry['request_id'] = request_id
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 添加自定义字段
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
        
        return json.dumps(log_entry, ensure_ascii=False, indent=2 if self.is_console else None)
    
    def __init__(self, is_console: bool = False):
        super().__init__()
        self.is_console = is_console

class ColoredConsoleFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # 获取颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化时间
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # 获取请求ID
        request_id = request_id_var.get('')
        request_id_str = f"[{request_id[:8]}]" if request_id else ""
        
        # 构建日志消息
        parts = [
            f"{color}{record.levelname:<8}{reset}",
            f"{timestamp}",
            f"{record.name}",
            f"{request_id_str}",
            record.getMessage()
        ]
        
        # 过滤空字符串
        parts = [part for part in parts if part.strip()]
        
        log_message = " ".join(parts)
        
        # 添加异常信息
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message

class RequestContextFilter(logging.Filter):
    """请求上下文过滤器"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 添加请求ID到日志记录
        record.request_id = request_id_var.get('')
        return True

def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "ai_agent",
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """设置日志配置"""
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 创建请求上下文过滤器
    context_filter = RequestContextFilter()
    
    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredConsoleFormatter())
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    if enable_file:
        # 应用日志文件
        app_log_file = log_path / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)
        
        # 错误日志文件
        error_log_file = log_path / f"{app_name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)
    
    # 设置第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    logging.info(f"📋 日志系统初始化完成 - 级别: {log_level}, 目录: {log_path}")

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)

def set_request_id(request_id: str):
    """设置当前请求ID"""
    request_id_var.set(request_id)

def get_request_id() -> str:
    """获取当前请求ID"""
    return request_id_var.get('')

def log_api_call(
    logger: logging.Logger,
    method: str,
    endpoint: str,
    request_data: Optional[Dict[str, Any]] = None,
    response_data: Optional[Dict[str, Any]] = None,
    status_code: Optional[int] = None,
    processing_time: Optional[float] = None
):
    """记录API调用日志"""
    
    extra_data = {
        'api_call': {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'processing_time': processing_time
        }
    }
    
    if request_data:
        extra_data['api_call']['request'] = request_data
    
    if response_data:
        extra_data['api_call']['response'] = response_data
    
    # 创建带有额外数据的日志记录
    log_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"API {method} {endpoint}",
        (),
        None
    )
    log_record.extra_data = extra_data
    
    logger.handle(log_record)

def log_mcp_operation(
    logger: logging.Logger,
    operation: str,
    tool_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
):
    """记录MCP操作日志"""
    
    extra_data = {
        'mcp_operation': {
            'operation': operation,
            'tool_name': tool_name,
            'status': status
        }
    }
    
    if details:
        extra_data['mcp_operation']['details'] = details
    
    log_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"MCP {operation}: {tool_name} - {status}",
        (),
        None
    )
    log_record.extra_data = extra_data
    
    logger.handle(log_record)

def log_llm_call(
    logger: logging.Logger,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    processing_time: float,
    success: bool,
    error: Optional[str] = None
):
    """记录LLM调用日志"""
    
    extra_data = {
        'llm_call': {
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'processing_time': processing_time,
            'success': success
        }
    }
    
    if error:
        extra_data['llm_call']['error'] = error
    
    log_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"LLM调用: {model_name} - {'成功' if success else '失败'}",
        (),
        None
    )
    log_record.extra_data = extra_data
    
    logger.handle(log_record)

def log_speech_operation(
    logger: logging.Logger,
    operation: str,  # 'recognition' or 'synthesis'
    model_name: str,
    input_size: int,
    output_size: int,
    processing_time: float,
    success: bool,
    language: Optional[str] = None,
    error: Optional[str] = None
):
    """记录语音操作日志"""
    
    extra_data = {
        'speech_operation': {
            'operation': operation,
            'model_name': model_name,
            'input_size': input_size,
            'output_size': output_size,
            'processing_time': processing_time,
            'success': success
        }
    }
    
    if language:
        extra_data['speech_operation']['language'] = language
    
    if error:
        extra_data['speech_operation']['error'] = error
    
    log_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"语音{operation}: {model_name} - {'成功' if success else '失败'}",
        (),
        None
    )
    log_record.extra_data = extra_data
    
    logger.handle(log_record)

# 中间件示例（用于自动设置请求ID）
class LoggingMiddleware:
    """日志中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # 生成请求ID
            import uuid
            request_id = str(uuid.uuid4())
            set_request_id(request_id)
            
            # 记录请求开始
            logger = get_logger("middleware")
            logger.info(f"🔄 请求开始 [{request_id}] {scope['method']} {scope['path']}")
        
        await self.app(scope, receive, send)