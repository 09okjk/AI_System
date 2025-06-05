#!/usr/bin/env python3
"""
AI Agent Backend 启动脚本 - 改进版
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查依赖"""
    required_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('uvicorn', 'ASGI server'),
        ('aiofiles', 'Async file operations'),
        ('aiohttp', 'Async HTTP client'),
        ('pydantic', 'Data validation'),
        ('psutil', 'System monitoring')
    ]
    
    optional_packages = [
        ('funasr', 'SensVoice/FunASR speech recognition'),
        ('whisper', 'OpenAI Whisper speech recognition'),
        ('edge_tts', 'Edge TTS speech synthesis'),
        ('openai', 'OpenAI API client'),
        ('anthropic', 'Anthropic API client')
    ]
    
    missing_required = []
    available_optional = []
    
    # 检查必需依赖
    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(f"  - {package}: {description}")
    
    # 检查可选依赖
    for package, description in optional_packages:
        try:
            __import__(package)
            available_optional.append(f"  ✅ {package}: {description}")
        except ImportError:
            available_optional.append(f"  ❌ {package}: {description} (未安装)")
    
    if missing_required:
        print("❌ 缺少必需依赖:")
        for dep in missing_required:
            print(dep)
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 核心依赖检查通过")
    
    print("\n📦 可选依赖状态:")
    for dep in available_optional:
        print(dep)
    
    return True

def setup_environment():
    """设置环境"""
    # 创建必要的目录
    directories = ["logs", "config", "tools", "tests", "assets"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 目录已准备: {directory}")
    
    # 检查配置文件
    config_files = [
        "config/app_config.json",
        "config/llm_configs.json", 
        "config/mcp_configs.json"
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"⚠️  配置文件不存在: {config_file}")
            print("  将在首次运行时自动创建默认配置")
    
    # 检查环境变量
    env_status = {
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),
        "XINFERENCE_ENDPOINT": os.getenv("XINFERENCE_ENDPOINT", "http://localhost:9997/v1"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COSYVOICE_MODEL_DIR": os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/CosyVoice2-0.5B"),
        "WHISPER_MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "base"),
        "SPEECH_DEVICE": os.getenv("SPEECH_DEVICE", "cpu")
    }
    
    print("\n🔧 环境变量状态:")
    for key, value in env_status.items():
        if value:
            masked_value = value if key not in ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"] else "*" * 10
            print(f"  ✅ {key}: {masked_value}")
        else:
            print(f"  ❌ {key}: 未设置")

def create_sample_env():
    """创建示例环境文件"""
    env_content = """# AI Agent Backend 环境变量配置

# DashScope API 配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Xinference 配置
XINFERENCE_ENDPOINT=http://localhost:9997/v1

# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here

# 语音模型配置
COSYVOICE_MODEL_DIR=pretrained_models/CosyVoice2-0.5B
WHISPER_MODEL_SIZE=base
SPEECH_DEVICE=cpu

# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env.example")
    if not env_file.exists():
        env_file.write_text(env_content)
        print(f"📝 创建了示例环境文件: {env_file}")
        print("  请复制为 .env 并配置相应的值")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Agent Backend Server")
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "0.0.0.0"), help="监听主机")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8000")), help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"), help="日志级别")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境，不启动服务")
    
    args = parser.parse_args()
    
    print("🚀 AI Agent Backend Server")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 创建示例环境文件
    create_sample_env()
    
    if args.check_only:
        print("\n✅ 环境检查完成")
        return
    
    # 启动服务器
    try:
        import uvicorn
        
        print(f"\n🌟 服务器配置:")
        print(f"  - 主机: {args.host}")
        print(f"  - 端口: {args.port}")
        print(f"  - 开发模式: {args.reload}")
        print(f"  - 日志级别: {args.log_level}")
        print(f"  - API文档: http://{args.host}:{args.port}/docs")
        print(f"  - 健康检查: http://{args.host}:{args.port}/api/health")
        print("\n" + "=" * 50)
        
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()