#!/usr/bin/env python3
"""
AI Agent Backend 启动脚本
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查依赖"""
    try:
        import fastapi
        import uvicorn
        import aiofiles
        import aiohttp
        print("✅ 核心依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def setup_environment():
    """设置环境"""
    # 创建必要的目录
    directories = ["logs", "config", "tools", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 目录已准备: {directory}")
    
    # 检查环境变量
    env_vars = {
        "DASHSCOPE_API_KEY": "DashScope API密钥",
        "XINFERENCE_ENDPOINT": "Xinference端点"
    }
    
    missing_vars = []
    for var, desc in env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {desc}")
    
    if missing_vars:
        print("⚠️  以下环境变量未设置（可选）:")
        for var in missing_vars:
            print(var)
        print("可以在 .env 文件中设置这些变量")

def main():
    """主函数"""
    print("🚀 启动 AI Agent Backend Server")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 导入并启动服务器
    try:
        from main import app
        import uvicorn
        
        print("\n🌟 服务器配置:")
        print(f"  - 主机: 0.0.0.0")
        print(f"  - 端口: 8000")
        print(f"  - API文档: http://localhost:8000/docs")
        print(f"  - 健康检查: http://localhost:8000/api/health")
        print("\n" + "=" * 50)
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()