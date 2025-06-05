# AI Agent 后端服务 v2.0

一个现代化的 AI 智能代理后端服务，基于 Python FastAPI 构建，专注于提供高性能的 API 接口服务。

## 🎯 项目特点

- **纯后端架构**: 专注于 API 服务，前后端完全分离
- **多模型支持**: 集成多种 LLM 提供商（DashScope、Xinference、OpenAI、Anthropic 等）
- **MCP 工具系统**: 支持 Model Context Protocol 工具扩展
- **语音处理**: 集成语音识别和语音合成功能（SensVoice、Whisper、CosyVoice、Edge TTS）
- **完整日志系统**: 结构化日志记录，运行步骤可视化
- **配置管理**: 灵活的配置系统，支持动态更新
- **测试框架**: 完整的测试套件，包括单元测试、集成测试和性能测试
- **智能启动**: 环境检查、依赖验证、配置管理一体化

## 🏗️ 架构设计

```
AI_Agent_Backend/
├── main.py                  # FastAPI 应用定义
├── start_server.py         # 启动脚本（推荐使用）
├── test_runner.py          # 测试运行器
├── requirements.txt        # 依赖包列表
├── .env.example           # 环境变量示例
├── src/
│   ├── __init__.py
│   ├── models.py          # 数据模型定义
│   ├── logger.py          # 日志系统
│   ├── config.py          # 配置管理
│   ├── utils.py           # 工具函数
│   ├── mcp.py             # MCP管理器
│   ├── llm.py             # LLM管理器
│   └── speech.py          # 语音处理器
├── config/                 # 配置文件目录
│   ├── mcp_configs.json
│   ├── llm_configs.json
│   └── app_config.json
├── tools/                  # MCP 工具目录
│   └── math_server.py     # 数学计算工具示例
├── logs/                   # 日志文件目录
├── tests/                  # 测试文件目录
└── assets/                 # 资源文件目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 conda

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/09okjk/AI_System.git
cd AI_System

# 安装核心依赖
pip install -r requirements.txt
```

### 环境变量配置

1. 复制示例环境文件：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，配置您的 API 密钥：

```env
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
```

### 启动服务

**推荐方式（使用 start_server.py）：**

```bash
# 首次运行 - 检查环境
python start_server.py --check-only

# 开发模式（带热重载）
python start_server.py --reload

# 生产模式
python start_server.py --host 0.0.0.0 --port 8000 --workers 4
```

**或者直接启动（不推荐）：**

```bash
python main.py --reload
```

服务启动后，访问以下地址：

- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/health

## 📚 API 接口文档

### 系统状态

- `GET /api/health` - 健康检查
- `GET /api/status` - 系统详细状态

### MCP 工具管理

- `GET /api/mcp/configs` - 获取所有 MCP 配置
- `POST /api/mcp/configs` - 创建 MCP 配置
- `PUT /api/mcp/configs/{id}` - 更新 MCP 配置
- `DELETE /api/mcp/configs/{id}` - 删除 MCP 配置
- `POST /api/mcp/configs/{id}/test` - 测试 MCP 配置

### LLM 模型管理

- `GET /api/llm/configs` - 获取所有 LLM 配置
- `POST /api/llm/configs` - 创建 LLM 配置
- `PUT /api/llm/configs/{id}` - 更新 LLM 配置
- `DELETE /api/llm/configs/{id}` - 删除 LLM 配置
- `POST /api/llm/configs/{id}/test` - 测试 LLM 配置

### 语音处理

- `POST /api/speech/recognize` - 语音识别（支持 SensVoice、Whisper）
- `POST /api/speech/synthesize` - 语音合成（支持 CosyVoice、Edge TTS）

### 对话功能

- `POST /api/chat/text` - 文本对话
- `POST /api/chat/stream` - 流式对话
- `POST /api/chat/voice` - 语音对话（语音输入 + 文本和语音输出）

## 🧪 测试系统

### 运行所有测试

```bash
python test_runner.py --test-type all
```

### 运行特定类型测试

```bash
# 功能测试
python test_runner.py --test-type integration

# 性能测试
python test_runner.py --test-type performance --concurrent-users 20

# 生成测试报告
python test_runner.py --output test_report.json
```

## 📋 配置管理

### MCP 工具配置示例

```json
{
  "name": "数学计算工具",
  "description": "提供基本的数学运算功能",
  "command": "python",
  "args": ["tools/math_server.py"],
  "transport": "stdio",
  "auto_start": false,
  "restart_on_failure": true,
  "timeout": 30
}
```

### LLM 模型配置示例

```json
{
  "name": "千问大模型",
  "provider": "dashscope",
  "model_name": "qwen-plus",
  "api_key": "your_api_key",
  "temperature": 0.7,
  "max_tokens": 2000,
  "is_default": true,
  "enabled": true
}
```

## 📊 日志系统

### 日志级别

- **DEBUG**: 调试信息
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 日志格式

```json
{
  "timestamp": "2025-06-05T02:48:25.000Z",
  "level": "INFO",
  "logger": "main",
  "message": "🚀 AI Agent Backend 正在启动...",
  "request_id": "uuid-string",
  "extra": {
    "api_call": {
      "method": "POST",
      "endpoint": "/api/chat/text",
      "status_code": 200,
      "processing_time": 1.23
    }
  }
}
```

### 查看日志

```bash
# 实时查看应用日志
tail -f logs/ai_agent.log

# 查看错误日志
tail -f logs/ai_agent_error.log

# 使用 jq 格式化查看
tail -f logs/ai_agent.log | jq '.'
```

## 🔧 开发指南

### 环境检查

在开发前，建议运行环境检查：

```bash
python start_server.py --check-only
```

### 添加新的 API 接口

1. 在 `src/models.py` 中定义请求/响应模型
2. 在 `main.py` 中添加路由处理函数
3. 添加相应的日志记录
4. 编写单元测试

### 集成新的 LLM 提供商

1. 在 `src/llm.py` 中添加新的 LLM 客户端类
2. 在 `src/models.py` 中更新 `LLMProvider` 枚举
3. 更新配置验证逻辑
4. 添加测试用例

### 添加新的 MCP 工具

1. 在 `tools/` 目录下编写 MCP 服务器脚本
2. 通过 API 或配置文件添加 MCP 配置
3. 测试工具功能
4. 更新文档

### 语音处理扩展

支持的语音引擎：

- **语音识别**: SensVoice (FunASR)、OpenAI Whisper
- **语音合成**: CosyVoice、Edge TTS

添加新引擎需要：

1. 在 `src/speech.py` 中实现相应的处理器类
2. 更新初始化逻辑
3. 添加配置选项

## 🐛 故障排除

### 常见问题

1. **服务启动失败**

   ```bash
   # 检查依赖
   python start_server.py --check-only
   
   # 查看详细错误
   python start_server.py --log-level DEBUG
   ```

2. **模型调用失败**
   - 检查 API 密钥配置：`cat .env | grep API_KEY`
   - 验证网络连接
   - 查看模型配置：`curl http://localhost:8000/api/llm/configs`

3. **MCP 工具无法启动**
   - 检查工具脚本权限：`ls -la tools/`
   - 测试工具配置：`curl -X POST http://localhost:8000/api/mcp/configs/{id}/test`
   - 查看 MCP 日志

4. **语音处理失败**
   - 检查模型文件路径
   - 验证音频格式支持
   - 查看语音处理器状态

### 调试技巧

```bash
# 启用调试模式
python start_server.py --log-level DEBUG

# 检查系统状态
curl http://localhost:8000/api/health

# 查看详细状态
curl http://localhost:8000/api/status

# 测试特定功能
python test_runner.py --test-type integration
```

## 📈 性能优化

### 推荐配置

**开发环境**:

```bash
python start_server.py --reload --log-level DEBUG
```

**生产环境**:

```bash
python start_server.py --workers 4 --log-level INFO
```

**容器化部署**:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "start_server.py", "--host", "0.0.0.0", "--workers", "4"]
```

### 监控指标

- API 响应时间
- 错误率和成功率
- 并发用户数
- 内存使用率
- CPU 使用率
- 语音处理延迟
- MCP 工具状态

## 🚀 部署指南

### Docker 部署

```bash
# 构建镜像
docker build -t ai-agent-backend .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env ai-agent-backend
```

### 系统服务部署

```bash
# 创建系统服务
sudo systemctl edit --force --full ai-agent-backend.service

# 启动服务
sudo systemctl enable ai-agent-backend
sudo systemctl start ai-agent-backend
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 代码规范

- 使用 Black 进行代码格式化
- 遵循 PEP 8 规范
- 添加适当的类型注解
- 编写完整的文档字符串

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

如有问题，请：

1. 查看文档和 FAQ
2. 运行环境检查：`python start_server.py --check-only`
3. 搜索现有的 Issues
4. 创建新的 Issue 并提供详细信息
5. 联系维护者：[@09okjk](https://github.com/09okjk)

## 📝 更新日志

### v2.0.0 (2025-06-05)

- 🎉 重构为纯后端架构
- ✨ 添加 MCP 工具系统支持
- 🎤 集成语音识别和合成功能
- 📊 实现结构化日志记录
- 🧪 完善测试框架
- 🔧 优化启动脚本和环境检查
- 📚 完善 API 文档

---

**Made with ❤️ by [09okjk](https://github.com/09okjk)**

## 🔄 主要更新内容

1. **更新了项目结构** - 反映了实际的文件组织
2. **强调使用 start_server.py** - 作为推荐的启动方式
3. **完善了环境配置说明** - 详细的环境变量配置步骤
4. **添加了语音处理详情** - 具体支持的引擎和配置
5. **丰富了故障排除部分** - 更详细的调试步骤
6. **添加了部署指南** - Docker 和系统服务部署
7. **完善了贡献指南** - 代码规范和提交流程
8. **添加了更新日志** - 版本变更记录
9. **修正了启动命令** - 使用正确的启动脚本
10. **添加了性能优化建议** - 开发和生产环境配置

这个更新后的 README 更准确地反映了您项目的实际架构和使用方式，特别是强调了 `start_server.py` 作为推荐启动方式的重要性。
