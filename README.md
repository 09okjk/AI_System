# AI Agent 后端服务 v2.0

一个现代化的 AI 智能代理后端服务，基于 Python FastAPI 构建，专注于提供高性能的 API 接口服务。

## 🎯 项目特点

- **纯后端架构**: 专注于 API 服务，前后端完全分离
- **多模型支持**: 集成多种 LLM 提供商（DashScope、Xinference、OpenAI 等）
- **MCP 工具系统**: 支持 Model Context Protocol 工具扩展
- **语音处理**: 集成语音识别和语音合成功能
- **完整日志系统**: 结构化日志记录，运行步骤可视化
- **配置管理**: 灵活的配置系统，支持动态更新
- **测试框架**: 完整的测试套件，包括单元测试、集成测试和性能测试

## 🏗️ 架构设计

```
AI_Agent_Backend/
├── main.py                  # 主服务入口
├── test_runner.py          # 测试运行器
├── requirements.txt        # 依赖包列表
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
├── logs/                   # 日志文件目录
└── tests/                  # 测试文件目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 conda

### 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd AI_Agent_Backend

# 安装依赖
pip install -r requirements.txt
```

### 环境变量配置

创建 `.env` 文件：

```env
# DashScope API 配置
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Xinference 配置
XINFERENCE_ENDPOINT=http://localhost:9997/v1

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs

# 服务配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### 启动服务

```bash
# 开发模式（带热重载）
python main.py --reload

# 生产模式
python main.py --host 0.0.0.0 --port 8000
```

服务启动后，访问以下地址：

- API 文档: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/api/health

## 📚 API 接口文档

### 系统状态

- `GET /api/health` - 健康检查
- `GET /api/status` - 系统状态

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

- `POST /api/speech/recognize` - 语音识别
- `POST /api/speech/synthesize` - 语音合成

### 对话功能

- `POST /api/chat/text` - 文本对话
- `POST /api/chat/stream` - 流式对话
- `POST /api/chat/voice` - 语音对话

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
  "name": "math_calculator",
  "description": "数学计算工具",
  "command": "python",
  "args": ["tools/math_server.py"],
  "transport": "stdio",
  "auto_start": true,
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
  "is_default": true
}
```

## 📊 日志系统

### 日志级别

- DEBUG: 调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

### 日志格式

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "logger": "main",
  "message": "服务启动成功",
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

### 添加新的 API 接口

1. 在 `src/models.py` 中定义请求/响应模型
2. 在 `main.py` 中添加路由处理函数
3. 添加相应的日志记录
4. 编写单元测试

### 集成新的 LLM 提供商

1. 在 `src/llm.py` 中添加新的 LLM 客户端
2. 在 `src/models.py` 中更新 `LLMProvider` 枚举
3. 更新配置验证逻辑
4. 添加测试用例

### 添加新的 MCP 工具

1. 编写 MCP 服务器脚本
2. 通过 API 添加 MCP 配置
3. 测试工具功能
4. 更新文档

## 🐛 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口是否被占用
   - 验证环境变量配置
   - 查看启动日志

2. **模型调用失败**
   - 检查 API 密钥是否正确
   - 验证网络连接
   - 查看模型配置

3. **MCP 工具无法启动**
   - 检查命令路径是否正确
   - 验证文件权限
   - 查看错误日志

### 调试技巧

```bash
# 启用调试模式
python main.py --log-level DEBUG

# 检查配置
curl http://localhost:8000/api/health

# 查看系统状态
curl http://localhost:8000/api/status
```

## 📈 性能优化

### 推荐配置

- 生产环境使用 Gunicorn + Uvicorn
- 配置适当的工作进程数
- 启用 Redis 缓存
- 配置负载均衡

### 监控指标

- API 响应时间
- 错误率
- 并发用户数
- 内存使用率
- CPU 使用率

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

如有问题，请：

1. 查看文档和 FAQ
2. 搜索现有的 Issues
3. 创建新的 Issue
4. 联系维护者
