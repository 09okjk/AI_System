# MCP服务器快速入门

## 让我们开始构建我们的天气服务器

### 前提知识

本快速入门假设你熟悉：

- Python
- LLMs，如 Claude

### 系统要求

- 已安装 Python 3.10 或更高版本。
- 你必须使用 Python MCP SDK 1.2.0 或更高版本。

### 设置你的环境

首先，让我们安装 uv 并设置我们的 Python 项目和环境：

MacOS/Linux

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

之后请务必重启你的终端，以确保 uv 命令被识别。

现在，让我们创建并设置我们的项目：

MacOS/Linux

```sh
# 为我们的项目创建一个新 directory
uv init weather
cd weather

# 创建 virtual environment 并激活它
uv venv
source .venv/bin/activate

# 安装 dependencies
uv add "mcp[cli]" httpx

# 创建我们的 server file
touch weather.py
```

现在让我们深入构建你的服务器。

## 构建你的服务器

### 导入 packages 并设置 instance

将这些添加到你的 weather.py 文件的顶部：

```python
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"
```

FastMCP class 使用 Python type hints 和 docstrings 来自动生成 tool definitions，从而轻松创建和维护 MCP tools。

### Helper functions

接下来，让我们添加 helper functions，用于查询和格式化来自 National Weather Service API 的数据：

```python
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """向 NWS API 发送请求，并进行适当的错误处理。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """将警报 feature 格式化为可读的字符串。"""
    props = feature["properties"]
    return f"""
事件: {props.get('event', 'Unknown')}
区域: {props.get('areaDesc', 'Unknown')}
严重性: {props.get('severity', 'Unknown')}
描述: {props.get('description', 'No description available')}
指示: {props.get('instruction', 'No specific instructions provided')}
"""

```

### 实现 tool execution

Tool execution handler 负责实际执行每个 tool 的逻辑。让我们添加它：

```python
@mcp.tool()
async def get_alerts(state: str) -> str:
    """获取美国州的天气警报。

    Args:
        state: 两个字母的美国州代码（例如 CA、NY）
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "无法获取警报或未找到警报。"

    if not data["features"]:
        return "该州没有活跃的警报。"

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """获取某个位置的天气预报。

    Args:
        latitude: 位置的纬度
        longitude: 位置的经度
    """
    # 首先获取预报网格 endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "无法获取此位置的预报数据。"

    # 从 points response 中获取预报 URL
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "无法获取详细预报。"

    # 将 periods 格式化为可读的预报
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # 仅显示接下来的 5 个 periods
        forecast = f"""
        {period['name']}:
        温度: {period['temperature']}°{period['temperatureUnit']}
        风: {period['windSpeed']} {period['windDirection']}
        预报: {period['detailedForecast']}
        """
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

```

### 运行 server

最后，让我们初始化并运行 server：

```python
if __name__ == "__main__":
    # 初始化并运行 server
    mcp.run(transport='stdio')
```

你的 server 已经完成！运行 uv run weather.py 以确认一切正常
