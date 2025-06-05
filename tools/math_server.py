#!/usr/bin/env python3
"""
数学计算 MCP 服务器示例
提供基本的数学运算功能
"""

import sys
import json
import math

def add(a: float, b: float) -> float:
    """两数相加"""
    return a + b

def subtract(a: float, b: float) -> float:
    """两数相减"""
    return a - b

def multiply(a: float, b: float) -> float:
    """两数相乘"""
    return a * b

def divide(a: float, b: float) -> float:
    """两数相除"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

def power(a: float, b: float) -> float:
    """计算a的b次方"""
    return a ** b

def sqrt(a: float) -> float:
    """计算平方根"""
    if a < 0:
        raise ValueError("不能计算负数的平方根")
    return math.sqrt(a)

def factorial(n: int) -> int:
    """计算阶乘"""
    if n < 0:
        raise ValueError("不能计算负数的阶乘")
    if n > 100:
        raise ValueError("数字太大，无法计算阶乘")
    return math.factorial(n)

def sin(x: float) -> float:
    """计算正弦值"""
    return math.sin(x)

def cos(x: float) -> float:
    """计算余弦值"""
    return math.cos(x)

def tan(x: float) -> float:
    """计算正切值"""
    return math.tan(x)

def log(x: float, base: float = math.e) -> float:
    """计算对数"""
    if x <= 0:
        raise ValueError("对数的真数必须大于0")
    if base <= 0 or base == 1:
        raise ValueError("对数的底数必须大于0且不等于1")
    return math.log(x, base)

# MCP 工具定义
TOOLS = {
    "add": {
        "function": add,
        "description": "计算两个数的和",
        "parameters": {
            "a": {"type": "number", "description": "第一个数"},
            "b": {"type": "number", "description": "第二个数"}
        }
    },
    "subtract": {
        "function": subtract,
        "description": "计算两个数的差",
        "parameters": {
            "a": {"type": "number", "description": "被减数"},
            "b": {"type": "number", "description": "减数"}
        }
    },
    "multiply": {
        "function": multiply,
        "description": "计算两个数的积",
        "parameters": {
            "a": {"type": "number", "description": "第一个因数"},
            "b": {"type": "number", "description": "第二个因数"}
        }
    },
    "divide": {
        "function": divide,
        "description": "计算两个数的商",
        "parameters": {
            "a": {"type": "number", "description": "被除数"},
            "b": {"type": "number", "description": "除数（不能为0）"}
        }
    },
    "power": {
        "function": power,
        "description": "计算a的b次方",
        "parameters": {
            "a": {"type": "number", "description": "底数"},
            "b": {"type": "number", "description": "指数"}
        }
    },
    "sqrt": {
        "function": sqrt,
        "description": "计算平方根",
        "parameters": {
            "a": {"type": "number", "description": "要计算平方根的数（非负数）"}
        }
    },
    "factorial": {
        "function": factorial,
        "description": "计算阶乘",
        "parameters": {
            "n": {"type": "integer", "description": "要计算阶乘的非负整数（≤100）"}
        }
    },
    "sin": {
        "function": sin,
        "description": "计算正弦值（弧度制）",
        "parameters": {
            "x": {"type": "number", "description": "角度（弧度）"}
        }
    },
    "cos": {
        "function": cos,
        "description": "计算余弦值（弧度制）",
        "parameters": {
            "x": {"type": "number", "description": "角度（弧度）"}
        }
    },
    "tan": {
        "function": tan,
        "description": "计算正切值（弧度制）",
        "parameters": {
            "x": {"type": "number", "description": "角度（弧度）"}
        }
    },
    "log": {
        "function": log,
        "description": "计算对数",
        "parameters": {
            "x": {"type": "number", "description": "真数（大于0）"},
            "base": {"type": "number", "description": "底数（大于0且不等于1），默认为自然对数", "default": math.e}
        }
    }
}

def handle_request(request_line: str) -> str:
    """处理 MCP 请求"""
    try:
        request_data = json.loads(request_line.strip())
        action = request_data.get("action")
        
        if action == "list_functions":
            # 返回可用函数列表
            functions = []
            for name, info in TOOLS.items():
                functions.append({
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"]
                })
            
            return json.dumps({"functions": functions})
        
        elif action == "call_function":
            # 调用指定函数
            function_name = request_data.get("name")
            arguments = request_data.get("arguments", {})
            
            if function_name not in TOOLS:
                return json.dumps({"error": f"函数不存在: {function_name}"})
            
            try:
                # 处理默认参数
                parameters = TOOLS[function_name]["parameters"]
                for param_name, param_info in parameters.items():
                    if param_name not in arguments and "default" in param_info:
                        arguments[param_name] = param_info["default"]
                
                # 调用函数
                function = TOOLS[function_name]["function"]
                result = function(**arguments)
                
                return json.dumps({
                    "result": result,
                    "function": function_name,
                    "arguments": arguments
                })
            
            except Exception as e:
                return json.dumps({
                    "error": f"函数执行错误: {str(e)}",
                    "function": function_name,
                    "arguments": arguments
                })
        
        else:
            return json.dumps({"error": f"未知操作: {action}"})
    
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON解析错误: {str(e)}"})
    
    except Exception as e:
        return json.dumps({"error": f"处理请求时发生错误: {str(e)}"})

def main():
    """主循环"""
    print(json.dumps({
        "status": "ready",
        "name": "数学计算工具",
        "version": "1.0.0",
        "functions": len(TOOLS)
    }), flush=True)
    
    try:
        # 从标准输入读取请求，处理后输出到标准输出
        for line in sys.stdin:
            if line.strip():
                response = handle_request(line)
                print(response, flush=True)
    
    except KeyboardInterrupt:
        print(json.dumps({"status": "shutdown"}), flush=True)
    
    except Exception as e:
        print(json.dumps({"error": f"服务器错误: {str(e)}"}), flush=True)

if __name__ == "__main__":
    main()