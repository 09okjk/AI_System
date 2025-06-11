#!/bin/bash
# MongoDB 安装脚本 - Ubuntu 24.04

echo "🔧 开始安装 MongoDB..."

# 1. 更新系统包
sudo apt update

# 2. 安装必要的包
sudo apt install -y curl gnupg lsb-release

# 3. 添加MongoDB官方GPG密钥
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor

# 4. 添加MongoDB APT仓库
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

# 5. 更新包列表
sudo apt update

# 6. 安装MongoDB
sudo apt install -y mongodb-org

# 7. 启动MongoDB服务
sudo systemctl start mongod
sudo systemctl enable mongod

# 8. 检查MongoDB状态
sudo systemctl status mongod

echo "✅ MongoDB 安装完成！"
echo "📍 MongoDB 默认端口: 27017"
echo "📁 数据目录: /var/lib/mongodb"
echo "📋 配置文件: /etc/mongod.conf"
echo "🔍 检查状态: sudo systemctl status mongod"
echo "🚀 启动服务: sudo systemctl start mongod"
echo "🛑 停止服务: sudo systemctl stop mongod"