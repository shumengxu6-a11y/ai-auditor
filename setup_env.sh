#!/bin/bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (仅对当前脚本执行有效，用户需使用 source 运行)
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

echo "✅ 虚拟环境已创建并安装依赖。"
echo "请运行 'source setup_env.sh' 或 'source venv/bin/activate' 来激活环境。"
