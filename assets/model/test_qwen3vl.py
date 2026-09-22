#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 一体化测试脚本
自动启动 vLLM 服务 -> 等待就绪 -> 执行测试 -> 可选保留服务

直接修改下面的配置后运行: python run_vllm_test.py
"""

import base64
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import time
import requests

# ==================== 在这里修改 ====================
IMAGE_PATH = "/home/lry/MoonUnrealEnv/image.png"
PROMPT = "描述这张图片的内容"

MODEL_PATH = "/home/lry/MoonUnrealEnv/Asset/model/Qwen3-VL-8B-Instruct"
MODEL_NAME = "Qwen3-VL-8B"
PREFERRED_PORT = 3001          # 首选端口，被占用时自动换
KEEP_SERVER = False            # True=测试后保留服务; False=测试后关闭
# ====================================================

# 其他固定配置
HOST = "127.0.0.1"
API_KEY = "EMPTY"
MAX_TOKENS = 512
TEMPERATURE = 0.7
STARTUP_TIMEOUT = 300          # 等待服务启动的最长秒数
LOG_FILE = "/home/lry/MoonUnrealEnv/Asset/model/vllm_auto.log"

# 修正后的 vLLM 启动参数
VLLM_ARGS = [
    "--gpu-memory-utilization", "0.90",   # 从 0.85 提到 0.90
    "--max-model-len", "4096",            # 从 8192 降到 4096
    "--max-num-seqs", "8",                # 从 16 降到 8
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--max-num-batched-tokens", "512",
]



# ---------------------- 工具函数 ----------------------
def find_free_port(preferred: int) -> int:
    """从 preferred 开始找一个可用端口"""
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def encode_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{img_b64}"


def wait_for_server(url: str, proc: subprocess.Popen, timeout: int) -> bool:
    """轮询 /v1/models 直到服务就绪或进程退出"""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"\n❌ vLLM 进程已退出，返回码 {proc.returncode}")
            print(f"   请查看日志: {LOG_FILE}")
            return False
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(f"\n❌ 等待服务启动超时 ({timeout}s)")
    return False


# ---------------------- 启动 vLLM ----------------------
def start_vllm(port: int) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", MODEL_NAME,
        "--host", HOST,
        "--port", str(port),
    ] + VLLM_ARGS

    print(f"🚀 启动 vLLM 服务 (端口 {port})")
    print(f"   日志: {LOG_FILE}")

    log_f = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,   # 新建进程组，方便整体 kill
    )
    return proc


def stop_vllm(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    print("🛑 停止 vLLM 服务...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ---------------------- 调用模型 ----------------------
def ask_qwen3vl(api_url: str, image_path: str, prompt: str) -> str:
    image_url = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 vLLM 服务"
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP 错误: {e}\n{resp.text}"

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"❌ 解析响应失败: {e}\n{resp.text}"


# ---------------------- 主流程 ----------------------
def main():
    # 1. 检查模型目录
    if not os.path.isdir(MODEL_PATH):
        print(f"❌ 模型目录不存在: {MODEL_PATH}")
        sys.exit(1)

    # ===== 添加环境变量，禁用 FlashInfer 采样器 =====
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print("ℹ️  已设置 VLLM_USE_FLASHINFER_SAMPLER=0 以禁用 FlashInfer 采样器")

    # 2. 找可用端口
    port = find_free_port(PREFERRED_PORT)
    if port != PREFERRED_PORT:
        print(f"⚠️  端口 {PREFERRED_PORT} 被占用，改用 {port}")

    # 3. 启动服务
    proc = start_vllm(port)
    base_url = f"http://{HOST}:{port}"
    api_url = f"{base_url}/v1/chat/completions"

    try:
        # 4. 等待就绪
        print("⏳ 等待服务启动", end="", flush=True)
        if not wait_for_server(f"{base_url}/v1/models", proc, STARTUP_TIMEOUT):
            stop_vllm(proc)
            sys.exit(1)
        print(f"\n✅ 服务已就绪: {base_url}")

        # 5. 执行测试
        print(f"\n📷 图像: {IMAGE_PATH}")
        print(f"💬 提示词: {PROMPT}")
        print("⏳ 正在请求模型...\n")

        answer = ask_qwen3vl(api_url, IMAGE_PATH, PROMPT)
        print("=" * 60)
        print(answer)
        print("=" * 60)

    finally:
        # 6. 决定是否关闭服务
        if KEEP_SERVER:
            print(f"\n🔒 服务保留运行，端口 {port}")
            print(f"   停止命令: pkill -f 'vllm serve'")
        else:
            stop_vllm(proc)
            print("\n✅ 已关闭 vLLM 服务")


if __name__ == "__main__":
    main()
