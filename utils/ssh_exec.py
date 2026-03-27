#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSH 远程执行工具
用于在远程服务器上执行命令和脚本
使用 paramiko 库支持密码认证
"""

import json
import os
import sys
from pathlib import Path
from io import StringIO

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("警告：paramiko 未安装，请运行：pip install paramiko")

# 配置信息
CONFIG_FILE = Path(__file__).parent.parent / "remote_config.json"

def load_config():
    """加载配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

class SSHClient:
    """SSH 客户端类"""

    def __init__(self):
        self.config = load_config()
        self.ssh_config = self.config['ssh']
        self.conda_config = self.config['conda']
        self.client = None

    def connect(self):
        """连接到远程服务器"""
        if not HAS_PARAMIKO:
            raise ImportError("paramiko 库未安装")

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"[SSH] 连接到 {self.ssh_config['host']}:{self.ssh_config['port']}...")
        self.client.connect(
            hostname=self.ssh_config['host'],
            port=self.ssh_config['port'],
            username=self.ssh_config['user'],
            password=self.ssh_config['password'],
            timeout=30,
            allow_agent=False,
            look_for_keys=False
        )
        print("[SSH] 连接成功!")

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            self.client = None

    def exec_command(self, command, activate_conda=True):
        """
        在远程服务器上执行命令

        Args:
            command: 要执行的命令
            activate_conda: 是否激活 conda 环境

        Returns:
            (stdout, stderr, return_code)
        """
        if not self.client:
            self.connect()

        # 构建完整命令
        if activate_conda:
            full_command = f"""
            source ~/.bashrc 2>/dev/null || true
            conda activate {self.conda_config['env_name']} 2>/dev/null || true
            {command}
            """
        else:
            full_command = command

        print(f"[SSH] 执行：{command}")

        stdin, stdout, stderr = self.client.exec_command(full_command)
        stdout_str = stdout.read().decode('utf-8')
        stderr_str = stderr.read().decode('utf-8')
        return_code = stdout.channel.recv_exit_status()

        if stdout_str:
            print(f"[STDOUT]\n{stdout_str}")
        if stderr_str:
            print(f"[STDERR]\n{stderr_str}")
        if return_code != 0:
            print(f"[ERROR] 返回码：{return_code}")

        return stdout_str, stderr_str, return_code

    def upload_file(self, local_path, remote_path):
        """
        上传文件到远程服务器

        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
        """
        if not self.client:
            self.connect()

        sftp = self.client.open_sftp()
        print(f"[SFTP] 上传 {local_path} -> {remote_path}")

        # 确保远程目录存在
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            print(f"[SFTP] 创建目录 {remote_dir}")
            self._mkdir_p(sftp, remote_dir)

        sftp.put(local_path, remote_path)
        sftp.close()
        print("[SUCCESS] 上传成功")

    def download_file(self, remote_path, local_path):
        """
        从远程服务器下载文件

        Args:
            remote_path: 远程文件路径
            local_path: 本地文件路径
        """
        if not self.client:
            self.connect()

        sftp = self.client.open_sftp()
        print(f"[SFTP] 下载 {remote_path} -> {local_path}")

        # 确保本地目录存在
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        sftp.get(remote_path, local_path)
        sftp.close()
        print("[SUCCESS] 下载成功")

    def _mkdir_p(self, sftp, remote_dir):
        """递归创建远程目录"""
        parts = remote_dir.strip('/').split('/')
        path = ''
        for part in parts:
            path += '/' + part
            try:
                sftp.stat(path)
            except FileNotFoundError:
                sftp.mkdir(path)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def ssh_exec(command):
    """快捷函数：执行 SSH 命令"""
    with SSHClient() as ssh:
        return ssh.exec_command(command)


def ssh_upload(local_path, remote_path):
    """快捷函数：上传文件"""
    with SSHClient() as ssh:
        ssh.upload_file(local_path, remote_path)


def ssh_download(remote_path, local_path):
    """快捷函数：下载文件"""
    with SSHClient() as ssh:
        ssh.download_file(remote_path, local_path)


def main():
    """主函数 - 示例用法"""
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
        ssh_exec(command)
    else:
        # 测试连接
        print("测试 SSH 连接...")
        with SSHClient() as ssh:
            stdout, stderr, code = ssh.exec_command("hostname && uname -a")
            if code == 0:
                print("SSH 连接成功!")
            else:
                print("SSH 连接失败!")

        # 测试文件列表
        print("\n测试列出远程目录...")
        with SSHClient() as ssh:
            stdout, stderr, code = ssh.exec_command("ls -la /18018998051/CTA/")


if __name__ == "__main__":
    main()
