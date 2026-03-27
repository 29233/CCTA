#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify imports on remote server
"""

from utils.ssh_exec import SSHClient
import tempfile
import os

def test_remote_imports():
    test_script = r'''
import sys
sys.path.insert(0, "/18018998051/CTA")

print("Python version:", sys.version)

try:
    import torch
    print("PyTorch imported successfully, version:", torch.__version__)

    from slowfast.config.defaults import get_cfg
    print("slowfast.config.defaults imported successfully")

    from slowfast.models.coronary_head import CoronaryMultiTaskHead
    print("CoronaryMultiTaskHead imported successfully")

    from slowfast.models.coronary_loss import CoronaryMultiTaskLoss
    print("CoronaryMultiTaskLoss imported successfully")

    from slowfast.datasets.coronary import CoronaryMultiTask
    print("CoronaryMultiTask dataset imported successfully")

    # Test build_model
    from slowfast.models import build_model
    print("build_model imported successfully")

    print("\n=== ALL IMPORTS OK ===")
except Exception as e:
    print("IMPORT ERROR:", str(e))
    import traceback
    traceback.print_exc()
'''

    with SSHClient() as ssh:
        # Write script using cat with heredoc
        print('Writing test script to remote...')

        # Use a different approach - write line by line
        stdin, stdout, stderr = ssh.client.exec_command('cat > /tmp/test_import.py')
        stdin.write(test_script)
        stdin.close()

        print('\nRunning test script...')
        cmd = 'cd /18018998051/CTA && /usr/bin/python /tmp/test_import.py'
        stdout, stderr, code = ssh.exec_command(cmd, activate_conda=False)

        return code == 0

if __name__ == "__main__":
    test_remote_imports()
