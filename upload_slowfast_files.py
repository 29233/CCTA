#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload all necessary SlowFast files to remote server
"""

import os
from pathlib import Path
from utils.ssh_exec import SSHClient

# Base paths
LOCAL_SLOWFAST = Path(__file__).parent / "SlowFast" / "slowfast"
REMOTE_SLOWFAST = "/18018998051/CTA/slowfast"

# Files to upload - core SlowFast files needed for multi-task training
FILES_TO_UPLOAD = [
    # Root __init__
    ("__init__.py", "__init__.py"),

    # Config files
    ("config/__init__.py", "config/__init__.py"),
    ("config/defaults.py", "config/defaults.py"),
    ("config/custom_config.py", "config/custom_config.py"),

    # Models __init__ and core files
    ("models/__init__.py", "models/__init__.py"),
    ("models/attention.py", "models/attention.py"),
    ("models/batchnorm_helper.py", "models/batchnorm_helper.py"),
    ("models/build.py", "models/build.py"),
    ("models/common.py", "models/common.py"),
    ("models/head_helper.py", "models/head_helper.py"),
    ("models/losses.py", "models/losses.py"),
    ("models/optimizer.py", "models/optimizer.py"),
    ("models/resnet_helper.py", "models/resnet_helper.py"),
    ("models/stem_helper.py", "models/stem_helper.py"),
    ("models/utils.py", "models/utils.py"),

    # Coronary specific models
    ("models/coronary_head.py", "models/coronary_head.py"),
    ("models/coronary_loss.py", "models/coronary_loss.py"),
    ("models/video_model_builder.py", "models/video_model_builder.py"),

    # Datasets files
    ("datasets/__init__.py", "datasets/__init__.py"),
    ("datasets/build.py", "datasets/build.py"),
    ("datasets/utils.py", "datasets/utils.py"),
    ("datasets/transform.py", "datasets/transform.py"),
    ("datasets/cv2_transform.py", "datasets/cv2_transform.py"),
    ("datasets/decoder.py", "datasets/decoder.py"),
    ("datasets/loader.py", "datasets/loader.py"),
    ("datasets/video_container.py", "datasets/video_container.py"),
    ("datasets/coronary.py", "datasets/coronary.py"),

    # Tools
    ("tools/__init__.py", "tools/__init__.py"),
    ("tools/run_net.py", "tools/run_net.py"),
    ("tools/train_coronary_multitask.py", "tools/train_coronary_multitask.py"),

    # Utils
    ("utils/__init__.py", "utils/__init__.py"),
    ("utils/benchmark.py", "utils/benchmark.py"),
    ("utils/bn_helper.py", "utils/bn_helper.py"),
    ("utils/checkpoint.py", "utils/checkpoint.py"),
    ("utils/distributed.py", "utils/distributed.py"),
    ("utils/env.py", "utils/env.py"),
    ("utils/logging.py", "utils/logging.py"),
    ("utils/lr_policy.py", "utils/lr_policy.py"),
    ("utils/meters.py", "utils/meters.py"),
    ("utils/metrics.py", "utils/metrics.py"),
    ("utils/misc.py", "utils/misc.py"),
    ("utils/multiprocessing.py", "utils/multiprocessing.py"),
    ("utils/weight_init_helper.py", "utils/weight_init_helper.py"),

    # Visualization (needed for tensorboard)
    ("visualization/__init__.py", "visualization/__init__.py"),
    ("visualization/tensorboard_vis.py", "visualization/tensorboard_vis.py"),
    ("visualization/utils.py", "visualization/utils.py"),
]

def upload_files():
    """Upload all necessary files to remote server"""

    print(f"Uploading SlowFast files to {REMOTE_SLOWFAST}")
    print("=" * 60)

    uploaded = 0
    skipped = 0
    errors = 0

    with SSHClient() as ssh:
        for local_rel, remote_rel in FILES_TO_UPLOAD:
            local_path = LOCAL_SLOWFAST / local_rel
            remote_path = f"{REMOTE_SLOWFAST}/{remote_rel}"

            if local_path.exists():
                try:
                    ssh.upload_file(str(local_path), remote_path)
                    uploaded += 1
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] {local_rel}: {e}")
            else:
                skipped += 1

    print("=" * 60)
    print(f"Upload complete: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    return uploaded > 0 and errors == 0

if __name__ == "__main__":
    success = upload_files()
    if success:
        print("\nAll files uploaded successfully!")
    else:
        print("\nSome files failed to upload.")
