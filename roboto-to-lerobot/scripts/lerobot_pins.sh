#!/bin/sh
#
# Install pins that depend on the lerobot version, shared by the Dockerfile
# and scripts/setup.sh. POSIX sh, because the Dockerfile runs it with /bin/sh.
#
#   lerobot_pins.sh torch <lerobot_version>
#       Prints the torch requirements to install from the PyTorch CPU index
#       before lerobot. They must stay within that lerobot's caps, or
#       installing lerobot replaces them with CUDA builds from PyPI.
#       lerobot 0.7 allows torch < 2.16 but torchvision < 0.30, and
#       torchvision 0.29 is built for torch 2.14: with torch 2.15 it fails to
#       import, and with it every lerobot.datasets module. torchcodec comes
#       from the CPU index too, because its PyPI build is a CUDA build that
#       fails to load without the CUDA libraries; < 0.18 is lerobot 0.7's cap.
#
#   lerobot_pins.sh extras <lerobot_version>
#       Prints the lerobot extras to install. 0.7 adds lancedb-convert, the
#       Lance converter behind the output_format parameter, which 0.3.x
#       lacks.
#
#   lerobot_pins.sh overrides <lerobot_version>
#       Prints uv overrides for installing pyproject.toml after lerobot.
#       lerobot 0.7 requires filelock >= 4.2.1 and the roboto SDK caps it
#       below 4; both call only FileLock(path, timeout=...) and Timeout,
#       which filelock 4 keeps.

set -eu

case "$1:$2" in
    torch:0.7.*) echo "torch>=2.14,<2.15 torchvision>=0.29,<0.30 torchcodec<0.18" ;;
    torch:*) echo "torch" ;;
    extras:0.7.*) echo "dataset,lancedb-convert" ;;
    extras:*) echo "dataset" ;;
    overrides:0.7.*) echo "filelock>=4.2.1,<5" ;;
    overrides:*) ;;
    *)
        echo "usage: $0 torch|extras|overrides <lerobot_version>" >&2
        exit 1
        ;;
esac
