# sourced by the start-*.sh scripts on tatooine
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
DEV="--device /dev/kfd --device /dev/dri --group-add $(stat -c %g /dev/kfd) --group-add $(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined"
GG=/data/models/gguf/qwen3.6-35b-a3b
