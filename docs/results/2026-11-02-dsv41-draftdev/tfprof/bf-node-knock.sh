#!/bin/bash
# bf-node.sh head|worker|stop|evict : one node's part of the BLACKFOLD TP2 serve (adapted from deploy/blackfold-native, 6366d2c + patches).
# Differences from the recipe script: no sudo (no rail-align, no drop_caches), API bound to the tailscale address with a key file,
# rank cache dir named rank-cache (CPU writer output), runs per node (no inter-node ssh), paths local to ~/code/blackfold-tp2.
# Fixed 2-rail setup: primary rocep1s0f0 + twin roceP2p1s0f0, RoCE v2 GID index 3 on both devices of both nodes (read from sysfs), MTU 9000.
set -eu
B=$HOME/code/blackfold-tp2
NAME=ds41-knock-dd
IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
HEAD_FABRIC=10.100.154.1
API_ADDR=100.120.88.104
PORT=18191
SERVED=DeepSeek-v4.1-Flash-1017-heretic-tf
MODEL=/mnt/models/dsv41-1017/blackfold/exl3-2.9bpw-heretic
RDMA=rocep1s0f0,roceP2p1s0f0
IFACE=enp1s0f0np0
GID=3
MAXTOK=${MAX_TOKENS:-32768}
CTX=${CTX:-65536}; PAR=${PAR:-1}; STREAMS=${STREAMS:-1}   # Mia run used 65536/1; 1048576/4 left 3.2 GiB free and earlyoom (-m 3,2 = 3.65 GiB) SIGTERMed the head

evict() {  # drop clean page cache of the big files (posix_fadvise DONTNEED, no sudo)
  python3 -I - "$B/cache" "$B/engram" <<'PY'
import os, sys
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            except OSError as e:
                print("skip", p, e)
PY
}

common="-e NCCL_SOCKET_IFNAME=$IFACE -e NCCL_IB_HCA=$RDMA -e TF_RDMA_DEVICES=$RDMA -e NCCL_IB_GID_INDEX=$GID -e TF_RDMA_GID_INDEX=$GID \
 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=${TF_SIDE:-1} -e TF_DS_L2_PREFETCH=${TF_PREFETCH:-1} -e TF_DS_ENGRAM_AIO=1 \
 -e TF_DS_STREAMS=$STREAMS -e TF_TP_WORLD=2 -e TF_TP_MASTER=$HEAD_FABRIC -e TF_TP_PORT=29620 \
 -e TF_DS_RANK_CACHE=/cache/rank-cache -e TF_DS_KIT=/kit -e TENSORFOLD_CUDA_KERNELS=/kit/aot \
 -e TF_DS_ENGRAM=/engram -e TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json -e TENSORFOLD_MEMORY_RESERVE_GIB=${TF_RESERVE_GIB:-11.6}"
base="${DRY_RUN:+echo }docker run -d --name $NAME --restart no --gpus all --network host --ipc host --ulimit memlock=-1 \
 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
mounts="-v $B/engine/out/zig-out:/native:ro -v $B/kit:/kit:ro -v $MODEL:/model:ro -v $B/engram:/engram:ro \
 -v $B/cache:/cache:ro -v $B/chat_template.native.jinja:/model/chat_template.jinja:ro"

case ${1:-} in
  evict) evict ;;
  stop) docker rm -f $NAME >/dev/null 2>&1 && echo "$(hostname) $NAME removed" || true ;;
  worker)
    $base $mounts $common $IMAGE /native/bin/tf-dsv41-lanes /model /cache/rank-cache 1 2 $HEAD_FABRIC 29620 /kit --engram /engram \
      --token-map /cache/dsv41_token_map.json --pool $CTX --rdma $RDMA --graphs 1 --side 1 --prefetch 1 --aio 1 --streams $STREAMS ;;
  head)
    $base $mounts -v $B/api-key:/run/blackfold-api-key:ro $common $IMAGE /native/native/bin/tensorfold-native serve /model \
      --host $API_ADDR --port $PORT --api-key-file /run/blackfold-api-key --name $SERVED --context $CTX --parallel $PAR --thinking \
      --max-tokens $MAXTOK --temperature 1 --top-p 0.95 --top-k 20 --backend cuda --no-update-check ;;
  *) echo "usage: $0 head|worker|stop|evict" >&2; exit 2 ;;
esac
