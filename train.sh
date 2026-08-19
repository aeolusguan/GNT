set -euo pipefail

OUTPUT_DIR=output_camera_prior
TARTANAIR_ROOT=/mnt/DATA/mobilerobot/tguan/TartanAir
ARKITSCENES_ROOT=/data/disk_24t/wenzhen_data/ARKitScenes_processed
DYNAMIC_REPLICA_ROOT=/data/disk_24t/wenzhen_data/dynamic_replica
ARKITSCENES_CACHE=gent/data/cache/ARKitScenes.pickle
DYNAMIC_REPLICA_CACHE=gent/data/cache/DynamicReplica.pickle

mkdir -p "${OUTPUT_DIR}"
if [ ! -f "${ARKITSCENES_CACHE}" ] || [ ! -f "${DYNAMIC_REPLICA_CACHE}" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/prepare_training_caches.py \
        "${ARKITSCENES_ROOT}" "${DYNAMIC_REPLICA_ROOT}"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29604 training.py \
    --config-name=gent_camera_train \
    data.tartanair.root="${TARTANAIR_ROOT}" \
    data.arkitscenes.root="${ARKITSCENES_ROOT}" \
    data.dynamic_replica.root="${DYNAMIC_REPLICA_ROOT}" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
