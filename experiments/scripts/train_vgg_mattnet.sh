

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="faster_rcnn"
ITERS=1190000
TAG="notime"
NET="vgg16"
ID="vgg_with_st"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_iters 30000 \
    --with_st 1 \
    --id ${ID}
