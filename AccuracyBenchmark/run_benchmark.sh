# models 
#models=(resnet18 resnet50 vgg11 vgg16 vgg19 googlenet resnet101)
#models=(vgg11 vgg16 vgg19 mobile dense121 dense201 lenet resnet50)
models=(dense201)

for model in ${models[@]};do
    CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch $model --dropping_scheme 'no_drop' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data 
    CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch $model --dropping_scheme 'jgsaw'   --port 23580 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data
done

