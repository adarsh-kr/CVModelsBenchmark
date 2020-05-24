# models 
#models=(resnet18 resnet50 vgg11 vgg16 vgg19 googlenet resnet101)
#models=(vgg11 vgg16 vgg19 mobile dense121 dense201 lenet resnet50)
model=(resnet18)

#for model in ${models[@]};do
#    CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch $model --dropping_scheme 'jgsaw'   --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --epochs 300 &
#    CUDA_VISIBLE_DEVICES=2,3 python accuracy_benchmark.py  --arch $model --dropping_scheme 'jgsaw'   --port 23580 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 5 --epochs 300 &
#    CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch $model --dropping_scheme 'jgsaw'   --port 23584 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 10 --epochs 300 &
 #   CUDA_VISIBLE_DEVICES=2,3 python accuracy_benchmark.py  --arch $model --dropping_scheme 'jgsaw'   --port 23590 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 2 --epochs 300 &
#done

#CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'no_drop'   --port 23578 --num_prcs 1 --update_granularity 1 -bsz 64 --optim sgd --start_epoch 0 --epochs 100 --lr 0.05 --decay_after_n 30 
#CUDA_VISIBLE_DEVICES=2,3 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw'   --port 23656 --num_prcs 4 --update_granularity 1 -bsz 128 --optim sgd --start_epoch 0 --epochs 100 --lr 0.05 --decay_after_n 30 

#CUDA_VISIBLE_DEVICES=0,1 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'no_drop'   --port 23528 --num_prcs 4 --update_granularity 1 -bsz 128 --optim sgd --start_epoch 0 --epochs 100 --lr 0.05 --decay_after_n 10 &
#CUDA_VISIBLE_DEVICES=2,3 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw'   --port 23598 --num_prcs 4 --update_granularity 1 -bsz 128 --optim sgd --start_epoch 0 --epochs 100 --lr 0.05 --decay_after_n 10


CUDA_VISIBLE_DEVICES=0 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'no_drop'   --port 23578 --num_prcs 1 --update_granularity 1 -bsz 64 --optim sgd --start_epoch 0 --epochs 100 --lr 0.1 --decay_after_n 30 &
CUDA_VISIBLE_DEVICES=1,2,3 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw'   --port 23656 --num_prcs 4 --update_granularity 1 -bsz 256 --optim sgd --start_epoch 0 --epochs 100 --lr 0.5 --decay_after_n 30 
#CUDA_VISIBLE_DEVICES=2,3 python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw'   --port 23965 --num_prcs 4 --update_granularity 1 -bsz 256 --optim sgd --start_epoch 0 --epochs 100 --lr 0.5 --decay_after_n 30 

