# optimizer 
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 2 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data 
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 2 --update_granularity 1 -bsz 128 --optim sgd --start_epoch 0 --shard_data 

# reduce op 
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data --reduce_op avg_w_count

# shard_data
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'no_drop' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 


# start_epoch
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 2 --shard_data 
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 5 --shard_data


# cyclic or random
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --cyclic --shard_data


# baseline
python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data

# # workers, what happends when increase the number of worker
# python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 4 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data 
# python accuracy_benchmark.py  --arch resnet18 --dropping_scheme 'jgsaw' --port 23578 --num_prcs 8 --update_granularity 1 -bsz 128 --optim adam --start_epoch 0 --shard_data 
