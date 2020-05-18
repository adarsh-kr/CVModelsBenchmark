#python distributed_resnet18.py -prob 0 0   --dropping_scheme 2 --port 23594 &
#python distributed_resnet18.py -prob 25 25 --dropping_scheme 2 --port 23596 &
#python distributed_resnet18.py -prob 50 50 --dropping_scheme 2 --port 23598 &
#python distributed_resnet18.py -prob 90 90 --dropping_scheme 2 --port 23601 &
#python distributed_resnet18.py -prob 99 99 --dropping_scheme 2 &
#python distributed_resnet18.py  --dropping_scheme 6 --port 23594 --num_prcs 4
#python distributed_resnet18.py  --dropping_scheme 8 --port 23598 --num_prcs 4
#python distributed_resnet18.py  --dropping_scheme 0 --port 23578 --num_prcs 4
python distributed_resnet18.py  --dropping_scheme 6 --port 23578 --num_prcs 4 --update_granularity 50 -bsz 128


            

