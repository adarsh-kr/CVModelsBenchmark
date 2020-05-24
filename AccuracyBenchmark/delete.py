update = 1
num_node = 4
rank = 0

#for batch_idx in range(20):
#    is_bucket = (rank + int(batch_idx/update))%num_node
#    print(is_bucket)

total_params = 62 
buckets = []
bucket_size = total_params/num_node
for i in range(num_node):
    buckets += [i*int(bucket_size)]

print(buckets)

