import random
import math
random.seed(2024)

user_item = dict()

mode = 'non_DEBUG'

with open('./train.txt', 'r') as f:
    for line in f:
        splits = line.strip().split()
        if splits[0] not in user_item.keys():
            user_item[splits[0]] = []
            user_item[splits[0]].append( splits[1] )
        else:
            user_item[splits[0]].append( splits[1] )

user_item_train = dict()
user_item_valid = dict()

for key,val in user_item.items():
    valid_sample = random.sample(val, math.ceil(len(val) * 0.1))
    user_item_valid[key] = valid_sample
    val_cpy = val.copy()
    for x in valid_sample:
        val_cpy.remove(x)
    train_sample = val_cpy
    user_item_train[key] = train_sample

if mode == 'DEBUG':
    check_num = 20
    print(user_item_valid['{}'.format(check_num)], user_item_train['{}'.format(check_num)], user_item['{}'.format(check_num)])


f_train = open('./train_data.txt','w')
f_valid = open('./valid_data.txt','w')


for key, val in user_item_valid.items():
    for item in val:
        f_valid.write(key + ' ' + item + '\n')

for key, val in user_item_train.items():
    for item in val:
        f_train.write(key + ' ' + item + '\n')
