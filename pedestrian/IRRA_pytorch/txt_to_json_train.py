import os
import json
import random

# s2 contain s1
def func_check_same_attri(s1, s2): 
    s1 = s1.split(',')
    s2 = s2.split(',')
    if len(s1) != 21: 
        assert False, "not person data"
    if s1[0] != s2[0]:
        return 0
    same = 0
    sum1 = 0
    for i in range(len(s1)):
        if s1[i] == '1' and s2[i] == '1':
            same += 1
        if s1[i] == '1': sum1 += 1
    return same == sum1

def process_data_for_train(split, num_start):
    #get data from txt to line_list[]
    path = f'contest_data/{split}/{split}_label.txt'
    line_list = []
    uu = 0
    with open(path,'r',encoding='utf-8')as f:               
        for line in f: 
            uu += 1
            if uu <= num_start:
                continue  
            line_list.append(line)
    all_data = []
    num = len(line_list)
    getid = {}
    cntid = 0
    #process data single
    for i in range(num):
        tmp = line_list[i].rstrip('\n').split('$')
        #print(i)
        d = {}
        d['file_path'] = tmp[0] 
        d['split']= split
        d['attribute'] = [tmp[1]]
        d['captions'] = [tmp[2]]
        if not tmp[1] in getid.keys():
            cntid += 1
            getid[tmp[1]] = cntid 
        d['id'] = getid[tmp[1]]
        all_data.append(d)
    print(f'train id {cntid}')
    return all_data 

def process_data_for_test(num_start):
    cnt = 0
    uu = 0
    all_data = []
    file_name = []
    for path in os.listdir('contest_data/test/test_person'):
        file_name.append(path)
    file_name.sort()
    path = 'contest_data/test/test_text.txt'

    with open(path,'r',encoding='utf-8')as f:               
        for line in f:   
            uu += 1
            if uu <= num_start:
                continue
            tmp = line.rstrip('\n').split('$')    
            d = {}
            d['file_path'] = file_name[cnt]
            cnt += 1
            d['split']='test'
            d['id'] = cnt  
            d['captions'] = [tmp[0]]  
            all_data.append(d)
    return all_data

#val:3413  train:42704 test:7611
sum = []
sum += process_data_for_train(split = 'train', num_start = 42704)
sum += process_data_for_train(split = 'val', num_start = 3413)
sum += process_data_for_test(num_start = 7611)
with open(f'data.json','w',encoding='utf-8')as file:   
    json.dump(sum, file, ensure_ascii=False, indent=4)
