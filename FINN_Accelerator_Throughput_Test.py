# Imports
import argparse
import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

# Instantiating the overlay for the generated accelerator
accel = FINNExampleOverlay(
        bitfile_name='../bitfile/finn-accel.bit',
        platform='zynq-iodma',
        io_shape_dict=io_shape_dict,
        batch_size=10,
        runtime_weight_dir="runtime_weights/",
    )

import pprint
lst_batches = [10, 50, 100, 500, 1000, 5000, 10000]
total = 10000
# prepare dict
accel.batch_size= 10
benchmark_results = accel.throughput_test()
data = benchmark_results

# Prepare csv file
import csv
fields = ['#', 10, 50, 100, 500, 1000, 5000, 10000]
# Creating rows list
rows = []
for kkey in data.keys():
    row = [kkey]
    rows.append(row)
    
# only test
#pprint.pprint(rows)
    
for bs in lst_batches:
    # reset data to 0
    for kkey in data.keys():
        data[kkey] = 0
    
    # run throughput test
    nbatch = int(total/bs)
    print("Real batchsize : ", bs)
    accel.batch_size = bs
    # accumulate results
    for ii in range(nbatch):
        #print(" . (%d/%d)"%(ii,nbatch),end='')
        print(".",end='')
        benchmark_results = accel.throughput_test()
        for kkey in data.keys():
            data[kkey] += benchmark_results[kkey]
    # average results
    print("")
    for kkey in data.keys():
        if kkey != 'copy_input_data_to_device[ms]' and kkey != 'copy_output_data_from_device[ms]' and kkey != 'fold_input[ms]' and kkey != 'pack_input[ms]' and kkey != 'runtime[ms]' and kkey != 'unfold_output[ms]' and kkey != 'unpack_output[ms]': 
            data[kkey] = data[kkey]/nbatch
        # limit values to 2 decimal places
        data[kkey] = round(data[kkey], 2)
            
    # TODO : append to a list of dictionaries (rows in csv file)
    kkeys = [kkey for kkey in data.keys()]
    for i in range(len(rows)):
        kkey = kkeys[i]
        rows[i].append(data[kkey])
        
    
    pprint.pprint(data)
    print("="*80)

# only test    
#print(fields)    
#pprint.pprint(rows) 

# export to csv file
filename = "throughput_test_result.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)  
