#!/bin/bash
#touch mnist_conv_stride_sweep.txt
#touch mnist_fc_stride_sweep.txt
touch cifar10_conv_stride_sweep.txt
for i in 4 8 16
do
    #python main.py mnist_conv 20 -s $i -e 100 >> conv_stride_sweep.txt
    #python main.py mnist_fc 20 -s $i -e 100 >> fc_stride_sweep.txt
    python main.py cifar10 40 -s $i -e 100 >> cifar10_conv_stride_sweep.txt
done
