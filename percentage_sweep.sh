#!/bin/bash
touch cifar10_conv_percentage_sweep.txt
touch mnist_fc_percentage_sweep.txt
touch mnist_conv_percentage_sweep.txt
for i in 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5
do
    python main.py cifar10 $i -e 100 >> cifar10_conv_percentage_sweep.txt
    python main.py mnist_fc $i -e 100 >> mnist_fc_percentage_sweep.txt 
    python main.py mnist_conv $i -e 100 >> mnist_conv_percentage_sweep.txt
done
for i in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
do
    python main.py cifar10 $i -e 100 >> cifar10_conv_percentage_sweep.txt 
    python main.py mnist_conv $i -e 100 >> mnist_conv_percentage_sweep.txt
    python main.py mnist_fc $i -e 100 >> mnist_fc_percentage_sweep.txt
done
