#!/bin/bash
touch conv_sweep.txt
touch fc_sweep.txt
for i in 95 90 85 80 75 70 65 60 55 50 45 40 35 30 25 20 15 10 5
do
    python main.py conv mnist $i -e 100 >> conv_sweep.txt
    python main.py fc mnist $i -e 100 >> fc_sweep.txt
done
for i in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
do
    python main.py conv mnist $i -e 100 >> conv_sweep.txt
    python main.py fc mnist $i -e 100 >> fc_sweep.txt
done
