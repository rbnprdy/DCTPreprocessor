#!/bin/bas
echo "Visualizing with MNIST (20% of components, stirde of 4) [exit window to continue demo]"
python visualizations.py mnist 20 4
echo "Visualizing with CIFAR-10 (40% of components, stirde of 4) [exit window to continue demo]"
python visualizations.py cifar10 40 4

echo "Training with MNIST FC (20% of components, stirde of 4)"
python main.py mnist_fc 20 -e 5 -v 1 
echo "Training with MNIST Conv (20% of components, stirde of 4)"
python main.py mnist_conv 20 -e 5 -v 1
echo "Training with CIFAR-10 Conv (40% of components, stirde of 4)"
python main.py cifar10 40 -e 5 -v 1 
