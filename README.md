This is the repository for the "Discrete-Cosine-Transform-Based Preprocessor for Deep Neural Networks" created by Ruben Purdy and Kray Althaus for ECE429 at The University of Arizona.

This code is written in Python 3 and uses TensorFlow, Keras, Numpy, Scipy, and Matplotlib.

In order to see a demo of the preprocessor, run `demo.sh`. This will show a visualization of the DCT transform using MNIST (with 20% of DCT components) and CIFAR-10 (with 40% of DCT components), and then run the training code for the fully connected and convolutional networks with the same MNIST and CIFAR-10 data.

To generate the data used in the paper, run `percentage_sweep.sh` and `stride_sweep.sh`. These will both create and populate text files with the results similar to the ones used in the paper (random initialization may cause some differences.)

To explore the python code more, `main.py` is the entrypoint for training networks, and `visualizations.py` is the entry point for visualizations. Both files have help descriptions which can be brought up by using the `-h` flag.
