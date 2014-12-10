Handwritten Digit Recognition using Deep Learning (Convolutional Neural Networks)
================

This Matlab implementation of a 3-layer convolutional network was tested on MNIST public dataset which includes 60,000 training samples and 10,000 testing samples. It achieves > 97% accuracy with 30 training iterations.

**Network Architecture:**

- Convolutional layer L1 with four feature maps from four 5x5 kernels.
- Subsampling layer L2 with non-overlapping 2 x 2 windows to compute local average.
- Fully-connected output layer L3. The true output vector is represented as y ∈ R^10 corresponding to 0-9.

For a visualization of the network, see: https://github.com/lhoang29/DigitRecognition/blob/master/threelayer.png

**How to run:**

1. Download dataset from http://yann.lecun.com/exdb/mnist/
2. Load data:`[trainlabels,trainimages,testlabels,testimages] = cnnload();`
3. Train & test:`[missimages, misslabels] = cnntrain(trainlabels,trainimages,testlabels,testimages,60000,30,0.01);`
The above specifies training on the maximum set of 60k samples with 30 iterations and learning rate η=0.01. For faster training time, use 10k samples with 10 iterations and η=0.01 which should achieve ~95% accuracy.
4. Show failed predictions: `showmiss(missimages,misslabels,testimages,testlabels,25,2);`
This sets numshow = 25 and numpages = 2 which displays 25 * 2 = 50 first failed predictions.
Each page is a separate figure organized as a square of size sqrt(numshow) x sqrt(numshow).

**How it works:**

Input sample of is a 28 x 28 grayscale image of a digit (0-9) which is size-normalized and centered. Pixel values are normalized to [-1,1] range for better convergence.

All nonlinear functions are defined as standard hyperbolic tangents: f(x)=tanh(x). Initial weight values in each layer are drawn randomly from a Gaussian distribution with mean 0 and standard deviation 0.01. Bias and scaling parameters are all set to 1. Parameter updates are done using stochastic gradient descent. Learning rate is set to η=0.01 was used.

