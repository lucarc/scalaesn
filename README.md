# scalaesn
a Scala Breeze implementation of an Echo State Network

Reservoir computing is a computational framework for recurrent neural networks in which
the input signal is fed into a large, recurrent pool of neurons called reservoir. The reservoir
is used to map the input to a higher dimension and a simple readout layer (usually a linear
or ridge regression) is then trained to read the state of the reservoir and map to the desired
output. Notable examples of reservoir computing systems are Liquid State Machines and
Echo State Networks (ESN).
Based on the model proposed in  Adaptive Nonlinear System Identification with Echo State Networks by H. Jaeger,  for Echo State Networks, we consider here a reservoir
computing system made up of three distinct layers:

1. input layer, which maps the input signal onto the reservoir in feed-forward mode;

2. reservoir layer, which is the truly recursive neural net (RNN).

3. readout layer, which is a feed-forward neural network that maps the state of the
reservoir to the output desired.

With ESNs, the readout layer is the only component of the network that is trained, via
supervised training, while the weights in both the input and the reservoir are initialized at
random, with some post-processing (see below), and never trained afterwards. 