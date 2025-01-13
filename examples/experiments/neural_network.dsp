import("stdfaust.lib");
df = library("diff.lib");

// Example neural network

// Layer specification.
// Each layer in a neural network is described by a pair of signals; the items
// in each pair represent:
// 1. the number of neurons in the layer.
// 2. the number of weights in each neuron.
// This layer specification describes a network with three layers; the first
// layer consists of two neurons, with one weight each; the second features
// three neurons, each with two weights; the third comprises one neuron with
// three weights.
LAYERSPEC = ((2, 1, 3, 2, 1, 3));

v = df.weightsAndBiases(LAYERSPEC);
d = df.env(v);

// TODO: gradient averaging
process = _ <: nn
with {
    nn(y) = hgroup("Neural Network",
        vgroup("Weights & Biases", d.nn(LAYERSPEC, d.activations.sigmoid)) :> vgroup("Loss & Gradients", d.losses.L2(1<<0, 1e-1, y))
    ) ~ (!,si.bus(v.N)) : (y,_,si.block(v.N));
};
