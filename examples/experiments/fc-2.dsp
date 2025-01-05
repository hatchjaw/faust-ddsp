df = library("diff.lib");
import("stdfaust.lib");

// You are expected to have gone through the fc.dsp example before this.
// This example is special -- since you don't have any other layers other than the output layer.

// As a result, the loss function automatically generates the appropriate gradients for backpropagation.
// You don't even need a backpropagation environment for this example!


process = si.bus(2) // This is your input signal (2 channels). Since you have 2 signals, you are expected to have a FC that can take in 2 input signals.
        : (df.fc(1, 2, df.activations.sigmoid, 1e-7)
        : df.losses.L1(1<<3, 1, 2)) // Note that this is the last layer of the NN. As a result, you can calculate the loss here. The loss function is L1.
        ~ (si.block(1), si.bus(3)); // Auto skip 1 signal -- that's the loss signal.
