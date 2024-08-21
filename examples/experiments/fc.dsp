df = library("diff.lib");
import("stdfaust.lib");

// This example is a demonstration of a neural network with three hidden layers. 
// The developers have attempted to make it as user-friendly as possible, but the user would need to use some basic math to create an example.

// We will go through this example step by step.
process = si.bus(1) // This is your input signal (1 channel). Since you have only one signal, you are expected to have a FC that can take in only 1 input signal.
        : (df.fc(2, 1, df.activations.sigmoid, 1e-7) // This is the first hidden layer. It has 2 neurons and 1 output signal. The activation function is sigmoid and the learning rate is 1e-7.
        // The user needs to note a few things here. The layer the user just defined has 2 neurons and 1 output signal. 
        // As a result, each neuron produces 1 signal. Thus, the next layer should have 2 input signals.
        : (df.fc(3, 2, df.activations.sigmoid, 1e-7), // This is the second hidden layer. It has 3 neurons and 2 input signals. The activation function is sigmoid and the learning rate is 1e-7.
        par(i, b.next_signals(1), _) // You must have this after the second hidden layer. This is a parallel operation that allows the gradients of the first layer to passed through the circuit for backpropagation.
        // For user-friendliness, you can just use b.next_signals(N) where N is the number of FCs that was defined before this layer.
        // The previous layer had 3 neurons, so the next layer should have 3 input signals.
        : ((df.fc(1, 3, df.activations.sigmoid, 1e-7) // This is the third hidden layer. It has 1 neuron and 3 input signals. The activation function is sigmoid and the learning rate is 1e-7.
        : df.losses.L1(1<<3, 1, 3)), // Note that this is the last layer of the NN. As a result, you can calculate the loss here. The loss function is L1.
        par(i, b.next_signals(2), _) // You must have this after the third hidden layer. This is a parallel operation that allows the gradients of the first+second layer to passed through the circuit for backpropagation.
        // This marks the end of the NN. It's now time to backpropagate the gradients.
        : b.start(b.N - 1)) // This is a static statement that tells the compiler to start backpropagation from the last layer.
        // Once the backpropagation is done, you can use the gradients to update the weights of the NN.
        // So... how does one know how many signals to backpropagate? 
        // We need to start from the most recent layer and work our way back to the first layer.
        // The third layer has 3 inputs and only 1 neuron. So, we would have 3 weights and 1 bias to update. (total: 4)
        ~ (si.block(1), si.bus(4))) // Auto skip 1 signal -- that's the loss signal.
        // The second layer has 2 inputs and 3 neurons. So, we would have 2*3 = 6 weights and 3 biases to update. (total: 9)
        ~ (si.block(5), si.bus(9))) // Now, we have 5 signals (from the first back-prop) to skip. (1+4)
        // The first layer has 1 input and 2 neurons. So, we would have 1*2 = 2 weights and 2 biases to update. (total: 4)
        ~ (si.block(14), si.bus(4)) // Now, we have 14 signals (from the second back-prop) to skip. (1+4+9)
        with {
            // To define the backpropagation environment, it needs to understand the variables that are being used in the NN.
            // As a result, we start from the last layer and work our way back to the first layer.
            // The arguments to this environment are a list of the number of neurons, number of inputs per layer.
            // For the last layer, we have 1 neuron and 3 inputs.
            // For the second last layer, we have 3 neurons and 2 inputs.
            // For the third last layer, we have 2 neurons and 1 input.
            // As a result, the arguments to this function would be (1, 3, 3, 2, 2, 1).
            b = df.backpropNN((1, 3, 3, 2, 2, 1));
        };
