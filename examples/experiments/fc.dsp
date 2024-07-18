import("stdfaust.lib");

// N = 1; this is coded into the dataset generator
N = 1;
N_eps = 5; // N_eps epochs per sample
learningRate = 0.5;


// trying to slow down the process
// outputs: current epoch, update status (1 or 0)
epoch_calculator = (_ ~ +(1)) : ((_ % 10000 == 0),0,1 : select2) <: +(_)~_, _
                : (_ <: attach(hbargraph("epochs", 0, 100))), _;  

// a simulation of how a dataset would look
// NB. maybe a generalized way to change the label can be figured out, depending on the number of epochs (atm: every sample is training on for N_eps epochs)
// Outputs: sample, label (0 = osc, 1 = square), change_bit
dataset_generator(change_bit) = change_bit <: _, _
                                : (change_bit, 0, 1 : select2), _
                                : (+(_)~((_ <: _ % N_eps == 0, _), 0 : select2)), _
                                : (_ <: attach(hbargraph("local-epoch", 0, 10))), _
                                : (((_ % N_eps == 0, 0, 1) : select2) : ba.toggle <: attach(hbargraph("label", 0, 1))), _
                                : (_ <: (_, 0, 1 : select2), _), _;
// mathematical helpers
// neuron-wise mathematical functions

// Matrix Multiplication (Nx1) dims
// in should be N signals; weights should be N signals; biases should be 1 signal (N = number of inputs to each neuron)
// Usage: in, weights, biases : mat_mul(N)
mat_mul(N) = route(2*N, 2*N, par(i, N, (i+1, dx), (i+N+1, dx+1)
        with {
            dx = 2*i+1;
        })), _
        : par(i, N, *), _
        : sum(i, N, _), _
        : +;

// Activation functions and their respective derivatives;
// z refers to the wx + b; out refers to the sigmoided output from the neuron.

// Sigmoid
sigmoid(x) = 1, (1, exp(-x) : +) : /;
dSigmoid(out) = out * (1 - out);

// Loss functions and their respective derivatives; these will produce both the loss and gradient.
// Output: loss, gradient (dL/dZ = dL/dY * derivative of activation function)
// NB. This will only be used for the last neuron of the network. 
// Inputs: Prediction (Y), Truth (Label)
lossMAE(pred, truth) = (pred - truth) 
                    : _ <: _, _
                    : abs, (_ <: _, (_ : abs))
                    : _, /;

// backpropagation per neuron
// Inputs needed: dL/dY (from the last neuron's backprop), Y, W, X (of that neuron), dL/dX (if available)
// Backpropagation goes from the last neuron all the way to the first neuron, so we need to ensure that the last neuron does its work well.

// last neuron backprop
// Inputs: dL/dY (1 valued-vector; from loss function), Y (1 vector), X (N vector), W (N vector)
// Outputs: dL/dW (N vector; for backprop), dL/db (1 vector; for backprop), dL/dX (N vector; for next backprop)
// Usage: dL/dZ, W, X : backpropLast(N, der_activationFn)
backpropLast(N, der_activationFn) = _, der_activationFn, par(i, 2*N, _)
                : *, par(i, 2*N, _)
                : route(2*N+1, 4*N+1, par(i, N, (1,2*i+1), (i+2,2*(i+1))), (1,2*N+1), par(j, N, (1,2*j+2*N+2), (j+N+2,2*j+2*N+3)))
                : par(i, N, *), _, par(i, N, *);

// regular neuron backprop
// Inputs: dL/dY (n vector; from last neuron (was dL/dX) - need to sum and reduce to 1 vector), Y (1 vector), X (N vector), W (N vector)
// Output: dL/dW (N vector; for backprop), dL/db (1 vector; for backprop), dL/dX (N vector; for next backprop)
// Usage: dL/dX, X, W, Y : backprop(n, N, der_activationFn)
backprop(n, N, der_activationFn) = sum(i, n, _), der_activationFn, par(i, 2*N, _)
            : *, par(i, 2*N, _)
            : route(2*N+1, 4*N+1, par(i, N, (1,2*i+1), (i+2,2*(i+1))), (1,2*N+1), par(j, N, (1,2*j+2*N+2), (j+N+2,2*j+2*N+3)))
            : par(i, N, *), _, par(i, N, *);

// Generic weights (n size) and bias (1 size)
weights(n) = par(i, n, _ : *(learningRate) : -~_ <: attach(hbargraph("weight%i", -1, 1)));
bias = _ : *(learningRate) : -~_ <: attach(hbargraph("bias", -1, 1));

// a neuron
// Inputs: in (N vectors), W (N vectors), b (1 vector)
// Outputs: Y, X, W
// Usage: in, W, b : neuron(N, activationFn)
neuron(N, activationFn) = si.bus(N), weights(N), bias
                        : route(2*N+1, 4*N+2, par(i, 2*N+1, (i+1,i+1), (i+1,i+2*N+2)))
                        : ((par(i, 2*N+1, _): mat_mul(N)) : activationFn), par(i, 2*N, _), !;

// fc layer (not last layer)
// Inputs: w_grads(N*n vectors; N vectors per neuron), b_grads(n vectors), in (N vectors), number of neurons (n)
// Outputs (need to make it useful for backprop): (Y, X, W) (2N+1 * n vectors) (need to route backprop well here)
fc(N, n, activationFn) = route(N+n*N+n, n*(2*N+1),
                        par(i, n, par(j, N, ((j+1)+N*i, (N+1+j)+(2*N+1)*i))),
                        par(i, n, (i+n*N+1, (i+1)*(2*N+1))),
                        par(i, n, par(j, N, (j+1+n*N+n, (2*N+1)*i+(j+1)))))
                        : par(i, n, (par(j, 2*N+1, _) : neuron(N, activationFn)));

process = si.bus(2) : fc(2, 2, sigmoid)~si.bus(6);
// process = epoch_calculator : _, dataset_generator(_)
//         : _, (route(N+3, N+3, (N+1,1),(N+2,2),(N+3,N+3),par(i,N,(i+1,i+3)))
//         : _, _, route(N+1, 2*N, par(i, N, (N+1, 2*i+1), (i+1, 2*(i+1))))
//         : _, _, par(i, N, (_, 0, _ : select2))
//         : _, route(N+1, N+1, (1, N+1), par(i, N, (i+2, i+1)))
//         : vgroup("Output layer [one neuron]", (_, weights(N) : neuron(N))), _
//         : (_ <: _, dSigmoid), par(i, N, _), _
//         : route(N+3, N+3, (1,1),(N+3,2),(2,3),par(i,N,(i+3,i+4)))
//         : (- <: lossL1, dLossL1), _, par(i, N, _)
//         : vgroup("backprop", _, *, par(i, N, _) : _, routeGrads : _, par(i, N, * <: attach(hbargraph("grad%i", -0.5, 0.5)))))
//         ~(si.block(1), par(i, N, _))
//         with {
//             routeGrads = route(N+1, 2*N, par(i, N, (1, dx), (i+2, dx+1)
//                     with {
//                         dx = 2*i + 1;
//                     }
//                 )
//             );
//         };
        