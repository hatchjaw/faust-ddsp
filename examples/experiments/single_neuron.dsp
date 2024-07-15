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
// NB. maybe a generalized way to change the label can be figured out, depending on the number of epochs (atm: every sample is available for N_eps epochs)
// Outputs: sample, label (0 = osc, 1 = square), change_bit
dataset_generator(change_bit) = change_bit <: _, _
                                : (change_bit, 0, 1 : select2), _
                                : (+(_)~((_ <: _ % N_eps == 0, _), 0 : select2)), _
                                : (_ <: attach(hbargraph("local-epoch", 0, 10))), _
                                : (((_ % N_eps == 0, 0, 1) : select2) : ba.toggle <: attach(hbargraph("label", 0, 1))), _
                                : (_ <: (_, os.osc(440), os.square(440) : select2), _), _;

// n = number of features
// in = input signal (n size)
// w = weights (n size)
// b = bias (n size) - we wont be introducing this rn
// update bit = update weights or no
// we need to duplicate the input signal to the end, we need them for gradient calculation
neuron(n) = grad_route
            : routeall, grad_flow
            : par(i, n, *), grad_flow
            // : routeall
            // : par(i, n, (_, _ : +))
            : sum(i, n, _), grad_flow
            : (sigmoid <: attach(hbargraph("output", 0, 1))), grad_flow
            with {
                routeall = route(2*n, 2*n, par(i, n, (i+1, dx), (i+n+1, dx+1)
                        with {
                            dx = 2*i+1;
                        }
                    )
                );

                grad_route = route(2*n, 3*n, par(i, n, (i+1, i+1), (i+1, i+1+2*n)), par(i, n, (i+n+1, i+n+1)));
                grad_flow = par(i, n, _);
            };

// Core weights and biases, with visualization; needs n inputs (gradients)
weights(n) = par(i, n, _ : *(learningRate) : -~_ <: attach(hbargraph("weight%i", -1, 1)));
// bias(n) = par(i, n, _ : -~_ <: attach(hbargraph("bias%i", -1, 1)))

// Activation functions
// Sigmoid
sigmoid(x) = 1, (1, exp(-x) : +) : /;

// Activation functions' derivatives
// sig_x : sigmoid of x.
dSigmoid(sig_x) = sig_x * (1 - sig_x);

// Loss functions 
lossL1 = abs <: attach(hbargraph("loss", -0.5, 0.5));
dLossL1 = (_ <: _, (abs, ma.EPSILON : max)) : /;

//process = epoch_calculator : _, dataset_generator(_);
process = epoch_calculator : _, dataset_generator(_)
        : _, (route(N+3, N+3, (N+1,1),(N+2,2),(N+3,N+3),par(i,N,(i+1,i+3)))
        : _, _, route(N+1, 2*N, par(i, N, (N+1, 2*i+1), (i+1, 2*(i+1))))
        : _, _, par(i, N, (_, 0, _ : select2))
        : _, route(N+1, N+1, (1, N+1), par(i, N, (i+2, i+1)))
        : vgroup("Output layer [one neuron]", (_, weights(N) : neuron(N))), _
        : (_ <: _, dSigmoid), par(i, N, _), _
        : route(N+3, N+3, (1,1),(N+3,2),(2,3),par(i,N,(i+3,i+4)))
        : (- <: lossL1, dLossL1), _, par(i, N, _)
        : vgroup("backprop", _, *, par(i, N, _) : _, routeGrads : _, par(i, N, * <: attach(hbargraph("grad%i", -0.5, 0.5)))))
        ~(si.block(1), par(i, N, _))
        with {
            routeGrads = route(N+1, 2*N, par(i, N, (1, dx), (i+2, dx+1)
                    with {
                        dx = 2*i + 1;
                    }
                )
            );
        };
        