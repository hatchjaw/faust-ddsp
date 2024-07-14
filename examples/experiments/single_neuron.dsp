import("stdfaust.lib");

N = 3; // hidden inputs
learningRate = 0.5;

// input-label vars
in = par(i, N, hslider("input%i", -0.2, -1, 1, 0.001));
label = 1;

// trying to slow down the process
// outputs: current epoch, update status (1 or 0)
epoch_calculator = (_ ~ +(1)) : ((_ % 10000 == 0),0,1 : select2) <: +(_)~_, _
                : (_ <: attach(hbargraph("epochs", 0, 100))), _;                

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

process = epoch_calculator
        : _, (route(N+1, 2*N, par(i, N, (N+1, 2*i+1), (i+1, 2*(i+1))))
        : par(i, N, (_, 0, _ : select2))
        : vgroup("Output layer [one neuron]", (in, weights(N) : neuron(N)))
        : (_ <: _, dSigmoid), par(i, N, _)
        : (-(label) <: lossL1, dLossL1), _, par(i, N, _)
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
        