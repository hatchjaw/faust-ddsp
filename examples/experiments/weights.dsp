import("stdfaust.lib");

// example
N = 10;
in = par(i, N, 1.0 <: hgroup(hbargraph("input%i", 0, 1)));
label = 1;
// n = number of features
// in = input signal (n size)
// w = weights (n size)
// b = bias (n size) - we wont be introducing this rn
neuron(in, w, n, label) = routeall
                    : par(i, n, (_, _ : *))
                    // : routeall
                    // : par(i, n, (_, _ : +))
                    : sum(i, n, _)
                    : sigmoid <: attach(hbargraph("output", 0, 1))
                    : lossL1(_, label) <: attach(hbargraph("loss", 0, 1))
                    : gradL1 <: attach(hbargraph("grad", -1, 1))
                    with {
                        routeall = route(n, 2*n, par(i, n, (i, dx), (i+n+1, dx+1))
                            with {
                                dx = 2*i+1;
                            }
                        )
                    };

// Core weights and biases, with visualization; needs n inputs (gradients)
weights(n) = par(i, n, _ : -~_ <: attach(hbargraph("weight%i", -1, 1)));
// bias(n) = par(i, n, _ : -~_ <: attach(hbargraph("bias%i", -1, 1)))

// Activation functions
// Sigmoid
sigmoid(x) = 1, (1, exp(-x) : +) : /;

// Loss functions (assume dy/dx_i = 1) (grad: (dy/dx_i) * loss / abs(loss))
lossL1(pred, label) = pred, label : - : abs;
gradL1(loss) = ((loss <: _ : *, _ : abs) : /);

process = neuron(in, weights(N), N, label);