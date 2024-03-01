
import("stdfaust.lib");
df = library("../lib/diff.lib");

hiddenGain = hslider("[0]Hidden gain", .5, 0, 2, .01);
hiddenDC = hslider("[1]Hidden dc", .5, -1, 1, .01);

gainDC(g, d) = _,g,d : *,_ : +;
groundTruth = gainDC(hiddenGain, hiddenDC);

NVARS = 2;

//===========================================================================

// Differentiable operators.

// Differentiable add.
// Takes two dual numbers as input;
// returns the sum and its derivative.
// <u, u'> + <v, v'> = <u+v, u'+v'>
diff(+) =
    route(nIN,nOUT,
        (u,1),(v,2), // u + v
        par(n,NVARS,
            (u+n+1,dx),(v+n+1,dx+1) // du/dx_n + dv/dx_n
            with {
                dx = 2*n+3; // Start of derivatives wrt nth var
            }
        )
    )
    with {
        nIN = 2 + 2*NVARS;
        nOUT = nIN;
        u = 1;
        v = 1 + NVARS + 1;
    }
    : +,par(n, NVARS, +);

// Differentiable multiplier.
// Takes two dual numbers as input;
// returns the product its derivative.
// <u, u'> * <v, v'> = <u*v, u'*v + u*v'>
diff(*) =
    route(nIN,nOUT,
        (u,1),(v,2), // u * v
        par(n,NVARS,
            (u,dx),(dvdx,dx+1),   // u * dv/dx_n
            (dudx,dx+2),(v,dx+3)  // du/dx_n * v
            with {
                dx = 4*n+3; // Start of derivatives wrt nth var
                dudx = u + n + 1;
                dvdx = v + n + 1;
            }
        )
    )
    with {
        nIN = 2 + 2*NVARS;
        nOUT = 2 + 4*NVARS;
        u = 1;
        v = 1 + NVARS + 1;
    }
    : *,par(n, NVARS, *,* : +);

// Differentiable sine function
// Takes a dual number as input;
// Returns the sine of the input, and its derivative.
// sin(<u, u'>) = <sin(u), u'*cos(u)>
diffSIN = route(NVARS+1,2*NVARS+1,(1,1),par(n,NVARS,(1,2*(n+1)+1),(n+2,2*(n+1))))
    : sin,par(n,NVARS, _,cos : *);

// A differentiable variable.
// Returns a list containing the variable and a vector of its partial derivatives.
// [ xi, dxi/dx1, dxi/dx2, ... , dxi/dxN ]q
diffVar(i,var) = var,par(n,NVARS,n+1==i); // v,diff(v);

// Derivative of the identity function.
// Returns n+1 parallel identity functions, where n is the number of variables.
diff(_) = si.bus(NVARS+1);

// A differentiable (audio) input.
// Returns the input, plus its partial derivatives wrt all variables in the system, which are all zero.
diffInput = _,par(n,NVARS,0);

//=============================================================================================

// Loss and gradient calculator.
// Inputs:
// - ground truth output
// - learnable output
// - NVARS partial derivatives of learnable output
// Outputs:
// - loss
// - NVARS parameter gradients
learn =
    // Window the input signals
    par(i,2+NVARS,window)
    // Swap the order of ground truth and learnable inputs
    : ro.cross(2),pds
    // Calculate loss (this is just for show, since there's no sensitivity threshold)
    : (- <: loss,_),pds
    // Calculate gradients
    : _,gradients
    // Scale gradients by the learning rate
    : _,par(n,NVARS,_,alpha : *)
with {
    // Learning rate
    alpha = .01;
    // Window function
    windowSize = 1<<5;
    window = ba.slidingMean(windowSize);
    // Loss function (L2 norm)
    loss = ^(2) <: attach(hbargraph("[3]loss",0,.05));
    // A way to move the partial derivatives around.
    pds = si.bus(NVARS);
    // Calculate gradients; for L2 norm: 2 * dy/dx_i * (learnable - groundtruth)
    gradients = _,par(n,NVARS, _,2 : *)
        : routeall
        : par(n,NVARS, * <: attach(hbargraph("[4]gradient %n",-.5,.5)));

    // A utility to duplicate the first input for combination with all remaining inputs.
    routeall = _,si.bus(NVARS)
        : route(NVARS+1,NVARS*2,par(n,NVARS,(1,2*n+1),(n+2,2*(n+1))));
};

//=============================================================================================

process =
    no.noise,no.noise
    // no.noises(2,0),no.noises(2,1)
    // os.osc(100),os.osc(100)
    // Route inputs to ground truth and learnable algos; route gradients to learnable algo.
    : (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : groundTruth,learnable // Could use si.bus here...
        // Copy ground truth and learnable outputs to audio outputs.
        : (_ <: _,_),(_ <: _,_),si.bus(NVARS)
        : route(4+NVARS,4+NVARS,(1,1),(2,NVARS+3),(3,2),(4,NVARS+4),par(n,NVARS,(n+5,n+3)))
        // Recurse gradients
        // : learn,_,_) ~ (!,si.bus(NVARS))
        : df.learn(1<<5,(.01),NVARS),_,_) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(NVARS),_,_
with {
    learnable = diffInput,diffVar(1,gain),diffVar(2,dc)
        : diff(*),diff(_)
        : diff(+)
    with {
        // Gradient descent is applied here
        gain = -~_ <: attach(hbargraph("[5]Learned gain",0,2));
        dc = -~_ <: attach(hbargraph("[6]Learned dc",-1,1));
    };
};
