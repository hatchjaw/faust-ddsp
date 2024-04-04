import("stdfaust.lib");

hiddenGain = .9;
hiddenDC = -1;
gainDC(g, d) = _,g,d : *,_ : +;
groundTruth = gainDC(hiddenGain, hiddenDC);

gain = hslider("[0]gain", .5, 0, 2, .01);
dc = hslider("[1]dc", 0, -3.14, ma.PI, .01);
NVARS = 2;

//===========================================================================

// Differentiable operators.

// Differentiable add.
// Takes two dual numbers as input; returns their sum.
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
// Takes two dual numbers as input; returns their product.
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

// sin(<u, u'>) = <sin(u), u'*cos(u)>
diffSIN = route(NVARS+1,2*NVARS+1,(1,1),par(n,NVARS,(1,2*(n+1)+1),(n+2,2*(n+1))))
    : sin,par(n,NVARS, _,cos : *);

// A differentiable variable.
// Returns a list containing a variable and a vector of its partial derivatives.
// [ xi, dxi/dx1, dxi/dx2, ... , dxi/dxN ]
diffVar(i,var) = var,par(n,NVARS,n+1==i); // v,diff(v);

// Derivative of the identity function.
// Returns n+1 parallel identity funcitons, where n is the number of variables.
diff(_) = si.bus(NVARS+1);

// A differentiable (audio) input.
// Returns the input, plus its partial derivatives wrt all variables in the system, which are all zero.
diffInput = _,par(n,NVARS,0);

//=============================================================================================

// Loss and gradient calculator.
gradient =
    // Window the input signals
    par(i,2+NVARS,w)
    // Swap the order of ground truth and learnable inputs
    : ro.cross(2),pds
    // Calculate loss
    : (- <: loss,_),pds
    : _,gradients
with {
    // Learning rate
    alpha = .01;
    // Window function
    windowSize = 1<<10;
    w = ba.slidingRMS(windowSize);
    // Loss functions
    l1norm = abs;
    l2norm = ^(2);
    // Chosen loss function
    loss = l2norm <: attach(hbargraph("[3]loss",0,2));
    // A way to move the partial derivatives around.
    pds = si.bus(NVARS);
    // Calculate gradients; for l2norm: 2 * dy/dx_i * (learnable - groundtruth)
    gradients = _,par(n,NVARS, _,2 : *)
        : routeall
        : par(n,NVARS, * <: attach(hbargraph("gradient %n",-2,2)));

    routeall = _,si.bus(NVARS)
        : route(NVARS+1,NVARS*2,par(n,NVARS,(1,2*n+1),(n+2,2*(n+1))));
};

//=============================================================================================

learnable = diffInput,diffVar(1,gain),diffVar(2,dc)
    : diff(*),diff(_)
    : diff(_),diffSIN
    : diff(+);

process = no.noise,no.noise : groundTruth,learnable
    // Route the learnable output out of the way
    : _,(_ <: _,_),si.bus(NVARS)
    : route(NVARS+3,NVARS+3,(1,1),(2,2),(3,NVARS+3),par(n,NVARS,(n+4,n+3)))
    // Calculate loss and gradients
    : gradient,_;
