import("stdfaust.lib");

// <u, u'> + <v, v'> = <u+v, u'+v'>
diff(+) = +,+;

diffADD(nvars) =
    route(nIN,nOUT,
        (u,1),(v,2), // u + v
        par(n,nvars,
            (u+n+1,dx),(v+n+1,dx+1) // du/dx_n + dv/dx_n
            with {
                dx = 2*n+3; // Start of derivatives wrt nth var
            }
        )
    )
    with {
        nIN = 2 + 2*nvars;
        nOUT = nIN;
        u = 1;
        v = 1 + nvars + 1;
    }
    : +,par(n, nvars, +);

// <u, u'> - <v, v'>' = <u-v, u'-v'>
diff(-) = -,-;

// <u, u'> * <v, v'> = <u*v, u'*v + u*v'>
// diff(*) = _,_,_,_ <: (_,!,!,!),(!,!,_,!),(!,_,!,!),(!,!,_,!),(_,!,!,!),(!,!,!,_) : *,(*,*: +);
diff(*) = route(4,6,(1,1),(3,2),(2,3),(3,4),(1,5),(4,6)) : *,(*,*: +);

diffMUL(nvars) =
    route(nIN,nOUT,
        (u,1),(v,2), // u * v
        par(n,nvars,
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
        nIN = 2 + 2*nvars;
        nOUT = 2 + 4*nvars;
        u = 1;
        v = 1 + nvars + 1;
    }
    : *,par(n, nvars, *,* : +);

// <u, u'> / <v, v'> = <u*v, (u'*v - u*v') / v^2>
// diff(/) = _,_,_,_ <: (_,!,!,!),(!,!,_,!),(!,_,!,!),(!,!,_,!),(_,!,!,!),(!,!,!,_),(!,!,_,!) : /,((*,* : -),^(2) : /);
diff(/) = route(4,7,(1,1),(3,2),(2,3),(3,4),(1,5),(4,6),(3,7)) : /,((*,* : -),^(2) : /);

// diff(n) = n;
diff(sin(x)) = sin(x),(diff(x),cos(x): *);

diff(_) = 0;

// diff(x) = 1,0;

// A differentiable variable.
// Returns a list containing a variable and a vector of its partial derivatives.
// [ xi, dxi/dx1, dxi/dx2, ... , dxi/dxN ]
diffVar(i,nvars,var) = var,par(n,nvars,n+1==i); // v,diff(v);

// Derivative of the identity function.
// Returns n+1 parallel identity funcitons, where n is the number of variables.
diffWire(nvars) = par(n,nvars+1,_); // si.bus(nvars+1)

// A differentiable (audio) input.
// Returns the input, plus its partial derivatives wrt all variables in the system, which are all zero.
diffInput(nvars) = _,par(n,nvars,0);

//=============================================================================================

gradient(nvars) =
    // Window the input signals
    par(i,2+nvars,w)
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
    pds = si.bus(nvars);
    // Calculate gradients; for l2norm: 2 * dy/dx_i * (learnable - groundtruth)
    gradients = _,par(n,nvars, _,2 : *)
        : routeall
        : par(n,nvars, * <: attach(hbargraph("gradient %n",-2,2)));

    routeall = _,si.bus(nvars)
        : route(nvars+1,nvars*2,par(n,nvars,(1,2*n+1),(n+2,2*(n+1))));
};

//=============================================================================================

hiddenGain = .9;
hiddenDC = -.15;
gainDC(g, d) = _,g,d : *,_ : +;
groundTruth = gainDC(hiddenGain, hiddenDC);

gain = hslider("[0]gain", .5, 0, 2, .01);
dc = hslider("[1]dc", 0, -1, 1, .01);
q = hslider("[2]q", 0, -1, 1, .01);
NVARS = 2;

// process = diffVar(_),diffVar(gain) : diff(+),!;

// process = _,diff(_),gain,diff(gain) : diff(*),dc,diff(dc) : route(4,4,(1,1),(2,3),(3,2),(4,4)) : diff(+);
// process = _ <: groundTruth,diffVar(_),diffVar(gain),diffVar(dc) :
//     _,diff(*),!,_,_,_ :
//     _,route(4,4,(1,1),(2,3),(3,2),(4,4)),! :
//     _,diff(+)
// with {
//     gainDC(g, d) = _,g,d : *,_ : +;
//     groundTruth = gainDC(hiddenGain, hiddenDC);
//     learnable = gainDC(gain, dc);
// };

learnable = diffInput(NVARS),diffVar(1,NVARS,gain),diffVar(2,NVARS,dc)
    : diffMUL(NVARS),diffWire(NVARS)
    : diffADD(NVARS);

learnable3 = diffInput(NVARS),diffVar(1,NVARS,gain),diffVar(2,NVARS,dc),diffVar(3,NVARS,q)
    : diffMUL(NVARS),diffWire(NVARS),diffWire(NVARS)
    : diffADD(NVARS),diffWire(NVARS)
    : diffADD(NVARS);

process = no.noise,no.noise : groundTruth,learnable
    : _,(_ <: _,_),si.bus(NVARS)
    : route(NVARS+3,NVARS+3,(1,1),(2,2),(3,NVARS+3),par(n,NVARS,(n+4,n+3)))
    : gradient(NVARS),_;
