import("stdfaust.lib");
df = library("diff.lib");

NVARS = 1;
hiddenCutoff = hslider("[0]cutoff [scale:log]", 440, 10, 20000, .1);

lpf(fc) = _
    <: *(b0), (mem : *(b1))
    // :> + ~ *(0-a1)
    : + : + ~ (_,(0,a1 : -) : *)
with {
    w = 2*ma.PI*fc;
    c = 1/tan(w*0.5/ma.SR);
    d = 1+c;
    b0 = 1/d;
    b1 = 1/d;
    a1 = (1-c)/d;
};

// diffvars(vars) = environment {
//     N = outputs(vars);
//     i = nth(vars,i),dvec(N,i);
//     dvec(N,i) = par(j,i,0),1,par(k,N-i-1,0);
// };

// vars = diffVars((...));

process = no.noise,no.noise
    : (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : groundTruth,learnable
        // Copy ground truth and learnable outputs to audio outputs.
        : (_ <: _,_),(_ <: _,_),si.bus(NVARS)
        : route(4+NVARS,4+NVARS,(1,1),(2,NVARS+3),(3,2),(4,NVARS+4),par(n,NVARS,(n+5,n+3)))
        //
        : learn,_,_) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(NVARS),_,_
with {
    groundTruth = lpf(hiddenCutoff);

    // learnable = df.input(NVARS),df.var(1,cutoff,NVARS) : df.diff(+,NVARS)
    learnable = learnableLPF
    with {
        // learnableLPF = df.input(NVARS)...
        learnableLPF = df.input(NVARS),si.bus(NVARS)
            <: (df.diff(_,NVARS),b0 : df.diff(*,NVARS)),(df.diff(mem,NVARS),b1 : df.diff(*,NVARS))
            : df.diff(+,NVARS)
            : (route(4,6,(1,1),(3,2),(2,3),(4,4)) : +,+,_,_) ~
                // (_,_)
                (a1,df.diff(_,NVARS) : df.diff(*,NVARS)) : _,_,!,!
                // a1
                // par(n,2,_,2 : *)
                // (df.diff(*,NVARS))
        with {
            cutoff = -~_ <: attach(hbargraph("[1]Learned cutoff",0,1000));
            w = df.const(2*ma.PI,NVARS),df.var(1,cutoff,NVARS) : df.diff(*,NVARS);
            c = df.const(1,NVARS),df.const(.5/ma.SR,NVARS),w
                : df.diff(_,NVARS),df.diff(*,NVARS)
                : df.diff(_,NVARS),df.diffTAN(NVARS)
                : df.diff(/,NVARS);
            d = df.const(1,NVARS),c : df.diff(+,NVARS);
            b0 = df.const(1,NVARS),d : df.diff(/,NVARS);
            b1 = df.const(1,NVARS),d : df.diff(/,NVARS);
            a1 = df.const(1,NVARS),c,d : df.diff(-,NVARS),df.diff(_,NVARS) : df.diff(/,NVARS);
        };
    };

    learn =
        // Window the input signals
        par(i,2+NVARS,window)
        // Swap the order of ground truth and learnable inputs
        : ro.cross(2),pds
        // Calculate loss (this is just for show, since there's no sensitivity threshold)
        // TODO: loss against batches of input, take abs of difference, normalise params...
        : (- <: loss,_),pds
        // Calculate gradients
        : _,gradients
        // Scale gradients by the learning rate
        : _,par(n,NVARS,_,alpha : *)
    with {
        // Learning rate
        alpha = 1e-2;
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
};
