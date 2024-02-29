import("stdfaust.lib");
df = library("diff.lib");

NVARS = 2;
MAXDELAY = 100;

q = df.input(NVARS),df.const(2,NVARS) : df.diff(@,NVARS);

process = p;

//=====================================================================================

test = f,g,h : df.diff(@,nvars),df.diff(_,nvars) : df.diff(*,nvars)
with {
    nvars = 2;
    delay = (-~_),0,1<<24 : max,_ : min;
    gain = -~_;

    f = df.input(nvars);
    g = df.var(1,delay,nvars);
    h = df.var(2,gain,nvars);
};

//=====================================================================================

p = in,in
    : hgroup("Differentiable delay",
        (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : vgroup("[0]Parameters", vgroup("Hidden", truth),vgroup("Learned", learnable))
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", learn(1<<3,2e-1,NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    ) : _,si.block(NVARS),_,_
with {
    in = no.noise; //os.osc(1000);

    hiddenDelay = hslider("Delay", 10, 0, MAXDELAY, 1);
    hiddenGain = hslider("Gain", .5, 0., 2., .01);
    truth =
        // _,hiddenGain : *;
        // _,hiddenDelay : @;
        _,hiddenDelay,hiddenGain : @,_ : *;

    learnable =
        // df.input(NVARS),df.var(1,gain,NVARS) : df.diff(*,NVARS)
        // df.input(NVARS),df.var(1,delay,NVARS) : df.diff(@,NVARS)
        df.input(NVARS),df.var(1,delay,NVARS),df.var(2,gain,NVARS) : df.diff(@,NVARS),df.diff(_,NVARS) : df.diff(*,NVARS)
    with {
        delay = (-~_),0,1<<24 : max,_ : min <: attach(hbargraph("Delay", 0, MAXDELAY));
        gain = -~_ <: attach(hbargraph("Gain",0,2));
    };

    learn(windowSize, learningRate, nvars) =
        // Window the input signals
        par(i,2+nvars,window)
        // Calculate the difference between the ground truth and learnable outputs
        // (Is cross necessary?)
        : (ro.cross(2) : - ),pds
        // Calculate loss (this is just for show, since there's no sensitivity threshold)
        : (_ <: loss,_),pds
        // Calculate gradients
        : _,gradients
        // Scale gradients by the learning rate
        : _,par(n,nvars,_,learningRate : *)
    with {
        window = ba.slidingMean(windowSize);
        // Loss function (L2 norm)
        loss = ^(2) <: attach(hbargraph("[100]loss",0,.05));
        // A way to move the partial derivatives around.
        pds = si.bus(nvars);
        // Calculate gradients; for L2 norm: 2 * dy/dx_i * (learnable - groundtruth)
        gradients = _,par(n,nvars, _,2 : *)
            : routeall
            : par(n,nvars, * <: attach(hbargraph("[101]gradient %n",-.5,.5)));

        // A utility to duplicate the first input for combination with all remaining inputs.
        routeall = _,si.bus(nvars)
            : route(nvars+1,nvars*2,par(n,nvars,(1,2*n+1),(n+2,2*(n+1))));
    };
};
