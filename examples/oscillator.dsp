import("stdfaust.lib");
df = library("diff.lib");

NVARS = 1;

p = df.osc(f0,NVARS)
with {
    f0 = -~_
        // Map [-1,1] to range
        <: attach(hbargraph("[1]Normalised freq",-1.,1.))
        : (_,1.,mid : +,_ : (*,mini : +))
        <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
    with {
        maxi = 10000.;
        mini = 50.;
        mid = maxi,mini,2 : +,_ : /;
    };
};

process = hgroup("Differentiable oscillator",
        vgroup("[0]Parameters", vgroup("Hidden", truth),vgroup("Learned", learnable))
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.learnL1(1<<0,5e-6,NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(NVARS),!,_
with {
    hiddenF0 = hslider("freq [scale:log]", 440.,50.,1000.,.01);
    // truth = os.osc(hiddenF0);

    truth = phasor(hiddenF0) : sin
    with {
        decimalPart(x) = x-int(x);
        phasor(f) = f/ma.SR : (+ : decimalPart) ~ _ : *(2 * ma.PI);
    };

    learnable = osc(df.var(1,f0,NVARS),NVARS)
    with {
        f0 = -~_
            // Map [-1,1] to range
            <: attach(hbargraph("[1]Normalised freq",-1.,1.))
            : (_,1.,mid : +,_ : (*,mini : +))
            <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
        with {
            maxi = 1000.;
            mini = 50.;
            mid = maxi,mini,2 : +,_ : /;
        };
    };

    phasor(f0,nvars) = f0,df.diff(ma.SR,nvars)
        : df.diff(/,nvars)
        : df.rec(f~g,0),df.diff(2*ma.PI,nvars)
        : df.diff(*,nvars)
        with {
            f = df.diff(+,nvars) <: df.diff(_,nvars),df.diff(int,nvars) : df.diff(-,nvars);
            g = df.diff(_,nvars);
        };

    osc(f0,nvars) = phasor(f0,nvars) : df.diff(sin,nvars);

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
