import("stdfaust.lib");
df = library("diff.lib");

// Attempting to learn the frequency of an oscillator in the time domain...
// Broken, but the loss function produces some pretty cool noises.

NVARS = 1;
MAXFREQ = 5000.;
MINFREQ = 20.;

// p = df.osc(f0,NVARS)
// with {
//     f0 = -~_
//         // Map [-1,1] to range
//         <: attach(hbargraph("[1]Normalised freq",-1.,1.))
//         : (_,1.,mid : +,_ : (*,mini : +))
//         <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
//     with {
//         maxi = 10000.;
//         mini = 50.;
//         mid = maxi,mini,2 : +,_ : /;
//     };
// };

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
        : vgroup("[1]Loss/gradient", learn(1<<0,hslider("alpha [scale:log]",1e-4,1e-8,1e-1,1e-8),NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Listen to the loss.
    : _,si.block(NVARS),!,! <: _,_
with {
    truth = os.osc(hslider("freq [scale:log]", 500.,MINFREQ,MAXFREQ,.01));

    learnable = osc(df.var(1,f0,NVARS),NVARS)
    with {
        f0 = -~_
            // Map [-1,1] to range
            <: attach(hbargraph("[1]Normalised freq",-1.,1.))
            : (_,1.,mid : +,_ : (*,mini : +))
            <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
        with {
            maxi = MAXFREQ;
            mini = MINFREQ;
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
        // "Loss" is just the difference between "truth" and learnable output
        loss = _ <: attach(hbargraph("[100]loss",-5,5));
        // A way to move the partial derivatives around.
        pds = si.bus(nvars);
        // Calculate gradients; for L2 norm: 2 * dy/dx_i * (learnable - groundtruth)
        gradients = _,par(n,nvars, _,2 : *)
            : routeall
            : par(n,nvars, * <: attach(hbargraph("[101]gradient %n",-1000,1000)));

        // A utility to duplicate the first input for combination with all remaining inputs.
        routeall = _,si.bus(nvars)
            : route(nvars+1,nvars*2,par(n,nvars,(1,2*n+1),(n+2,2*(n+1))));
    };
};
