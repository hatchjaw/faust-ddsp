import("stdfaust.lib");
df = library("diff.lib");

// Attempting to learn the frequency of an oscillator in the time domain...
// Broken, but the loss function produces some pretty cool noises.

MAXFREQ = 5000.;
MINFREQ = 20.;

// p = df.osc(f0,vars.N)
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
        : route(2+vars.N,4+vars.N,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,vars.N+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,vars.N+4),
            // Route gradients to df.learn.
            par(n,vars.N,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", learn(1<<0,hslider("alpha [scale:log]",1e-4,1e-8,1e-1,1e-8)),_,_)) ~ (!,si.bus(vars.N))
    // Listen to the loss.
    : _,si.block(vars.N),!,!
    : _,en.adsr(.05,.05,1    ,.9,gate)
    with {
        gate = button("gate");
    } : *
    : ma.tanh <: _,_
with {
    truth = os.osc(hslider("freq [scale:log]", 500.,MINFREQ,MAXFREQ,.01));

    vars = df.vars((f0))
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

    learnable = df.env(vars).osc(vars.var(1));

    learn(windowSize, learningRate) =
        // Window the input signals
        par(i,2+vars.N,window)
        // Calculate the difference between the ground truth and learnable outputs
        // (Is cross necessary?)
        : (ro.cross(2) : - ),pds
        // Calculate loss (this is just for show, since there's no sensitivity threshold)
        : (_ <: loss,_),pds
        // Calculate gradients
        : _,gradients
        // Scale gradients by the learning rate
        : _,par(n,vars.N,_,learningRate : *)
    with {
        window = ba.slidingMean(windowSize);
        // "Loss" is just the difference between "truth" and learnable output
        loss = _ <: attach(hbargraph("[100]loss",-5,5));
        // A way to move the partial derivatives around.
        pds = si.bus(vars.N);
        // Calculate gradients; for L2 norm: 2 * dy/dx_i * (learnable - groundtruth)
        gradients = _,par(n,vars.N, _,2 : *)
            : routeall
            : par(n,vars.N, * <: attach(hbargraph("[101]gradient %n",-1000,1000)));

        // A utility to duplicate the first input for combination with all remaining inputs.
        routeall = _,si.bus(vars.N)
            : route(vars.N+1,vars.N*2,par(n,vars.N,(1,2*n+1),(n+2,2*(n+1))));
    };
};
