// Work in progress

import("stdfaust.lib");
df = library("diff.lib");

NVARS = 1;
hiddenCutoff = hslider("[0]cutoff [scale:log]", 440., 50., 20000., .1);

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
process = in,in
    : (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : vgroup("Hidden", groundTruth),vgroup("Learned", learnable)
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("Loss/gradient", learn(1<<0,1e-1,NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(NVARS),_,_
with {
    in = no.noise;

    groundTruth = lpf(hiddenCutoff);

    // learnable = df.input(NVARS),df.var(1,cutoff,NVARS) : df.diff(+,NVARS)
    learnable = learnableLPF
    with {
        // learnableLPF = df.input(NVARS)...
        learnableLPF = df.input(NVARS),(si.bus(NVARS) <: si.bus(NVARS*2))
            : (df.diff(_,NVARS),si.bus(NVARS) <: (df.diff(_,NVARS),b0 : df.diff(*,NVARS)),(df.diff(mem,NVARS),b1 : df.diff(*,NVARS))),_
            : df.diff(+,NVARS),_
            : route(3,4,(1,3),(2,4),(3,1),(3,2))
            : df.rec(f~g,2)
            with {
                f = df.diff(+,NVARS);
                g = df.diff(_,NVARS),(df.const(0,NVARS),a1 : df.diff(-,NVARS)) : df.diff(*,NVARS);
            }
        with {
            cutoff = -~_
                // Map [-1,1] to [50,20000]
                <: attach(hbargraph("[1]Normalised cutoff",-1.,1.))
                : (_,1.,mid : +,_ : (*,mini : +))
                with {
                    maxi = 20000.;
                    mini = 50.;
                    mid = maxi,mini,2 : +,_ : /;
                }
                <: attach(hbargraph("[1]Cutoff [scale:log]",50.,20000.));
            w = df.const(2*ma.PI,NVARS),df.var(1,cutoff,NVARS) : df.diff(*,NVARS);
            c = df.const(1,NVARS),df.const(.5/ma.SR,NVARS),w
                : df.diff(_,NVARS),df.diff(*,NVARS)
                : df.diff(_,NVARS),df.diff(tan,NVARS)
                : df.diff(/,NVARS);
            d = df.const(1,NVARS),c : df.diff(+,NVARS);
            b0 = df.const(1,NVARS),d : df.diff(/,NVARS);
            b1 = df.const(1,NVARS),d : df.diff(/,NVARS);
            a1 = df.const(1,NVARS),c,d : df.diff(-,NVARS),df.diff(_,NVARS) : df.diff(/,NVARS);
        };
    };

    learn(windowSize, learningRate, nvars) =
        // Window the input signals
        par(i,2+nvars,window)
        // Calculate the difference between the ground truth and learnable outputs
        : (ro.cross(2) : - ),pds
        // Calculate loss (this is just for show, since there's no sensitivity threshold)
        : (_ <: loss,_),pds
        // Calculate gradients
        : _,gradients
        // Scale gradients by the learning rate
        : _,par(n,nvars,_,learningRate : *)
    with {
        window = ba.slidingMean(windowSize);
        // Loss function (L1 norm)
        loss = abs <: attach(hbargraph("[100]loss",0.,.05));
        // A way to move the partial derivatives around.
        pds = si.bus(nvars);
        // Calculate gradients; for L1 norm: dy/dx_i * (learnable - groundtruth) / abs(learnable - groundtruth)
        gradients = route(nvars+1,2*nvars+1,(1,1),par(n,nvars,(1,n*2+3),(n+2,2*n+2)))
            : (abs,1e-10 : max),par(n,nvars, *)
            : route(nvars+1,nvars*2,par(n,nvars,(1,2*n+2),(n+2,2*n+1)))
            : par(n,nvars, / <: attach(hbargraph("[101]gradient %n",-.5,.5)));
    };
};
