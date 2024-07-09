import("stdfaust.lib");
df = library("diff.lib");

// Multiple training signals...
NTRAIN = 10;

rec(F~G,ngrads) = (G,si.bus(n):F)~si.bus(m)
with {
    n = inputs(F)-outputs(G);
    m = inputs(G)-ngrads;
};

process =
    par(n,NTRAIN,no.noises(NTRAIN,n)),in
    : hgroup("DDSP",
        route(nIN,nOUT,
            // Route gradients to learnable algo.
            par(n,vars.N,(n+1,n+2+NTRAIN)),
            // Route training inputs to ground truth algo.
            par(n,NTRAIN,(n+vars.N+1,n+1)),
            // Route input to learnable algo.
            (vars.N+NTRAIN+1,NTRAIN+1)
        ) with {
            nIN = vars.N+NTRAIN+1;
            nOUT = nIN;
        }
        : vgroup("Parameters",vgroup("Hidden", truth),vgroup("Learned", learnable))
        // : si.bus(NTRAIN),(_ <: _,_),si.bus(vars.N)
        : route(nIN,nOUT,
            // Route first ground truth output to output
            (1,nOUT-1),
            // Route ground truth outputs to df.learn
            par(n,NTRAIN,(n+1,n+1)),
            // Route learnable output to df.learn
            (NTRAIN+1,NTRAIN+1),
            // Route learnable output to output
            (NTRAIN+1,nOUT),
            // Route gradients to df.learn
            par(n,vars.N,(n+NTRAIN+2,n+NTRAIN+2))
        ) with {
            nIN = NTRAIN+vars.N+1;
            nOUT = nIN+2;
        }
        : vgroup("[0]Loss & gradients", d.learnN(1<<0,d.optimizeSGD(5e-3),NTRAIN)),_,_
    ) ~ (!,si.bus(vars.N))
    // : _,si.block(NVARS),_,_
with {
    in = no.noise;

    hiddenFB = hslider("[1]Feedback", .5, 0, .999, .001);
    hiddenGain = hslider("[2]Gain", .5, 0, 2, .001);
    truth = par(n,NTRAIN,+ ~ (hiddenFB,_ : *) : _,hiddenGain : *);

    vars = df.vars((a,b))
    with {
        a = -~_ : _,.99 : min <: attach(hbargraph("[50]Feedback",0,1));
        b = -~_ <: attach(hbargraph("[51]Gain",0,2));
    };

    d = df.env(vars);

    learnable = d.input,par(n,vars.N,grad)
        : route(1+2*vars.N,1+2*vars.N,(1,2),par(n,vars.N,(n+2,n+3)),(vars.N+2,1),(1+2*vars.N,1+2*vars.N))
        : rec(f~g,1),grad : h
    with {
        grad = _;

        f = d.diff(+);
        g = d.diff(_),vars.var(1) : d.diff(*);
        h = d.diff(_),vars.var(2) : d.diff(*);
    };
};
