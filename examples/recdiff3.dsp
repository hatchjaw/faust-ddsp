import("stdfaust.lib");
df = library("diff.lib");

NVARS = 2;
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
            par(n,NVARS,(n+1,n+2+NTRAIN)),
            // Route training inputs to ground truth algo.
            par(n,NTRAIN,(n+NVARS+1,n+1)),
            // Route input to learnable algo.
            (NVARS+NTRAIN+1,NTRAIN+1)
        ) with {
            nIN = NVARS+NTRAIN+1; 
            nOUT = nIN;
        }
        : vgroup("Parameters",vgroup("Hidden", truth),vgroup("Learned", learnable))
        // : si.bus(NTRAIN),(_ <: _,_),si.bus(NVARS)
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
            par(n,NVARS,(n+NTRAIN+2,n+NTRAIN+2))
        ) with {
            nIN = NTRAIN+NVARS+1; 
            nOUT = nIN+2;
        }
        : vgroup("[0]Loss & gradients", df.learnN(1<<0,5e-3,NTRAIN,NVARS)),_,_
    ) ~ (!,si.bus(NVARS))
    // : _,si.block(NVARS),_,_
with {
    in = no.noise;

    hiddenFB = hslider("[1]Feedback", .5, 0, .999, .001);
    hiddenGain = hslider("[2]Gain", .5, 0, 2, .001);
    truth = par(n,NTRAIN,+ ~ (hiddenFB,_ : *) : _,hiddenGain : *);

    learnable = df.input(NVARS),par(n,NVARS,grad)
        // Routing here doesn't generalise; maybe wrap g in an environment that contains the number of expected gradients?
        : route(1+2*NVARS,1+2*NVARS,(1,2),par(n,NVARS,(n+2,n+3)),(NVARS+2,1),(1+2*NVARS,1+2*NVARS))
        : rec(f~g,1),grad : h
    with {
        a = -~_ : _,.99 : min <: attach(hbargraph("[50]Feedback",0,1));
        b = -~_ <: attach(hbargraph("[51]Gain",0,2));
        grad = _;

        f = df.diff(+,NVARS);
        g = df.diff(_,NVARS),df.var(1,a,NVARS) : df.diff(*,NVARS);
        h = df.diff(_,NVARS),df.var(2,b,NVARS) : df.diff(*,NVARS);
    };
};
