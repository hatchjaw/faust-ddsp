import("stdfaust.lib");
df = library("diff.lib");

NVARS = 2;

rec(F~G,ngrads) = (G,si.bus(n):F)~si.bus(m)
with {
    n = inputs(F)-outputs(G);
    m = inputs(G)-ngrads;
};

process =
    in,in
    : hgroup("DDSP",
        route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : vgroup("Parameters",vgroup("Hidden", truth),vgroup("Learned", learnable))
        : (_ <: _,_),(_ <: _,_),si.bus(NVARS)
        : route(4+NVARS,4+NVARS,(1,1),(2,NVARS+3),(3,2),(4,NVARS+4),par(n,NVARS,(n+5,n+3)))
        : vgroup("[0]Loss & gradients", df.learn(1<<4,5e-3,NVARS)),_,_
    ) ~ (!,si.bus(NVARS))
    : _,si.block(NVARS),_,_
with {
    in = no.noise;

    hiddenFB = hslider("[1]alpha", .5, 0, .999, .001);
    hiddenGain = hslider("[2]gain", .5, 0, 2, .001);
    truth = + ~ (hiddenFB,_ : *) : _,hiddenGain : *;

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
