import("stdfaust.lib");
df = library("diff.lib");

NVARS = 1;

rec(F~G) = (G,si.bus(n):F)~si.bus(m)
with {
    n = inputs(F)-outputs(G);
    m = inputs(G)-NVARS;
};

process = in,in
    : (
        route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : gt,learnable
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn, and to output.
            (1,1),(1,nOUT-1),
            // Route learnable output to df.learn, and to output.
            (2,2),(2,nOUT),
            par(n,NVARS,(n+3,n+3))
        ) with {
            nIN = 2+NVARS;
            nOUT = nIN+2;
        }
        : df.learn(1<<4,5e-4,NVARS),_,_
    ) ~ (!,si.bus(NVARS))
    : _,si.block(NVARS),_,_
with {
    in = no.noise;

    hiddenA = hslider("Feedback", .5, 0, .999, .001);
    gt = + ~ (hiddenA,_ : *);

    learnable = df.input(NVARS),si.bus(NVARS)
        : route(2+NVARS,2+NVARS,(1,NVARS+1),par(n,NVARS,(n+2,n+NVARS+2),(n+NVARS+2,n+1)))
        : rec(f ~ g)
    with {
        a = -~_ : _,.99 : min <: attach(hbargraph("[10]Learned feedback",0,1));
        f = df.diff(+,NVARS);
        g = df.diff(_,NVARS),df.var(1,a,NVARS) : df.diff(*,NVARS);
    };
};
