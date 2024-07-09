import("stdfaust.lib");
df = library("diff.lib");

process = in <: _,_ : df.backprop(truth,learnable,d.learnMSE(1<<3,d.optimizeSGD(1e-2)))
with {
    in = no.noise;

    hiddenFB = hslider("[1]Feedback", .5, 0, .999, .001);
    hiddenGain = hslider("[2]Gain", .5, 0, 2, .001);
    truth = + ~ (hiddenFB,_ : *) : _,hiddenGain : *;

    vars = df.vars((a,b))
    with {
        a = -~_ : _,.99 : min <: attach(hbargraph("[50]Feedback",0,1));
        b = -~_ <: attach(hbargraph("[51]Gain",0,2));
    };

    d = df.env(vars);

    learnable = d.input,par(n,vars.N,grad)
        : route(1+2*vars.N,1+2*vars.N,(1,2),par(n,vars.N,(n+2,n+3)),(vars.N+2,1),(1+2*vars.N,1+2*vars.N))
        : df.rec(f~g,1),grad : h
    with {
        grad = _;

        f = d.diff(+);
        g = d.diff(_),vars.var(1) : d.diff(*);
        h = d.diff(_),vars.var(2) : d.diff(*);
    };
};
