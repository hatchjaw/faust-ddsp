import("stdfaust.lib");
df = library("diff.lib");

process = in <: _,_ : df.backprop(gt,learnable,d.learnMSE(1<<4,5e-4))
with {
    in = no.noise;

    hiddenA = hslider("Feedback", .5, 0, .999, .001);
    gt = + ~ (hiddenA,_ : *);

    vars = df.vars((a))
    with {
        a = -~_ : _,.99 : min <: attach(hbargraph("[10]Learned feedback",0,1));
    };
    d = df.env(vars);

    learnable = d.input,si.bus(vars.N)
        : route(2+vars.N,2+vars.N,(1,vars.N+1),par(n,vars.N,(n+2,n+vars.N+2),(n+vars.N+2,n+1)))
        : df.rec(f~g,1)
    with {
        f = d.diff(+);
        g = d.diff(_),vars.var(1) : d.diff(*);
    };
};
