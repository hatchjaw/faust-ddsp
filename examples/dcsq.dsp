no = library("noises.lib");
df = library("diff.lib");

process = no.noise <: _,_
    : hgroup("Differentiable gain", df.backprop(truth,learnable,d.learnMSE(1<<0,d.optimizers.SGD(1e-3))))
with {
    truth = _,hslider("dc",.75,-1,1,.01),2 : _,^ : +;

    vars = df.vars((dc)) with {
        dc = -~_ : _,1e-10 : max <: attach(hbargraph("dc",-1,1));
        // dc = -~_ <: attach(hbargraph("dc",-1,1));
    };
    d = df.env(vars);

    learnable = d.input,vars.var(1),d.diff(2)
        // : par(i,6, _ <: attach(hbargraph("0init%i",-2,2)))
        : d.diff(_),d.diff(^)
        // : par(i,4, _ <: attach(hbargraph("1postpow%i",-2,2)))
        : d.diff(+);
        // : par(i,2, _ <: attach(hbargraph("2postadd%i",-2,2)));
};
