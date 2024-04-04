import("stdfaust.lib");
df = library("diff.lib");

NTAPS = 8;

process = in <: _,_
    : hgroup("Differentiable FIR",df.backprop(truth,learnable,d.learnL2(1<<3,1e-1)))
    // : !,_,_
with {
    in = no.noise;

    truth = _ <: sum(n, NTAPS, _,n : @,bn : *
    with {
        bn = hslider("b%n", sin(n), -1., 1., .001);
    });

    vars = df.vars(par(n,NTAPS,b(n)))
    with {
        b(n) = -~_ <: attach(hbargraph("b%n",-1,1));
    };

    d = df.env(vars);

    learnable = route(1+NTAPS,NTAPS*2,par(n,NTAPS,(1,2*n+1),(n+2,2*n+2))) :
        par(n, NTAPS,
            d.input,d.diff(n),_
            : d.diff(@),vars.var(n+1)
            : d.diff(*)
        ) : d.sumall(NTAPS);
};
