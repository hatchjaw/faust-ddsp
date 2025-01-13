import("stdfaust.lib");
df = library("diff.lib");

N = 4;

H = (_ <: si.bus(N)),si.bus(N) : df.routes.interleave(N, 2) : sum(i, N, @(i),_ : *);

vars = df.vars(coeffs)
with {
    coeffs = par(i, N, df.var(hbargraph("b%i", -1, 1)));
};

d = df.env(vars);

fir = _,coeffs : H
with {
    coeffs = par(i, N, hslider("b%i", 0, -1, 1, .01));
};

dfir = _,coeffs : d.fwdADInN(H, 1)
with {
    coeffs = par(i, N, vars.var(i+1));
};

process = no.noise <: hgroup("Differentiable FIR", df.paramOpt(fir, dfir, d.losses.L2(1<<0, alpha)))
with {
    alpha = hslider("alpha [scale:log]",1e-3,1e-5,1e-1,1e-8);
};
