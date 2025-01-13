import("stdfaust.lib");
df = library("diff.lib");

N = 3;

H = ((sub,si.bus(N-1))
    ~ (bank(N-1))),si.bus(N)
    : _,si.block(N-1),si.bus(N)
    : bank(N)
with {
    sub = _,_ <: !,_,_,! : -;
    bank(n) = (_ <: si.bus(n)),si.bus(n) : df.routes.interleave(n, 2) : sum(i, n, @(i),_ : *);
};

vars = df.vars(coeffs)
with {
    coeffs = par(i, N-1, df.var(hbargraph("a%j", -1, 1), 0) with {j = i+1;}),par(i, N, df.var(hbargraph("b%i", -1, 1), 0));
};

d = df.env(vars);

iir = _,coeffs : H
with {
    coeffs = par(i, N-1, hslider("a%j", 0, -1, 1, .01) with {j = i+1;}),par(i, N, hslider("b%i", i==0, -1, 1, .01));
};

diir = _,coeffs : d.fwdADInN(H, 1)
with {
    coeffs = par(i, 2*N-1, vars.var(i+1));
};

process = no.noise <: hgroup("Differentiable IIR", df.paramOpt(iir, diir, d.losses.L2(1<<0, alpha)))
with {
    alpha = hslider("alpha [scale:log]", 1e-3, 1e-5, 1e-1, 1e-8);
};
