import("stdfaust.lib");
df = library("diff.lib");

process = in <: _,_
    : hgroup("Differentiable Gain + DC", df.backprop(target, estimate, d.losses.L1(1<<0, alpha)))
    : route(3,4,(3,1),(3,2),(1,3),(2,4)) : par(i, 4, *(out))
with {
    out = hslider("Volume", 0, 0, 1, .01);
    in = no.noise;

    alpha = hslider("alpha [scale:log]",1e-3,1e-5,1e-1,1e-8);

    H = *,_ : +;

    target = _,gain,dc : H
    with {
        gain = hslider("[0]Gain", .5, 0, 2, .01);
        dc = hslider("[1]DC", .5, -1, 1, .01);
    };

    vars = df.vars((gain,dc))
    with {
        gain = df.var(hbargraph("[5]Gain", 0, 2), ma.EPSILON);
        dc = df.var(hbargraph("[6]DC", -1, 1), ma.EPSILON);
    };

    d = df.env(vars);

    estimate = _,vars.var(1),vars.var(2) : d.fwdADInN(H, 1);
};
