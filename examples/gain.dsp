no = library("noises.lib");
df = library("diff.lib");

process = no.noise <: _,_
    : hgroup("Differentiable gain", df.backprop(truth,learnable,d.learnMAE(1<<5,1e-3)))
with {
    truth = _,hslider("gain",.5,0,1,.01) : *;

    vars = df.vars((gain))
    with {
        gain = -~_ <: attach(hbargraph("gain",0,1));
    };

    d = df.env(vars);

    learnable = d.input,vars.var(1) : d.diff(*);
};
