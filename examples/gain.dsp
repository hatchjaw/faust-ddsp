no = library("noises.lib");
df = library("diff.lib");

process = no.noise <: _,_
    : hgroup("Differentiable gain", df.backprop(truth,learnable,df.learnL1(1<<5,1e-3,NVARS)))
with {
    truth = _,hslider("gain",.5,0,1,.01) : *;

    NVARS = 1;
    learnable = df.input(NVARS),df.var(1,gain,NVARS) : df.diff(*,NVARS)
    with {
        gain = -~_ <: attach(hbargraph("gain",0,1));
    };
};
