import("stdfaust.lib");
df = library("diff.lib");

NVARS = 1;
MAXDELAY = 100;

p = f,g,h : df.diff(@,NVARS),df.diff(_,NVARS) : df.diff(*,NVARS)
with {
    delay = (-~_),0,1<<24 : max,_ : min;
    gain = -~_;

    f = df.input(NVARS);
    g = df.var(1,delay,NVARS);
    h = df.var(2,gain,NVARS);
};

process = in,in
    : hgroup("Differentiable lowpass",
        (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : vgroup("[0]Parameters", vgroup("Hidden", truth),vgroup("Learned", learnable))
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.learnL2(1<<0,1e-3,NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    ) : _,si.block(NVARS),_,_
with {
    in = no.noise;

    hiddenDelay = hslider("Delay", 10, 0, MAXDELAY, 1);
    hiddenGain = hslider("Gain", .5, 0., 2., .01);
    truth =
        // _,hiddenDelay,hiddenGain : @,_ : *;
        _,hiddenDelay : @;

    learnable = df.input(NVARS),df.var(1,delay,NVARS) : df.diff(@,NVARS)
    // learnable = df.input(NVARS),df.var(1,delay,NVARS),df.var(2,gain,NVARS) : df.diff(@,NVARS),df.diff(_,NVARS) : df.diff(*,NVARS)
    with {
        delay = (-~_),0,1<<24 : max,_ : min <: attach(hbargraph("Delay", 0, MAXDELAY));
        gain = -~_ <: attach(hbargraph("Gain",0,2));
    };
};
