import("stdfaust.lib");
df = library("diff.lib");

NTAPS = 10;

p = in,in
    : hgroup("Differentiable FIR",
        (route(NTAPS+2,NTAPS+2,par(n,NTAPS,(n+1,n+3)),(NTAPS+1,1),(NTAPS+2,2))
        : vgroup("Hidden", truth),vgroup("Learned", learnable)
        : route(2+NTAPS,4+NTAPS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NTAPS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NTAPS+4),
            // Route gradients to df.learn.
            par(n,NTAPS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.learnL2(1<<3,1e-1,NTAPS),_,_)) ~ (!,si.bus(NTAPS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    ) : _,si.block(NTAPS),!,!
with{
    in = no.noise;

    // Impulse response version...
    hiddenDelay1 = hslider("Delay1", (2,(NTAPS-1) : min), 0, NTAPS-1, 1);
    hiddenDelay2 = hslider("Delay2", (8,(NTAPS-1) : min), 0, NTAPS-1, 1);

    truth = _ <: _,hiddenDelay1,_,hiddenDelay2 : @,@ : +;

    learnable = route(1+NTAPS,NTAPS*2,par(n,NTAPS,(1,2*n+1),(n+2,2*n+2))) :
        par(n, NTAPS,
            df.input(NTAPS),_ : delay,coeff : multiply
            with {
                bn = -~_ <: attach(hbargraph("b%n",-1,1));

                delay = df.diff(_,NTAPS),df.const(n,NTAPS) : df.diff(@,NTAPS);
                coeff = df.var(n+1,bn,NTAPS);
                multiply = df.diff(*,NTAPS);
            })
            : sumAll
            with {
                sumAll = seq(n,NTAPS-2,df.diff(+,NTAPS),par(m,NTAPS-n-2,df.diff(_,NTAPS))) : df.diff(+,NTAPS);
            };
};

process = p;
