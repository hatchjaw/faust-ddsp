import("stdfaust.lib");
df = library("diff.lib");

NTAPS = 8;

o = _,1,_',2,_'',3,_''',4 : par(i,4,*) : par(i,2,+),par(i,0,_) : +;

ppp = no.noise <: sum(n, NTAPS, _,n : @,bn : *
    with {
        bn = hslider("b%n", 0., -1., 1., .001);
    });

pppp = no.noise
    <: par(n, NTAPS, df.input(NTAPS),df.const(n,NTAPS)
    : df.diff(@,NTAPS),df.var(n+1,bn,NTAPS)
    : df.diff(*,NTAPS)
    with {
        bn = 1-n/3;
    })
    : par(n,NTAPS/2,df.diff(+,NTAPS)),par(n,NTAPS%2,df.diff(_,NTAPS)) : df.diff(+,NTAPS);

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
    ) : _,si.block(NTAPS),_,_
with{
    in = no.noise;

    truth = _ <: sum(n, NTAPS, _,n : @,bn : *
        with {
            bn = hslider("b%n", sin(n), -1., 1., .001);
        });

    learnable = route(1+NTAPS,NTAPS*2,par(n,NTAPS,(1,2*n+1),(n+2,2*n+2))) :
        par(n, NTAPS,
            df.input(NTAPS),df.const(n,NTAPS),_
            : df.diff(@,NTAPS),df.var(n+1,bn,NTAPS)
            : df.diff(*,NTAPS)
            with {
                bn = -~_ <: attach(hbargraph("b%n",-1,1));
            })
            // Recreate a sum iteration:
            : seq(n,NTAPS-2,df.diff(+,NTAPS),par(m,NTAPS-n-2,df.diff(_,NTAPS))) : df.diff(+,NTAPS);
};

process = p;

// process = si.bus(4) : +,si.bus(2) : +,si.bus(1) : +;
// process = seq(n,NTAPS-2,+,si.bus(NTAPS-n-2)) : +;
// process = seq(n,NTAPS-2,df.diff(+,NTAPS),par(m,NTAPS-n-2,df.diff(_,NTAPS))) : df.diff(+,NTAPS);
