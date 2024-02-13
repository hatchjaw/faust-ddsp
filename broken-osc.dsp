import("stdfaust.lib");
df = library("diff.lib");

// This is completely broken, but sounds amazing

NVARS = 1;

p = df.osc(f0,NVARS)
with {
    f0 = -~_
        // Map [-1,1] to range
        <: attach(hbargraph("[1]Normalised freq",-1.,1.))
        : (_,1.,mid : +,_ : (*,mini : +))
        <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
    with {
        maxi = 10000.;
        mini = 50.;
        mid = maxi,mini,2 : +,_ : /;
    };
};

process = hgroup("Differentiable oscillator",
        vgroup("[0]Parameters", vgroup("Hidden", truth),vgroup("Learned", learnable))
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.learn(1<<0,hslider("alpha [scale:log]",1e-4,1e-6,1e-1,1e-8),NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(NVARS),!,! : _,1,.5 : -,_ : *
with {
    truth = os.osc(hslider("freq [scale:log]", 1000.,50.,10000.,.01));

    learnable = osc(df.var(1,f0,NVARS),NVARS)
    with {
        f0 = -~_
            // Map [-1,1] to range
            <: attach(hbargraph("[1]Normalised freq",-1.,1.))
            : (_,1.,mid : +,_ : (*,mini : +))
            <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
        with {
            maxi = 10000.;
            mini = 50.;
            mid = maxi,mini,2 : +,_ : /;
        };
    };

    phasor(f0,nvars) = f0,df.const(ma.SR,nvars)
        : df.diff(/,nvars)
        : df.rec(f~g,0),df.const(2*ma.PI,nvars)
        : df.diff(*,nvars)
        with {
            f = df.diff(+,nvars) <: df.diff(_,nvars),df.diff(int,nvars) : df.diff(-,nvars);
            g = df.diff(_,nvars);
        };

    osc(f0,nvars) = phasor(f0,nvars) : df.diff(sin,nvars);
};
