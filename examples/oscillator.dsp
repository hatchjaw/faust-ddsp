import("stdfaust.lib");
df = library("diff.lib");

process = hgroup("Differentiable oscillator",
        vgroup("[0]Parameters", vgroup("Hidden", truth),vgroup("Learned", learnable))
        : route(2+vars.N,4+vars.N,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,vars.N+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,vars.N+4),
            // Route gradients to df.learn.
            par(n,vars.N,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.env(vars).learnMAE(1<<0,d.optimizer.SGD(5e-6)),_,_)) ~ (!,si.bus(vars.N))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(vars.N),!,_
with {
    hiddenF0 = hslider("freq [scale:log]", 440.,50.,1000.,.01);
    // truth = os.osc(hiddenF0);

    truth = phasor(hiddenF0) : sin
    with {
        decimalPart(x) = x-int(x);
        phasor(f) = f/ma.SR : (+ : decimalPart) ~ _ : *(2 * ma.PI);
    };

    vars = df.vars((f0))
    with {
        f0 = -~_
            // Map [-1,1] to range
            <: attach(hbargraph("[1]Normalised freq",-1.,1.))
            : (_,1.,mid : +,_ : (*,mini : +))
            <: attach(hbargraph("[1]Frequency [scale:log]", mini, maxi))
        with {
            maxi = 1000.;
            mini = 50.;
            mid = maxi,mini,2 : +,_ : /;
        };
    };

    learnable = df.env(vars).osc(vars.var(1));
};
