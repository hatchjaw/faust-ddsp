import("stdfaust.lib");
df = library("diff.lib");

process = in <: _,_
    : hgroup("Differentiable lowpass",
        df.backprop(groundTruth,learnable,d.learnMAE(1<<0,d.optimizer.SGD(5e-1)))
    )
with {
    vars = df.vars((cutoff))
    with {
        cutoff = -~_
            // Map [-1,1] to [50,20000]
            <: attach(hbargraph("[1]Normalised cutoff",-1.,1.))
            : (_,1.,mid : +,_ : (*,mini : +))
            with {
                maxi = 20000.;
                mini = 50.;
                mid = maxi,mini,2 : +,_ : /;
            }
            <: attach(hbargraph("[1]Cutoff [scale:log]",50.,20000.));
    };

    d = df.env(vars);

    in = no.noise;

    groundTruth = lpf(hiddenCutoff)
    with {
        hiddenCutoff = hslider("[0]cutoff [scale:log]", 440., 50., 20000., .1);

        lpf(fc) = _
            <: *(b0), (mem : *(b1))
            // :> + ~ *(0-a1)
            : + : + ~ (_,(0,a1 : -) : *)
        with {
            w = 2*ma.PI*fc;
            c = 1/tan(w*0.5/ma.SR);
            d = 1+c;
            b0 = 1/d;
            b1 = 1/d;
            a1 = (1-c)/d;
        };
    };

    learnable = learnableLPF
    with {
        dd = d.diff;

        learnableLPF = d.input,(si.bus(vars.N) <: si.bus(vars.N*2))
            : (dd(_),si.bus(vars.N) <: (dd(_),b0 : dd(*)),(dd(mem),b1 : dd(*))),_
            : dd(+),_
            : route(3,4,(1,3),(2,4),(3,1),(3,2))
            : df.rec(f~g,2)
            with {
                f = dd(+);
                g = dd(_),(dd(0),a1 : dd(-)) : dd(*);
            }
        with {
            w = dd(2*ma.PI),vars.var(1) : dd(*);
            c = dd(1),dd(.5/ma.SR),w
                : dd(_),dd(*)
                : dd(_),dd(tan)
                : dd(/);
            d = dd(1),c : dd(+);
            b0 = dd(1),d : dd(/);
            b1 = dd(1),d : dd(/);
            a1 = dd(1),c,d : dd(-),dd(_) : dd(/);
        };
    };
};
