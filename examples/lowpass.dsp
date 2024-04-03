import("stdfaust.lib");
df = library("diff.lib");

diffVars(vars) = environment {
    N = outputs(vars);
    var(i) = ba.take(i,vars),pds(N,i)
    with {
        pds(N,i) = par(j,N,i-1==j);
    };
};

process = in,in
    : hgroup("Differentiable lowpass",
        df.backprop(groundTruth,learnable,df.learnL1(1<<0,5e-1,vars.N))
    )
with {
    vars = diffVars((cutoff))
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
        learnableLPF = df.input(vars.N),(si.bus(vars.N) <: si.bus(vars.N*2))
            : (df.diff(_,vars.N),si.bus(vars.N) <: (df.diff(_,vars.N),b0 : df.diff(*,vars.N)),(df.diff(mem,vars.N),b1 : df.diff(*,vars.N))),_
            : df.diff(+,vars.N),_
            : route(3,4,(1,3),(2,4),(3,1),(3,2))
            : df.rec(f~g,2)
            with {
                f = df.diff(+,vars.N);
                g = df.diff(_,vars.N),(df.diff(0,vars.N),a1 : df.diff(-,vars.N)) : df.diff(*,vars.N);
            }
        with {
            w = df.diff(2*ma.PI,vars.N),vars.var(1) : df.diff(*,vars.N);
            c = df.diff(1,vars.N),df.diff(.5/ma.SR,vars.N),w
                : df.diff(_,vars.N),df.diff(*,vars.N)
                : df.diff(_,vars.N),df.diff(tan,vars.N)
                : df.diff(/,vars.N);
            d = df.diff(1,vars.N),c : df.diff(+,vars.N);
            b0 = df.diff(1,vars.N),d : df.diff(/,vars.N);
            b1 = df.diff(1,vars.N),d : df.diff(/,vars.N);
            a1 = df.diff(1,vars.N),c,d : df.diff(-,vars.N),df.diff(_,vars.N) : df.diff(/,vars.N);
        };
    };
};
