import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,df.learnL1(1<<5,1e-3))
        with {
            m = hslider("Mode",0,0,1,1);
            truth = (*(m),_ : max) ~ _;

            vars = df.vars((mode))
            with {
                mode = -~_ <: attach(hbargraph("Mode",0,1));
            };

            d = df.env(vars);
            learnable = vars.var(1), d.input : df.rec(f~g, 1)
            with {
                f = d.diff(*), d.diff(_) : d.diff(max);
                g = d.diff(_);
            };
        };