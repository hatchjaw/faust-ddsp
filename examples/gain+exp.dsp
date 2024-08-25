import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,d.learnMAE(1<<3,d.optimizers.SGD(1e-3)))
        with {
            g = hslider("[0] gain", 0.5, 0, 1, 0.001);
            truth = _, (((g,2 : ^), -1 : *) : exp) : *;

            vars = df.vars((gain))
            with {
                gain = -~_,ma.EPSILON : + <: attach(hbargraph("[0] learned gain", 0, 1));
            };

            d = df.env(vars);
            learnable = d.input,vars.var(1),d.diff(2),d.diff(-1)
                        : d.diff(_), d.diff(^), d.diff(_)
                        : d.diff(_), d.diff(*)
                        : d.diff(_), d.diff(exp)
                        : d.diff(*);
        };