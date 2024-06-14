import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,df.learnL1(1<<5,1e-3))
        with {
            p = hslider("Period",0,0,1,0.01);
            truth = _ <: _,_ : * : fi.avg_tau(p);

            vars = df.vars((period))
            with {
                period = -~_ <: attach(hbargraph("[0] Period",0,1));
            };

            d = df.env(vars);
            learnable = diffVars, vars.var(1) : fi.avg_tau
            with {
                diffVars = d.input <: d.diff(_),d.diff(_)
                : d.diff(*);
            };
        };