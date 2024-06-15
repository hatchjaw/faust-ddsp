import("stdfaust.lib");
fi = library("filters.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,d.learnL2(1<<3,1e-3))
        with {
            p = hslider("Period",0,0,1,0.01);
            truth = _ <: * : fi.avg_tau(p);

            vars = df.vars((period))
            with {
                period = -~_ <: attach(hbargraph("[0] Period",0,1));
            };

            d = df.env(vars);
            learnable = diffVars, vars.var(1) : avgTau
            with {
                diffVars = d.input <: d.diff(*);
                avgTau = route(4, 4, (1,1), (3,2), (2,3), (4,4)) : lptn, _, _
                with {
                    lptn = _ : si.smooth(ba.tau2pole(_ / log(10.0^(float(N)/20.0))));
                    N = 8.6858896381;
                };
            };
        };