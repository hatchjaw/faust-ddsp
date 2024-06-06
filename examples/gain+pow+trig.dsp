import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,d.learnL1(1<<3,1e-3))
        with {
            g = hslider("[0] gain", 0.5, 0, 1, 0.001);
            truth = _, (g,cos(g) : ^) : *;

            vars = df.vars((gain))
            with {
                gain = -~_ <: attach(hbargraph("[0] learned gain", 0, 1));
            };

            d = df.env(vars);
            learnable = d.input,vars.var(1)
                        : d.diff(_),
                        route(1+vars.N,2*(1+vars.N),
                        par(i,1+vars.N,(i+1,i+1),(i+1,i+vars.N+1)))
                        : d.diff(_),d.diff(^)
                        : d.diff(*);
        };