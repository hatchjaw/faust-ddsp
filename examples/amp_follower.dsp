import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,df.learnMAE(1<<5,1e-3))
        with {
            rel = hslider("Release Time",0,0,1,0.01);
            truth = _, abs : env
            with {
                p = ba.an.tau2pole(rel);
                env(x) = (x,(1 - p) : *) : (+ : max(x,_)) ~ *(p);
            }

            vars = df.vars((rel))
            with {
                rel = -~_, ma.EPSILON : + <: attach(hbargraph("[0] Release Time",0,1))
            };

            d = df.env(vars);
            learnable = d.input,vars.var(1) :
            with {
                // TODO
            }
        };