import("stdfaust.lib");
df = library("diff.lib");

process = no.noise <: _,_
        : df.backprop(truth,learnable,d.learnMAE(1<<3,d.optimizers.SGD(1e-3)))
        with {
            p = button("Trigger");
            truth = - ~ _ * p;

            vars = df.vars((period))
            with {
                period = -~_ <: attach(hbargraph("[0] Trigger",0,1));
            };

            d = df.env(vars);
            learnable = diffVars
            with {
                diffVars = df.rec(f~g, 0);
                f = d.diff(-);
                g = d.input,vars.var(1) : d.diff(*);
            };
        };