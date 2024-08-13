no = library("noises.lib");
df = library("diff.lib");
import("stdfaust.lib");

process = hgroup("Differentiable gain", df.backprop(truth,learnable,d.learnLinearFreq(1<<5,d.optimizeRMSProp(1e-1, 0.9))))
with {
    truth = os.osc(in);
    in = hslider("Frequency", 500, 0, 20000, 0.1);
    vars = df.vars((freq))
    with {
        freq = -~_ <: attach(hbargraph("Frequency learned",0,20000));
    };

    d = df.env(vars);

    learnable = d.osc(vars.var(1));
};
