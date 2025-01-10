import("stdfaust.lib");
df = library("diff.lib");

NINPUTS = 2;
LAYERSPEC = ((1,1,2,2,1,2));

// TODO: abstract this stuff away.
Wb = par(i, outputs(LAYERSPEC) / 2,
    par(j, ba.take(2 * i + 1, LAYERSPEC),
        par(k, ba.take(2 * i + 2, LAYERSPEC),
            df.var(hbargraph("w%i%j%k", -5, 5), 0)
        ),df.var(hbargraph("b%i%j", -5, 5), 0)
    )
);

v = df.vars(Wb);
d = df.env(v);

// TODO: gradient averaging
process = no.noise <: nn
with {
    nn(y) = hgroup("",
        vgroup("Weights", d.nn(LAYERSPEC, d.activations.sigmoid)) :> vgroup("Loss", d.losses.L2(1<<0, 1e-2, y))
    ) ~ (!,si.bus(v.N)) : (y,_,si.block(v.N));
};
