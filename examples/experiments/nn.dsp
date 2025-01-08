import("stdfaust.lib");
df = library("diff.lib");

NINPUTS = 2;
LAYERSPEC = ((2,3,1,2,3,2));

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

process = d.nn(LAYERSPEC, d.activations.sigmoid) :> _;
