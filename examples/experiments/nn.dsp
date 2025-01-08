import("stdfaust.lib");
df = library("diff.lib");

NINPUTS = 2;
LAYERSPEC = ((2, 2, 1, 2));//, 4, 1));

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

processE = d.layers.inputLayer(LAYERSPEC)
    : d.layers.dense(2, 2, d.activations.sigmoid);
    //<: d.layers.dense(3, 4, d.activations.sigmoid);

process = d.nn(LAYERSPEC, d.activations.sigmoid);// :> _;
