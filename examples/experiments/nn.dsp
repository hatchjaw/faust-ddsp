import("stdfaust.lib");
df = library("diff.lib");

NINPUTS = 2;
LAYERSPEC = ((2, 2));//, 4, 1));

// TODO: abstract this stuff away.
b = par(n, outputs(LAYERSPEC) / 2, par(m, ba.take(2 * n + 1, LAYERSPEC), df.var(hbargraph("b%n%m", -5, 5), 0)));
W = par(i, outputs(LAYERSPEC) / 2, par(j, ba.take(2 * i + 1, LAYERSPEC), par(k, ba.take(2 * i + 2, LAYERSPEC), df.var(hbargraph("w%i%j%k", -5, 5), 0))));

v = df.vars((W,b));
d = df.env(v);

processE = d.layers.inputLayer(LAYERSPEC)
    : d.layers.dense(2, 2, d.activations.sigmoid(d));
    //<: d.layers.dense(3, 4, d.activations.sigmoid(d));

process = d.nn(LAYERSPEC, d.activations.sigmoid(d)) :> _;
