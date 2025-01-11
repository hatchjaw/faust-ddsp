import("stdfaust.lib");
df = library("diff.lib");

LAYERSPEC = ((4,2,1,4));

v = df.weightsAndBiases(LAYERSPEC);
d = df.env(v);

// TODO: gradient averaging
process = no.noise <: nn
with {
    nn(y) = hgroup("Neural Network",
        vgroup("Weights & Biases", d.nn(LAYERSPEC, d.activations.sigmoid)) :> vgroup("Loss & Gradients", d.losses.L2(1<<0, 1e-1, y))
    ) ~ (!,si.bus(v.N)) : (y,_,si.block(v.N));
};
