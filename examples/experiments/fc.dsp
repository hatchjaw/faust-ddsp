df = library("diff.lib");
import("stdfaust.lib");

process = si.bus(4) : df.fc(2, 4, df.activations.sigmoid, 0.1) : df.fcLast(1, 2, df.activations.sigmoid, df.losses.L1, 1, 0.1), par(i, 20, _);