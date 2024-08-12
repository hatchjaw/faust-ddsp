df = library("diff.lib");
import("stdfaust.lib");


process = si.bus(2) : (df.fc(3, 2, df.activations.sigmoid, 0.1)
        : ((df.fc(1, 3, df.activations.sigmoid, 0.1) : df.losses.L1(1<<3, 1, 3)), par(i, 15, _))
        ~ (si.block(1), si.bus(4)) : si.block(5), par(i, 18, _)
        : df.gradAveraging(1, 3), par(i, 15, _)
        : df.chainApply(3, 2, 0) : df.chainApply(2, 2, 1) : df.chainApply(1, 2, 2))
        ~ par(i, 3, si.bus(3), si.block(2)) : par(i, 3, si.block(3), si.bus(2))
        : df.gradAveraging(3, 2);
