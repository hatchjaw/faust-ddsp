df = library("diff.lib");
import("stdfaust.lib");

process = si.bus(1) : (df.fc(2, 1, df.activations.sigmoid, 0.1)
        : (df.fc(3, 2, df.activations.sigmoid, 0.1), par(i, b.next_lines(1), _)
        : ((df.fc(1, 3, df.activations.sigmoid, 0.1) : df.losses.L1(1<<3, 1, 3)), par(i, b.next_lines(2), _)
        : b.start(b.N - 1))
        ~ (si.block(1), si.bus(4)))
        ~ (si.block(5), si.bus(9)))
        ~ (si.block(14), si.bus(4))
        with {
            b = df.backpropNN((1, 3, 3, 2, 2, 1));
        };
