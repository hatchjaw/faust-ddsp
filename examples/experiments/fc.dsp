df = library("diff.lib");
import("stdfaust.lib");

// outputs: current epoch, update status (1 or 0)
epoch_calculator = (_ ~ +(1)) : ((_ % 10000 == 0),0,1 : select2) <: +(_)~_, _
                : (_ <: attach(hbargraph("epochs", 0, 100))), _;  

// Outputs: sample, label (0 = osc, 1 = square), change_bit
dataset_generator(change_bit) = change_bit <: _, _
                                : (change_bit, 0, 1 : select2), _
                                : (+(_)~((_ <: _ % N_eps == 0, _), 0 : select2)), _
                                : (_ <: attach(hbargraph("local-epoch", 0, 10))), _
                                : (((_ % N_eps == 0, 0, 1) : select2) : ba.toggle <: attach(hbargraph("label", 0, 1))), _
                                : (_ <: (_, 0, 1 : select2), _), _;

// process = si.bus(2) : df.fcLast(1, 2, df.activations.sigmoid, df.losses.L1, 1, 0.1);
// process = si.bus(5) : df.neuron(2, df.activations.sigmoid, 0.1);
process = si.bus(11) : df.fc(3, 2, df.activations.sigmoid, 0.1) : ((df.fc(1, 3, df.activations.sigmoid, 0.1) : df.losses.L1(1<<3, 1, 3)), par(i, 15, _)
        : df.backpropFC(3, 2, 0) : df.backpropFC(2, 2, 1))
        ~ si.bus(4);
        // ~ par(i, 3, si.bus(3), si.block(2)) : par(i, 3, si.block(3), si.bus(2));