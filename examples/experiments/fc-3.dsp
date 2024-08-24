    df = library("diff.lib");
    import("stdfaust.lib");
     
     
    // trying to slow down the process to attempt to visualise how weights and biases interact with each other
    // outputs: current epoch, update status (1 or 0)
    epoch_calculator = (_ ~ +(1)) : ((_ % 10000 == 0),0,1 : select2) <: +(_)~_, _
                    : (_ <: attach(hbargraph("epochs", 0, 100))), _;  
     
    // a simulation of how a dataset would look
    // Outputs: label (0 = osc, 1 = square), sample, change_bit
    dataset_generator(change_bit) = change_bit <: _, _
                                    : (change_bit, 0, 1 : select2), _
                                    : (+(_)~((_ <: _ % 10 == 0, _), 0 : select2)), _
                                    : (_ <: attach(hbargraph("local-epoch", 0, 10))), _
                                    : (((_ % 10 == 0, 0, 1) : select2) : ba.toggle <: attach(hbargraph("label", 0, 1))), _
                                    : (_ <: _, (_, os.osc(440), os.square(440) : select2)), _;
     
    ml_setup(Y) = (df.fc(1, 1, df.activations.sigmoid, 1e-7)
            : df.losses.L1(1<<3, Y, 1)) // Note that this is the last layer of the NN. As a result, you can calculate the loss here. The loss function is L1.
            ~ (si.block(1), si.bus(2))
            : (_ : hbargraph("losses", 0, 1)), si.bus(3);
     
     
    process = epoch_calculator : _, dataset_generator(_) : _, (_, _ : ml_setup), _;