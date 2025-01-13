df = library("diff.lib");
import("stdfaust.lib");

// This example is a simple neural network that can classify between a sine wave and a square wave.
// Just to visualise how the weights and biases interact with each other, I have added a delay of 10000 epochs.
// The dataset is generated in such a way that the label is changed every 10 epochs.
// The neural network has 1 input and 1 output. The activation function is sigmoid and the loss function is L1.
// Of course, the weights and biases here are parameters of the neural network and can be used for inference later on.

// Slows down the process by 10000 epochs -- just to visualise the weights and biases.
// The update status is 1 if the current epoch is a multiple of 10000, else 0 (this is a helper signal).
// outputs: current epoch, update status (1 or 0)
epoch_calculator = (_ ~ +(1)) : ((_ % 10000 == 0),0,1 : select2) <: +(_)~_, _
                : (_ <: attach(hbargraph("epochs", 0, 100))), _;  

// This is a dataset generator that generates 2 types of signals: sine wave and square wave.
// It also generates a label (0 = sine wave, 1 = square wave) and a change_bit signal that changes the label every 10 epochs.
// Outputs: label (0 = os.osc, 1 = os.square), sample, change_bit
// In this case, the dataset consists of 1 sine wave of 440 Hz and 1 square wave of 440 Hz.
dataset_generator(change_bit) = change_bit <: _, _
                                : (change_bit, 0, 1 : select2), _
                                : (+(_)~((_ <: _ % 10 == 0, _), 0 : select2)), _
                                : (_ <: attach(hbargraph("local-epoch", 0, 10))), _
                                : (((_ % 10 == 0, 0, 1) : select2) : ba.toggle <: attach(hbargraph("label", 0, 1))), _
                                : (_ <: _, (_, os.osc(440), os.square(440) : select2)), _;

// This is the neural network setup.
// The neural network has one layer with one weight (and one bias).
// The activation function is sigmoid and the loss function is L2.
LAYERSPEC = ((1, 1));

v = df.weightsAndBiases(LAYERSPEC);
d = df.env(v);

nn(y) = hgroup("Neural Network",
    vgroup("Weights & Biases", d.nn(LAYERSPEC, d.activations.sigmoid)) :> vgroup("Loss & Gradients", d.losses.L2(1<<3, 1e-2, y))
) ~ (!,si.bus(v.N)) : (y,_,si.block(v.N));

// The process is the composition of the epoch_calculator, dataset_generator and the neural network.
// This setup is used to visualise the weights and biases of the neural network.
process = epoch_calculator : _,dataset_generator(_) : _,(_,_ : nn),_;
