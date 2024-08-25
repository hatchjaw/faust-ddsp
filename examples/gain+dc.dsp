import("stdfaust.lib");
df = library("diff.lib");

process =
    no.noise,no.noise
    // no.noises(2,0),no.noises(2,1)
    // os.osc(100),os.osc(100)
    // Route inputs to ground truth and learnable algos; route gradients to learnable algo.
    : (route(vars.N+2,vars.N+2,par(n,vars.N,(n+1,n+3)),(vars.N+1,1),(vars.N+2,2))
        : groundTruth,learnable // Could use si.bus here...
        // Copy ground truth and learnable outputs to audio outputs.
        : (_ <: _,_),(_ <: _,_),si.bus(vars.N)
        : route(4+vars.N,4+vars.N,(1,1),(2,vars.N+3),(3,2),(4,vars.N+4),par(n,vars.N,(n+5,n+3)))
        // Recurse gradients
        // : learn,_,_) ~ (!,si.bus(vars.N))
        : d.learnMSE(1<<5,d.optimizer.SGD(1e-2)),_,_) ~ (!,si.bus(vars.N))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    : _,si.block(vars.N),_,_
with {
    hiddenGain = hslider("[0]Hidden gain", .5, 0, 2, .01);
    hiddenDC = hslider("[1]Hidden dc", .5, -1, 1, .01);

    gainDC(g, d) = _,g,d : *,_ : +;
    groundTruth = gainDC(hiddenGain, hiddenDC);

    vars = df.vars((gain,dc))
    with {
        gain = -~_ <: attach(hbargraph("[5]Learned gain",0,2));
        dc = -~_ <: attach(hbargraph("[6]Learned dc",-1,1));
    };

    d = df.env(vars);

    learnable = d.input,vars.var(1),vars.var(2)
        : d.diff(*),d.diff(_)
        : d.diff(+);
};
