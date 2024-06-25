import("stdfaust.lib");

N = 1 << 4;

in = hslider("frequency_learnable", 2000, -20000, 20000, 0.01);
// dy/dx
learnable = (_ : -~_ : hbargraph("learned frequency", -20000, 20000)) <: _,(os.square : an.rfft_analyzer_magsq(N)), 1
    : _,par(i, N/2+1, si.smooth(ba.tau2pole(0.05))),_
    : _,route(N/2+1, N+2, par(i, N+2, (i+1,i+1), (i+1,i+N/2+2))),_
    : (_ ,(display :> _) : attach), par(i,N/2+1,_),_
    with {
        display = par(n,N/2+1,meter(n));
        meter(n) = graph(n);
        graph(n) = group(vbargraph("bin%2n",0,250));
        group(x) = hgroup("Spectrum analyser_learnable", x);
    };

truth = 500 <: _,(os.square : an.rfft_analyzer_magsq(N))
    : _,par(i, N/2+1, _ : si.smooth(ba.tau2pole(0.05)))
    : _,route(N/2+1, N+2, par(i, N+2, (i+1,i+1), (i+1,i+N/2+2))) 
    : (_,(display :> _) : attach), par(i,N/2+1,_)
    with {
        display = par(n,N/2+1,meter(n));
        meter(n) = graph(n);
        graph(n) = group(vbargraph("bin%2n",0,250));
        group(x) = hgroup("Spectrum analyser_truth", x);
    };

loss = route(N+2, N+2, par(i,N/2+1,(i+1,x),(i+N/2+2,x+1)
    with{
        x = 2*i+1;
    }))
    : par(i,N/2+1,-) 
    : sum(i,N/2+1,_)
    : /(N/2+1) : hbargraph("loss",0,1);

N_vars = 1;

// dy/dxi * loss
gradient = ((_, (_, 1e-6 : max) : *), _: *) <: attach(hbargraph("grad", -0.01, 0.01));

process = (truth, learnable
        : _, route(N/2+2, N/2+2, (N/2+2,1),par(i,N/2+1,(i+1,i+2))), par(i,N/2+1,_), _
        : _, _, loss, _
        : _, _, (_, _, 1e-1 : gradient)) ~ (!, !, _);
