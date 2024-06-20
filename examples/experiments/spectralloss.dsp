import("stdfaust.lib");

N = 1 << 4;

in = os.osc(hslider("freq [scale:log]", 500, 20, 20000, 0.1));

truth = in <: _,an.rfft_analyzer_magsq(N) 
    : par(i, N/2+2, _ : si.smooth(ba.tau2pole(0.05)))
    : route(N/2+2, N+3, (1,1), par(i, N+2, (i+2,i+2), (i+2,i+N/2+3))) 
    : (_,(display :> _) : attach), par(i,N/2+1,_)
    with {
        display = par(n,N/2+1,meter(n));
        meter(n) = graph(n);
        graph(n) = group(vbargraph("bin%2n",0,250));
        group(x) = hgroup("Spectrum analyser_truth", x);
    };

learnable = os.osc(440) <: _,an.rfft_analyzer_magsq(N) 
    : par(i, N/2+2, _ : si.smooth(ba.tau2pole(0.05)))
    : route(N/2+2, N+3, (1,1), par(i, N+2, (i+2,i+2), (i+2,i+N/2+3))) 
    : (_,(display :> _) : attach), par(i,N/2+1,_)
    with {
        display = par(n,N/2+1, int : meter(n));
        meter(n) = graph(n);
        graph(n) = group(vbargraph("bin%2n",0,250));
        group(x) = hgroup("Spectrum analyser_learnable", x);
    };

loss = route(N+2, N+2, par(i,N/2+1,(i+1,x),(i+N/2+2,x+1)
    with{
        x = 2*i+1;
    }))
    : par(i,N/2+1,- : abs) 
    : sum(i,N/2+1,_)
    : /(N/2+1) : hbargraph("loss",0,1);
    
process = truth, learnable
        : _, route(N/2+2, N/2+2, (N/2+2,1),par(i,N/2+1,(i+1,i+2))), par(i,N/2+1,_)
        : _, _, loss;