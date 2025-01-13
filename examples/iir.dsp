import("stdfaust.lib");
df = library("diff.lib");

// TODO: force estimate algo to respect RoC?

NTAPS = 3;

// process = in <: _,_ : hgroup("Differentiable IIR", df.backprop(truth, estimate, d.learnMSE(1<<3, d.optimizers.SGD(alpha))))
// with{
//     alpha = hslider("alpha", 1e-3, 1e-4, 1e-1, 1e-5);
//     in = no.noise;

process = in <: _,_
    : hgroup("Differentiable IIR", df.backprop(target, estimate, d.losses.L2(1<<2, alpha)))
    : route(3,4,(3,1),(3,2),(1,3),(2,4)) : par(i, 4, *(out))
with {
    out = hslider("Volume", 0, 0, 1, .01);
    in = no.noise;
    alpha = hslider("alpha [scale:log]", 1e-3, 1e-4, 1, 1e-8);

    target = f : +~g
    with {
        f = bank(NTAPS,0);
        g = bank(NTAPS-1,1);

        bank(0,isFB) = 0;
        bank(N,isFB) = _ <: sum(n, N, _,n : @,coeff : *
            with {
                m = n+isFB;
                coeff = isFB,hslider("[0]b%m", m==0, -1, 1, .001),hslider("[1]a%m",  0, -1, 1, .001) : select2;
            });
    };

    vars = df.vars(coeffs)
    with {
        coeffs = par(n,2*NTAPS-1,-~_);
    };

    d = df.env(vars);

    estimate = d.input,si.bus(vars.N) // A differentiable input, N gradients
        // FIR bank, gradients for g
        : f,si.bus(NTAPS-1)
        // Output and partial derivatives to h; gradients to g.
        : ro.crossNM(vars.N+1, NTAPS-1)
        with {
            nIN = 1+vars.N+NTAPS-1; // 1 output + N partial derivatives + NTAPS-1 gradients
        }
        // Sum ~ IIR bank
        : df.rec(h~g,NTAPS-1)
        with {
            f = bank(NTAPS,0,0);
            g = bank(NTAPS-1,NTAPS,1);
            h = d.diff(+);

            // N:      number of filter taps
            // offset: offset for the index of the differentiable parameter
            // isFB:   is this a feedforward (0) or a feedback (1) tap?
            bank(0,offset,isFB) = d.diff(0);
            bank(N,offset,isFB) =
                route(vars.N+N+1,N*vars.N+2*N,
                    par(n,N,
                        // Partial derivatives to differentiable delay
                        par(m,vars.N+1,(m+1,(m+1)+n*(vars.N+2))),
                        // Gradients to filter coefficients
                        (vars.N+n+2,(vars.N+2)*(n+1))
                    )
                ) : par(n, N,
                    d.diff(_),d.diff(n),_
                    : d.diff(@),coeff
                    : d.diff(*)
                    with {
                        m = n+isFB;
                        // Not ideal to have to attach the meter here...
                        coeff = vars.var(n+offset+1) : (_ <: attach((isFB,hbargraph("[0]b%m",-1,1),hbargraph("[1]a%m",-1,1)) : select2)),si.bus(vars.N);
                    }
                ) : d.sumall(N);
        };
};
