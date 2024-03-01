import("stdfaust.lib");
df = library("diff.lib");

// TODO: force learnable algo to respect RoC?

NTAPS = 3;
NVARS = 2*NTAPS-1;

p = in,in
    : hgroup("Differentiable IIR",
        (route(NVARS+2,NVARS+2,par(n,NVARS,(n+1,n+3)),(NVARS+1,1),(NVARS+2,2))
        : vgroup("Hidden", truth),vgroup("Learned", learnable)
        : route(2+NVARS,4+NVARS,
            // Route ground truth output to df.learn and to output.
            (1,1),(1,NVARS+3),
            // Route learnable output to df.learn and to output.
            (2,2),(2,NVARS+4),
            // Route gradients to df.learn.
            par(n,NVARS,(n+3,n+3))
        )
        : vgroup("[1]Loss/gradient", df.learnL2(1<<3,2e-1,NVARS),_,_)) ~ (!,si.bus(NVARS))
    // Cut the gradients, but route loss to output so the bargraph doesn't get optimised away.
    ) : _,si.block(NVARS),_,_
with{
    in = no.noise;

    // truth = _ <: sum(n, NTAPS, _,n : @,bn : *
    //     with {
    //         bn = hslider("[0]b%n", n==0, -1., 1., .001);
    //     }) : + ~ (_ <: sum(n, NTAPS-1, _,n : @,an : *
    //     with {
    //         m = n+1;
    //         an = hslider("[1]a%m", 0, -1., 1., .001);
    //     }));

    truth = f : +~g
    with {
        f = bank(NTAPS,0);
        g = bank(NTAPS-1,1);

        bank(N,isA) = _ <: sum(n, N, _,n : @,coeff : *
            with {
                m = n+isA;
                coeff = isA,hslider("[0]b%m", m==0, -1, 1, .001),hslider("[1]a%m",  0, -1, 1, .001) : select2;
            });
    };

    learnable = df.input(NVARS),si.bus(NVARS)
        : f,si.bus(NTAPS-1)
        : route(NVARS+NTAPS,NVARS+NTAPS,
            par(n,NVARS+1,(n+1,n+1+NTAPS-1)),
            par(n,NTAPS-1,(NVARS+n+2,n+1)))
        : df.rec(h~g,NTAPS-1)
        with {
            f = bank(NTAPS,0,0) : sumall(NTAPS);
            g = bank(NTAPS-1,NTAPS,1) : sumall(NTAPS-1);
            h = df.diff(+,NVARS);

            sumall(1) = df.diff(_,NVARS);
            sumall(2) = df.diff(+,NVARS);
            sumall(N) = seq(n,N-2,df.diff(+,NVARS),par(m,N-n-2,df.diff(_,NVARS))) : sumall(2);

            bank(N,offset,isA) =
                route(NVARS+N+1,N*NVARS+2*N,
                    par(n,N,
                        par(m,NVARS+1,(m+1,(m+1)+n*(NVARS+2))),
                        (NVARS+n+2,(NVARS+2)*(n+1))
                    )
                ) : par(n, N,
                    df.diff(_,NVARS),df.const(n,NVARS),_
                    : df.diff(@,NVARS),df.var(n+offset+1,coeff,NVARS)
                    : df.diff(*,NVARS)
                    with {
                        m = n+isA;
                        coeff = -~_ <: attach((isA,hbargraph("[0]b%m",-1,1),hbargraph("[1]a%m",-1,1)) : select2);
                    }
                );
        };
};

process = p;
