import("stdfaust.lib");
df = library("diff.lib");

N = 5;

var(i) = hslider("x%i", 0, -1, 1, .01);

// vars = df.vars(par(i, N, -~_ <: attach(hbargraph("x%j", -1, 1)) with {j = i + 1; }));
vars = df.vars(par(i, N, hslider("x%j", 0, -1, 1, .01) with {j = i + 1; }));

d = df.env(vars);

estimate1 = _,.5,-.1 : *,_ : + : sin;

estimate2 = _,var(1),var(2) : *,_ : +;

estimate3 = _ <: sum(i, N, _,i : @,var(i+1) : *);

estimate4 = _ <: par(i, N, _,i : @,var(i+1) : *) :> _ : seq(i, 2, sin);

estimate5 = + ~ (var(1),_ : *);

estimate6 = f : + ~ g
with {
    f = bank(ceil(N/2),0);
    g = bank(n, 1)
    with {
        n = ceil(N/2),ceil(N/2)-1 : select2((N,2 : %,0 : !=));
    };

    bank(0,fb) = 0;
    bank(NTAPS,fb) = _ <: sum(n, NTAPS, _,n : @,var(idx) : *
    with {
        idx = n+1,n+NTAPS+1 : select2(fb);
    });
};

estimate7 = lpf(400)
with {
    lpf(fc) = _
        <: *(b0), (mem : *(b1))
        : + : + ~ (_,(0,a1 : -) : *)
    with {
        w = 2*ma.PI*fc;
        c = 1/tan(w*0.5/ma.SR);
        d = 1+c;
        b0 = 1/d;
        b1 = 1/d;
        a1 = (1-c)/d;
    };
};

estimate8 = fi.fir(par(i, N, var(i+1)));

estimate9(x,y) = y-x;

activations = environment {
    sigmoid = *(-1) : 1,exp,1 : _,+ : /;
};

estimate10 = neuron(N, activations.sigmoid)
with {
    neuron(NINPUTS, Y) = Z : Y
    with {
        WX = sum(i, NINPUTS-1, var(i+1),_ : *);
        b = var(2*NINPUTS+1);

        Z = WX + b;
    };
};

process = d.fwdAD(estimate10);
