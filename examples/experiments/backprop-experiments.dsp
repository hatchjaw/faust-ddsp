df = library("diff.lib");
import("stdfaust.lib");

N = 4; // no of inputs
n = 2; // no of neurons

process = si.bus(8) : backpropFC(1,2);

backpropFC(N, n) = route((2*N+1)*n + n, (2*N+1)*n + n, par(i, n, (i+1, i+1+(2*N+1)*i), 
        par(j, 2*N+1, (n+j+1+(2*N+1)*i, (i*(2*N+2)+1) + j+1))))
        : par(i, n, autoChainRule) : prepareGrads(N, n) 
        : par(i, (N+1)*n, _), gradAveraging(N, n)
        with {
            autoChainRule = route(2*N+2, 4*N+2, par(i, 2*N+1, (1, 2*i+1), (i+2, 2*i+2))) : autoChain;
            autoChain = par(i, 2*N+1, (v.var(1), d.input : d.diff(*)) : _, si.block(4*N+2)
                        with {
                            v = df.vars((par(j, 4*N+2, _)));
                            d = df.env(v);
                        }
                    );

            prepareGrads(N, n) = route((2*N+1)*n, (2*N+1)*n, par(i, n, 
                        par(j, N+1, (j+1+i*(2*N+1), (1+N)*i+(j+1))), 
                        par(k, N, (k+1+N+1+(i)*(2*N+1),(k+1+i*N+((N+1)*n))))));
            
            gradAveraging(N, n) = route(N*n, N*n, par(i, N, par(j, n, (j*N+i+1, i*n+1+j)))) : par(i, N, sum(j, n, _)) : par(i, N, _ : /(n));
        };