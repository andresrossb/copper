#no \psi core
#80% i understood as 4 to 1 forces to energy
#not using stepwise cutoff
#7.4 not 3.9
#not optimizing the radial function, i.e missing out on 500 ish parameters
#did 3 and 12 to get 910 params, compared to 756 in paper

#Initialize all the workers
using Distributed
Distributed.addprocs(70)
@everywhere using Pkg
@everywhere Pkg.activate(pwd())
@everywhere Pkg.instantiate()
@show nprocs()

#import packages
@everywhere using JuLIP, Zygote, Flux, IPFitting, ACE, StaticArrays, ACEflux # LinearAlgebra, RecipesBase, 
using Optim, LineSearches, JLD#, TensorBoardLogger, Logging
@everywhere import Base.copyto!


# # ------------------------------------------------------------------------
# #    Define the model
# # ------------------------------------------------------------------------

@show "imported"

F(ϕi) = sign(ϕi)*(sqrt(abs(ϕ) + e^(-abs(ϕi))/4) - e^(-abs(ϕi))/2)
FS(ϕ) = (ϕ[1] + F(ϕ[2]))#*fcut(ϕcore) + ϕcore   
model = Chain(Linear_ACE(3, 12, 2), GenLayer(FS), sum) #not sure about correlation order
s1,s2 = size(model[1].weight)

target = deepcopy(collect(Iterators.flatten(Flux.params(model)[1])))

@show length(target)

@eval @everywhere model=$model

@everywhere pot = FluxPotential(model, 7.4) ;

#loss function to minimize
@everywhere sqr(x) = x.^2
@everywhere loss(pt,x,y) =  (energy(pt, x) - y[1])^2 + 4*sum(sum(sqr.(forces(pt, x) - y[2])))

# # ------------------------------------------------------------------------
# #    Data
# # ------------------------------------------------------------------------

#import and take only the data we need
data = IPFitting.Data.read_xyz("/zfs/users/aross88/aross88/copper/Si.xyz", energy_key="dft_energy", force_key="dft_force")

begin 
   Y = []  
   X = []
   for i in 1:length(data)
      push!(Y, [ data[i].D["E"][1], ACEflux.matrix2svector(reshape(data[i].D["F"], (3, Int(length(data[i].D["F"])/3)))) ])
      push!(X, data[i].at)
   end
end

atlen = []
for x in X
   push!(atlen, length(x))
end

@show sum(atlen)
@show sum(atlen) / length(X)
@show length(X)

# @everywhere function ereff(X, Y)
#    return sum([Y[i][1] / length(X[i]) for i in 1:length(Y)]) / length(Y)
# end

# @everywhere function substractreff!(Y, ereff)
#    for y in Y
#       y[1] -= ereff
#    end
# end

# @everywhere substractreff!(Y, ereff(X, Y))


#Use all workers available, except the main one
#the main one is used for running optim
@everywhere np = nprocs() - 1 

#takes the number of processors and divides X and Y into them based on their length
#it puts the data on all processors, this is inefficient since a worker only needs
#the data it will work with.
function data4workers(np, X, Y)
   lendata = [length(x) for x in X]
   srt = reverse(sortperm(lendata)) #the sorted indexes by length max to min
   X .= X[srt]
   Y .= Y[srt] 
   #deals jobs same way cards are dealt in a "snake draft" manner
   #meaning we go bottom to top of the list of workers dealing data and then back
   #top to bottom. This is to get the best distribution of work.
   i = 1
   rev = 1
   x4worker = [[] for _ in 1:np]
   y4worker = [[] for _ in 1:np]
   while i <= length(X)
      for j in 1:np
         if(i <= length(X))
            if(rev%2 == 0)
               push!(x4worker[np - j + 1], X[i])
               push!(y4worker[np - j + 1], Y[i])
            else
               push!(x4worker[j], X[i])
               push!(y4worker[j], Y[i])
            end
            i+=1
         else
            break
         end
      end
      rev += 1
   end
   return x4worker, y4worker
end

x4worker, y4worker = data4workers(np, X, Y);
@eval @everywhere x4worker=$x4worker
@eval @everywhere y4worker=$y4worker

# # ------------------------------------------------------------------------
# #    functions (for now they have to be here once switched they will need to take
# #    arguments instead of using global variables)
# # ------------------------------------------------------------------------

#need some consistent way to do this (our gradient objects for forces are very ugly)
@everywhere veclength(params::Flux.Params) = sum(length, params.params)
@everywhere veclength(x) = length(x)
@everywhere Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

@everywhere function copyto!(v::AbstractArray, pars::Flux.Params)
   @assert length(v) == veclength(pars)
   s = 1
   for g in pars.params
       l = length(g)
       v[s:s+l-1] .= vec(g)
       s += l
   end
   v
end

@everywhere function copyto!(pars::Flux.Params, v::AbstractArray)
   s = 1
   for p in pars.params
       l = length(p)
       p .= reshape(v[s:s+l-1], size(p))
       s += l
   end
   pars
end

#HARDCODED
#will only work with a model with 1 layer of parameters
#idea is we simply add over the gradients and return an array
function gsum2mat(gs)
   sol = gs[1].grads[gs[1].params[1]]
   for i in 2:length(gs)
      sol = sol .+ gs[i].grads[gs[i].params[1]]
   end
   return sol
end

#gradient performed by each worker 
@everywhere function _gl(x, y)
   l, back = Zygote.pullback(()->loss(pot,x,y), Flux.params(model))
   return(l, back(1))
end
   
#function that dispatches all workers and returns the loss and gradients
function gradcalc()
   futures = Vector{Future}(undef, np)
   for i in 1:np
       f = @spawnat (i+1) map((x,y) -> _gl(x,y), x4worker[i], y4worker[i])
       futures[i] = f
   end
   fg = fetch.(futures)
   tl = 0.0
   grds = []
   for c1 in fg
       for c2 in c1
           l, g = c2
           tl += l
           push!(grds, g)
       end
   end

   return tl / length(X), gsum2mat(grds) ./ length(X)
end

#function to only calculate loss, still multi-processed 
function losscalc()
   futures = Vector{Future}(undef, np)
   for i in 1:np
      f = @spawnat (i+1) sum(map((x,y)->loss(pot,x,y), x4worker[i], y4worker[i]))
      futures[i] = f
   end
   return sum(fetch.(futures)) / length(X)
end


#the optim wrapper, we use collect instead of the copyto which is sketchy
#will most likely work for FS, but for more layers will need to find a consistent
#way to flattten gradients
fg! = function (F,G,w)

   model[1].weight = reshape(w,s1,s2) #we reshape the flat array into a matrix and put it inside the local model
   deepp = deepcopy(Flux.params(model)) #we deepcopy the parameters of the model
   @eval @everywhere deepp=$deepp #we send to all workers the deep copy
   @everywhere Flux.loadparams!(model, deepp) #we load this new parameters to the model in each worker

   # @eval @everywhere w=$w
   # @everywhere pot.model[1].weight = reshape(w,2,64)

   if G != nothing
      l, grads = gradcalc()
      copyto!(G, collect(Iterators.flatten(grads)))
      return l
   end
   if F != nothing
      return losscalc()
   end
end


# # ------------------------------------------------------------------------
# #    IP linear QR
# # ------------------------------------------------------------------------
# weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0 ),)

# ord = 3
# maxdeg = 14

# Bsel = SimpleSparseBasis(ord, maxdeg)
# species = [:Si]
# B1p = ACEatoms.ZμRnYlm_1pbasis(; maxdeg=maxdeg, Bsel = Bsel, 
#                                  rin = 1.2, rcut = 5.0)
# ACE.init1pspec!(B1p, Bsel)
# basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
# cSi = randn(length(basis))
# dm = Dict(:Si => ACE.LinearACEModel(basis, cSi; evaluator = :standard))

# V = ACEatoms.ACESitePotential(dm)
# ipbasis = ACEatoms.basis(V)


# dB = LsqDB("./testfile", ipbasis, data);
# (IP,info) = lsqfit(dB; verbose=true, asmerrs=false, lasso=false, weights = weights, regularisers = [])


# # ------------------------------------------------------------------------
# #    opt lbfgs
# # ------------------------------------------------------------------------

p0 = ones(length(target))

juliacb = tr -> begin
   iteration = tr[end].iteration
   @show iteration
   @show tr[end].value
   save("Si_linear_LBFGS.jld", "tloss", [t.value for t in tr], "gnorm", [t.g_norm for t in tr], "param", tr[end].metadata["x"])
   false
end

@show "start opt"
iters = 1000

res = Optim.optimize(Optim.only_fg!(fg!), p0, LBFGS(), Optim.Options(callback=juliacb, show_every=10, iterations=iters, store_trace=true, extended_trace=true))

@show res

using Plots, JLD

d = load("Si_linear_LBFGS.jld")

plot(d["tloss"], yaxis=:log, xlab="iterations", ylab="train loss", label="LBFGS", title="FS Si")

plot(d["gnorm"], yaxis=:log, xlab="iterations", ylab="grad norm", label="LBFGS", title="FS Si")

