#=doc
Initializers
============
Interface
---------
=#

#=doc
.. class:: AbstractInitializer

   The abstract base class for all initializers.

To define a new initializer, it is
enough to derive a new type, and implement one or more of the following methods:

.. function:: _init_weight(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)

Or, if full behavior customization is needed, override the following function

.. function:: call(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
=#
abstract AbstractInitializer

function call(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  strname = string(name)
  if endswith(strname, "bias")
    _init_bias(self, name, array)
  elseif endswith(strname, "gamma")
    _init_gamma(self, name, array)
  elseif endswith(strname, "beta")
    _init_beta(self, name, array)
  elseif endswith(strname, "weight")
    _init_weight(self, name, array)
  elseif endswith(strname, "moving_mean")
    _init_zero(self, name, array)
  elseif endswith(strname, "moving_var")
    _init_zero(self, name, array)
  else
    _init_default(self, name, array)
  end
end

function _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end
function _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 1
end
function _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end
function _init_zero(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end

#=doc
Built-in initializers
---------------------
=#
#=doc
.. class:: UniformInitializer

   Initialize weights according to a uniform distribution within the provided scale.
=#
immutable UniformInitializer <: AbstractInitializer
  scale :: AbstractFloat
end
#=doc
.. function UniformInitializer(scale=0.07)

   Construct a :class:`UniformInitializer` with the specified scale.
=#
UniformInitializer() = UniformInitializer(0.07)

function _init_weight(self :: UniformInitializer, name :: Base.Symbol, array :: NDArray)
  rand!(-self.scale, self.scale, array)
end

#=doc
.. class:: NormalInitializer

   Initialize weights according to a univariate Gaussian distribution.
=#
immutable NormalInitializer <: AbstractInitializer
  μ :: AbstractFloat
  σ :: AbstractFloat
end
#=doc
.. function:: NormalIninitializer(; mu=0, sigma=0.01)

   Construct a :class:`NormalInitializer` with mean ``mu`` and variance ``sigma``.
=#
NormalInitializer(; mu=0, sigma=0.01) = NormalInitializer(mu, sigma)

function _init_weight(self :: NormalInitializer, name :: Base.Symbol, array :: NDArray)
  randn!(self.μ, self.σ, array)
end

#=doc
.. class:: XavierInitializer

   The initializer documented in the paper [Bengio and Glorot 2010]: *Understanding
   the difficulty of training deep feedforward neuralnetworks*.

   There are several different version of the XavierInitializer used in the wild.
   The general idea is that the variance of the initialization distribution is controlled
   by the dimensionality of the input and output. As a distribution one can either choose
   a normal distribution with μ = 0 and σ² or a uniform distribution from -σ to σ.

   Several different ways of calculating the variance are given in the literature or are
   used by various libraries.

   - [Bengio and Glorot 2010]: ``mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 1)``
   - [K. He, X. Zhang, S. Ren, and J. Sun 2015]: ``mx.XavierInitializer(distribution = mx.xv_gaussian, regularization = mx.xv_in, magnitude = 2)``
   - caffe_avg: ``mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 3)``
=#

@enum XavierDistribution xv_uniform xv_normal
@enum XavierRegularization xv_avg xv_in xv_out

immutable XavierInitializer <: AbstractInitializer
  distribution :: XavierDistribution
  regularization :: XavierRegularization
  magnitude :: Float64
end

XavierInitializer(; distribution = xv_uniform, regularization = xv_avg, magnitude = 3.0) = XavierInitializer(distribution, regularization, magnitude)

function _init_weight(self :: XavierInitializer, name :: Base.Symbol, array :: NDArray)
  dims    = size(array)
  fan_in  = prod(dims[2:end])
  fan_out = dims[1]

  if self.distribution == xv_uniform
    func(σ, data) = rand!(-σ, σ, data)
  elseif self.distribution == xv_normal
    func(σ, data) = randn!(0.0, σ, data)
  end

  if self.regularization == xv_avg
    factor = (fan_in + fan_out) / 2
  elseif self.regularization == xv_in
    factor = fan_in
  elseif self.regularization == xv_out
    factor = fan_out
  end

  σ = √(self.magnitude / factor)

  func(σ, array)
end

immutable BilinearInitializer <: AbstractInitializer
end

function _init_weight(self :: BilinearInitializer, name :: Base.Symbol, array :: NDArray)
  dims = size(array) # (K, K, C, S)
  @assert length(dims) == 4

  width = dims[1]
  height = dims[2]

  @assert width == height

  f = ceil(Int, width / 2) # factor
  c = (2f - 1 - f % 2) / 2f # center

  filter = Base.zeros(width, height)

  for x in 0:(width-1)
    for y in 0:(height-1)
      filter[x+1, y+1] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    end
  end

  @nd_as_jl rw=array begin
    for i in 1:dims[4] # Samples
      for j in 1:dims[3] # Channel
        array[:, :, j, i] = filter
      end
    end
  end
end
