"""
Custom and native operators written in Julia.
The interface is given by the abstract type [`Operactor`](@ref).
"""
module Custom

using Compat
import Compat: String

import ..mx
import ..mx: NDArray

export assign

"""
    Operator
"""
abstract Operator

"""
    forward(op, is_train, req, in_data, out_data, aux)

Forward interface. Custom operators must override this.

# Arguments:
* `is_train::Bool`: Whether we are in training
* `req::Vector{Symbol}`: How to assign to out_data. Can be :null, :write, :inplace, or :add. You can use `assign(dst, req, src)` to handle this
* `in_data::Vector{NDArray}`
* `out_data::Vector{NDArray}`
* `aux::Vector{NDArray}`
"""
function forward(op::Operator, is_train, req, in_data, out_data, aux)
  throw(MethodError(forward, (op, is_train, req, in_data, out_data)))
end

"""
Backwards interface. Custom operators must override this.
"""
function backward(op::Operator, req, out_grad, in_data, out_data, in_grad, aux) 
  throw(MethodError(backward, (op, req, out_grad, in_data, out_data, in_grad, aux)))
end

function assign(dst, req, src)
  if req == :null
    return nothing
  elseif req == :write || req == :inplace
    dst[:] = src
  elseif req == :add
    dst[:] += src
  else
    error("Unable to handle $req in assign.")
  end
  return nothing
end

abstract CustomOpProp

function needs_top_grad(self :: CustomOpProp)
  return false
end

function infer_shape(self :: CustomOpProp, in_shape)
  return in_shape, [in_shape[1]], []
end

function list_outputs(self :: CustomOpProp)
  return String["output"]
end

function list_arguments(self :: CustomOpProp)
  return String["data"]
end

function list_auxiliary_states(self :: CustomOpProp)
  return String[]
end

function declare_backward_dependency(self :: CustomOpProp, out_grad, in_data, out_data)
  deps = Int[]
  if needs_top_grad(self)
    append!(deps, out_grad)
  end
  append!(deps, in_data)
  append!(deps, out_data)
end

include("_impl_custom.jl")
end
