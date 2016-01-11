#=doc
Native operators in Julia
=========================
=#
module Native
import ..mx: NDArray

#=doc
.. class:: Operator

   Abstract type user have to derive from to implement Operators in Julia
=#
abstract Operator

#=doc
.. function:: forward(op :: Operator, in_data :: Vector{NDArray}, out_data :: Vector{NDArray})
=#
function forward(op :: Operator, in_data :: Vector{NDArray}, out_data :: Vector{NDArray})
  throw(MethodError(forward, (op, in_data, out_data)))
end

#=doc
.. function:: backward(op :: Operator, out_grad :: Vector{NDArray}, in_data :: Vector{NDArray}, out_data :: Vector{NDArray}, in_grad :: Vector{NDArray})
=#
function backward(op :: Operator, out_grad :: Vector{NDArray}, in_data :: Vector{NDArray}, out_data :: Vector{NDArray}, in_grad :: Vector{NDArray})
  throw(MethodError(backward, (op, out_grad, in_data, out_data, in_grad)))
end

#=doc
.. function:: infer_shape(op :: Operator, in_shapes :: Vector{Vector{UInt32}})

   Calculates the shapes of input and output.

   Returns two tuples of shapes. One for the inputs and one for the outputs.
   `return (data_shape, label_shape), (out_shape, )`. Shapes are stored as
   vectors of unsigned integers.

   :param in_shapes: Current shapes of inputs.
=#
function infer_shape(op :: Operator, in_shapes :: Vector{Vector{UInt32}})
  throw(MethodError(infer_shape, (op, in_shapes)))
end

#=doc
.. function:: list_outputs(op :: Operator)
=#
function list_outputs(:: Operator)
  return ["output"]
end

#=doc
.. function:: list_arguments(op :: Operator)
=#
function list_arguments(:: Operator)
  return ["data"]
end

#=doc
.. function:: need_top_grad(op :: Operator)
=#
need_top_grad(:: Operator) = true

#=doc
.. function:: declare_backward_dependency(op :: Operator, out_grad :: Vector{Int32}, in_data :: Vector{Int32}, out_data :: Vector{Int32})

  Declare dependencies of this operator for backward pass.

  Return value needs to be an integer array.
=#
function declare_backward_dependency(op :: Operator, out_grad :: Vector{Int32}, in_data :: Vector{Int32}, out_data :: Vector{Int32})
  deps = Int[]
  if need_top_grad(op)
    append!(deps, out_grad)
  end
  append!(deps, in_data)
  append!(deps, out_data)

  return deps
end

###
# NDArrayOpInfo mirrors the struct in include/mxnet/c_api.h and consists of five function
# pointers that work as callbacks. Each p_... ia a opaque pointer that contains the
# necessary information to call the right function.
#
# Forward and backward functions call require more care because they can happen async, we
# have to handle them in the task that wait for the functions to be called.
#
# Todo: Cleanup tasks.
###
immutable NDArrayOpInfo
  forward :: Ptr{Void}
  backward :: Ptr{Void}
  infer_shape :: Ptr{Void}
  list_outputs :: Ptr{Void}
  list_arguments :: Ptr{Void}
  declare_backward_dependecy :: Ptr{Void}

  p_forward :: Ptr{Void}
  p_backward :: Ptr{Void}
  p_infer_shape :: Ptr{Void}
  p_list_outputs :: Ptr{Void}
  p_list_arguments :: Ptr{Void}
  p_declare_backward_dependency :: Ptr{Void}

  function NDArrayOpInfo(op :: Operator)
    # infer_shape, list_args, list_outputs are called directly and use dynamic dispatch,
    # for finding the correct operator.
    p_is = p_lo = p_la = p_dbd = pointer_from_objref(op)

    c_wrapper_fb = cfunction(_wrapper_fb, Bool, (Cint, Ptr{Ptr{Void}}, Ptr{Cint}, Ptr{Void}))
    c_wrapper_infer = cfunction(_wrapper_infer, Bool, (Cint, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Void}))
    c_wrapper_list_outputs = cfunction(_wrapper_list_outputs, Bool, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_wrapper_list_arguments = cfunction(_wrapper_list_arguments, Bool, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_wrapper_declare_backward_dependency = cfunction(_wrapper_declare_backward_dependency, Bool, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Void}))

    # Setting up for handling backward/forward. Each function has a condition for a task
    # to wait on and a libuv callback that notifies that condition, to handle the call.
    cond_forward = Condition()
    cond_backward = Condition()
    cb_f = Base.SingleAsyncWork(data -> notify(cond_forward))
    cb_b = Base.SingleAsyncWork(data -> notify(cond_backward))

    # The return value of the callbacks is stored in _FB and the first element is the
    # libuv handle.
    r_forward = Ref(_FB(cb_f.handle))
    r_backward = Ref(_FB(cb_f.handle))

    p_f = Base.unsafe_convert(Ptr{Void}, r_forward)
    p_b = Base.unsafe_convert(Ptr{Void}, r_backward)

    # Task for handling forward
    @schedule begin
      try
        while true
           wait(cond_forward)
           cond_forward = Condition()
           _entry_forward(op, r_forward[])
        end
      catch
        rethrow()
      finally
        Base.close(cb_f)
      end
    end

    # Task for handling backward
    @schedule begin
      try
        while true
           wait(cond_backward)
           cond_backward = Condition()
           _entry_backward(op, r_backward[])
        end
      catch
        rethrow()
      finally
        Base.close(cb_f)
      end
    end

    new(c_wrapper_fb, c_wrapper_fb, c_wrapper_infer, c_wrapper_list_outputs,
        c_wrapper_list_arguments,  c_wrapper_declare_backward_dependency,
        p_f, p_b, p_is, p_lo, p_la, p_dbd)
  end
end

###
# Infer and list are called in sync.
###
function _wrapper_infer(size :: Cint, ndims :: Ptr{Cint}, tensor_shapes :: Ptr{Ptr{Cuint}}, _op :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(_op) :: Operator

    n_in = length(list_arguments(op))
    n_out = length(list_outputs(op))
    @assert size == n_in + n_out

    shapes = Vector{Cuint}[]
    for i in 1:n_in
      # Get size of array and create julia array.
      jj = unsafe_load(ndims, i)
      r_shapes = zeros(Cuint, jj)
      # Get pointer to array
      tshapes = unsafe_load(tensor_shapes, i)
      # Copy values
      for j in 1:jj
        r_shapes[j] = unsafe_load(tshapes, jj - (j-1)) # reverse Shapes.
      end
      push!(shapes, r_shapes)
    end

    ishape, oshape = infer_shape(op, shapes)
    @assert length(ishape) == n_in
    @assert length(oshape) == n_out

    rshapes = Vector{Cuint}[ishape..., oshape...]

    for i in 1:size
      reverse!(rshapes[i]) # reverse shapes back
      unsafe_store!(tensor_shapes, pointer(rshapes[i]), i)
      unsafe_store!(ndims, length(rshapes[i]), i)
    end
  catch error
    println(STDERR, "Error in infer_shape: ")
    showerror(STDERR, error)
    return false
  end
  return true
end

# TODO: Lifetime of julia objects (GC!)
function _wrapper_list_arguments(data :: Ptr{Ptr{Ptr{Cchar}}}, _op :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(_op) :: Operator
    arguments = list_arguments(op)
    ptrs = Ptr{Cchar}[Base.unsafe_convert(Ptr{Cchar}, s) for s in arguments]
    push!(ptrs, C_NULL)
    r_args = Ref(ptrs)
    unsafe_store!(data,  Base.unsafe_convert(Ptr{Ptr{Cchar}}, r_args), 1)
  catch error
    println(STDERR, "Error in list_arguments: ")
    showerror(STDERR, error)
    return false
  end
  return true
end

# TODO: Lifetime of julia objects (GC!)
function _wrapper_list_outputs(data :: Ptr{Ptr{Ptr{Cchar}}}, _op :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(_op) :: Operator
    outputs = list_outputs(op)
    ptrs = Ptr{Cchar}[Base.unsafe_convert(Ptr{Cchar}, s) for s in outputs]
    push!(ptrs, C_NULL)
    r_out = Ref(ptrs)
    unsafe_store!(data, Base.unsafe_convert(Ptr{Ptr{Cchar}}, r_out), 1)
  catch error
    println(STDERR, "Error in list_outputs: ")
    showerror(STDERR, error)
    return false
  end
  return true
end

# TODO: Lifetime of julia objects (GC!)
function _wrapper_declare_backward_dependency(_out_grad :: Ptr{Cint},
                                              _in_data  :: Ptr{Cint},
                                              _out_data :: Ptr{Cint},
                                              num_dep  :: Ptr{Cint},
                                              deps     :: Ptr{Ptr{Cint}},
                                              _op      :: Ptr{Void})
  try
    op = unsafe_pointer_to_objref(_op) :: Operator
    out_grad = pointer_to_array(_out_grad, length(list_outputs(op)), false) :: Vector{Int32}
    in_data = pointer_to_array(_in_data, length(list_arguments(op)), false) :: Vector{Int32}
    out_data = pointer_to_array(_out_data, length(list_outputs(op)), false) :: Vector{Int32}

    rdeps = declare_backward_dependency(op, out_grad, in_data, out_data) :: Vector{Int32}

    unsafe_store!(num_dep, length(rdeps), 1)
    r_rdeps = Ref(rdeps) # Lifetime?
    unsafe_store!(deps, Base.unsafe_convert(Ptr{Cint}, r_rdeps), 1)
  catch error
    println(STDERR, "Error in declare_backward_dependecy: ")
    showerror(STDERR, error)
    return false
  end
  return true
end

##
# Forward and backward can be called from different threads in libmxnet and
# so we need to take special care in handling these callbacks correctly in
# the julia runtime.
##

# Callback struct for forward-backward
immutable _FB
  handle :: Ptr{Void}
  size :: Cint
  data :: Ptr{Ptr{Void}}
  tags :: Ptr{Cint}
end
_FB(handle :: Ptr{Void}) = _FB(handle, 0, 0, 0)
@assert isbits(_FB)

# This function is called async and because the Julia runtime is not thread safe, we are
# very limited in the things we can do. Using a immutable that is a bitstype we can pass,
# return values to the handling tasks.
function _wrapper_fb(size :: Cint, data :: Ptr{Ptr{Void}}, tags :: Ptr{Cint}, payload :: Ptr{Void})
  # Load the libuv async handle
  ptr = convert(Ptr{_FB}, payload)
  handle = unsafe_load(ptr, 1).handle

  # Create result
  val = _FB(handle, size, data, tags)
  unsafe_store!(ptr, val, 1)

  ccall(:uv_async_send, Void, (Ptr{Void},), handle)
  return true # Better solution?
end

# Todo: handle the c callback and call the correct function
function _entry_forward(op :: Operator, payload :: _FB)
  num_ndarray = payload.size
  ndarraies = payload.data
  tags = payload.tags

  tensors = [NDArray[] for i in 1:4]

  # Tags are zero-based
  for i in 1:num_ndarray
    handle = mx.MX_NDArrayHandle(unsafe_load(ndarraies, i))
    tag = unsafe_load(tags, i)
    if tag == 1
      push!(tensors[tag + 1], NDArray(handle, true))
    else
      push!(tensors[tag + 1], NDArray(handle, false))
    end
  end
  forward(op, tensors[1], tensors[2])
end

# Todo: handle the c callback and call the correct function
function _entry_backward(op :: Operator, payload :: _FB)
  num_ndarray = payload.size
  ndarraies = payload.data
  tags = payload.tags

  tensors = [NDArray[] for i in 1:4]

  for i in 1:num_ndarray
    handle = mx.MX_NDArrayHandle(unsafe_load(ndarraies, i))
    tag = unsafe_load(tags, i)
    if tag == 2
      push!(tensors[tag + 1], NDArray(handle, true))
    else
      push!(tensors[tag + 1], NDArray(handle, false))
    end
  end
  backward(op, tensors[1], tensors[2], tensors[3], tensors[4])
end

end
