#=doc
Native operators in Julia
=========================
=#
module Native

#=doc
.. class:: Operator

   Abstract type user have to derive from to implement Operators in Julia
=#
abstract Operator

#=doc
.. function:: forward(op :: Operator, in_data, out_data)
=#
function forward(:: Operator, in_data, out_data)
  out_data[0][:] = in_data[0]
end

#=doc
.. function:: backward(op :: Operator, out_grad, in_data, out_data, in_grad)
=#
function backward(:: Operator, out_grad, in_data, out_data, in_grad)
  in_grad[0][:] = 1.0
end

#=doc
.. function:: infer_shape(op :: Operator, in_shape)
=#
function infer_shape(:: Operator, in_shape)
   return in_shape, [in_shape[0]]
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

###
# NativeOpInfo mirrors the struct in include/mxnet/c_api.h and consists of five function
# pointers that work as callbacks. Each p_... ia a opaque pointer that contains the
# necessary information to call the right function.
#
# Forward and backward functions call require more care because they can happen async, we
# have to handle them in the task that wait for the functions to be called.
#
# Todo: Cleanup tasks.
###
immutable NativeOpInfo
  forward :: Ptr{Void}
  backward :: Ptr{Void}
  infer_shape :: Ptr{Void}
  list_outputs :: Ptr{Void}
  list_arguments :: Ptr{Void}
  p_forward :: Ptr{Void}
  p_backward :: Ptr{Void}
  p_infer_shape :: Ptr{Void}
  p_list_outputs :: Ptr{Void}
  p_list_arguments :: Ptr{Void}

  function NativeOpInfo(op :: Operator)
    # infer_shape, list_args, list_outputs are called directly and use dynamic dispatch,
    # for finding the correct operator.
    p_is, p_la, p_la = pointer_from_objref(op)

    c_wrapper_fb = cfunction(_wrapper_fb, Void, (Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Cint}, Ptr{Void}))
    c_wrapper_infer = cfunction(_wrapper_infer, Void, (Cint, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Void}))
    c_wrapper_list_outputs = cfunction(_wrapper_list_outputs, Void, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))
    c_wrapper_list_arguments = cfunction(_wrapper_list_arguments, Void, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))

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

    p_f = convert(Ptr{Void}, r_forward)
    p_f = convert(Ptr{Void}, r_backward)

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

    new(c_wrapper_fb, c_wrapper_fb, c_wrapper_infer, c_wrapper_list,
        c_wrapper_list, p_f, p_b, p_is, p_lo, p_la)
  end
end

###
# Infer and list are called in sync.
###
function _wrapper_infer(size :: Cint, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, _op :: Ptr{Void})
  op = unsafe_pointer_to_objref(_op) :: Operator

  n_in = length(list_arguments(op))
  n_out = length(list_outputs(op))
  @assert size == n_in + n_out

  shapes = [[tensor_shapes[i][j] for j in 1:tensor_dims[i]] for i in 1:n_in]]

  ishape, oshape = infer_shape(op, shapes)
  @assert length(ishape) == n_in
  @assert length(oshape) == n_out

  rshape = cat(ishape, oshape)
  unsafe_store!(shapes, rshapes)
  return nothing
end

function _wrapper_list_arguments(data :: Ptr{Ptr{Cstring}}, _op :: Ptr{Void})
  op = unsafe_pointer_to_objref(_op) :: Operator
  arguments = list_arguments(op)
  unsafe_store!(data, arguments)
  return nothing
end

function _wrapper_list_outputs(data :: Ptr{Ptr{Cstring}}, _op :: Ptr{Void})
  op = unsafe_pointer_to_objref(_op) :: Operator
  outputs = list_outputs(op)
  unsafe_store!(data, outputs)
  return nothing
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
  data :: Ptr{Ptr{Cfloat}}
  ndims :: Ptr{Cint}
  shapes :: Ptr{Ptr{Cuint}}
  tags :: Ptr{Cint}
end
_FB(handle :: Ptr{Void}) = _FP(handle, 0, 0, 0, 0, 0)
@assert isbits(_FB)

# This function is called async and because the Julia runtime is not thread safe, we are
# very limited in the things we can do. Using a immutable that is a bitstype we can pass,
# return values to the handling tasks.
function _wrapper_fb(size :: Cint, data :: Ptr{Ptr{Cfloat}}, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, tags :: Ptr{Cint}, payload :: Ptr{Void})
  # Load the libuv async handle
  ptr = convert(Ptr{_FB}, payload)
  handle = unsafe_load(ptr, 1).handle

  # Create result
  val = _FB(handle, size, data, ndims, shapes, tags)
  unsafe_store!(ptr, val, 1)

  ccall(:uv_async_send, Void, (Ptr{Void},), handle)
  nothing
end

# Todo: handle the c callback and call the correct function
function _entry_forward(:: Operator, payload :: _FB)
end

# Todo: handle the c callback and call the correct function
function _entry_backward(:: Operator, payload :: _FB)
end

# pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
# mx._Native(name = :test, info = pstring)
end
