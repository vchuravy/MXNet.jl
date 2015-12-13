module Native

abstract Operator

function forward(:: Operator, in_data, out_data)
  out_data[0][:] = in_data[0]
end

function backward(:: Operator, out_grad, in_data, out_data, in_grad)
  in_grad[0][:] = 1.0
end

function infer_shape(:: Operator, in_shape)
   return in_shape, [in_shape[0]]
end

function list_outputs(:: Operator)
  return ["output"]
end

function list_arguments(:: Operator)
  return ["data"]
end

need_top_grad(:: Operator) = true

###
# NativeOpInfo mirrors the struct in include/mxnet/c_api.h and consists of five function
# pointers that work as callbacks. Each p_... ia a opaque pointer that contains the
# necessary information to call the right function.
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

  function NativeOpInfo(op :: Operator, forward, backward)
    p_is, p_loa, p_la = pointer_from_objref(op)

    c_wrapper_fb = cfunction(_wrapper_fb, Void, (Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Cint}, Ptr{Void}))
    c_wrapper_infer = cfunction(_wrapper_infer, Void, (Cint, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Void}))
    const c_wrapper_list = cfunction(_wrapper_list, Void, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))

    cond_forward = Condition()
    cond_backward = Condition()
    cb_f = Base.SingleAsyncWork(data -> notify(cond_forward))
    cb_b = Base.SingleAsyncWork(data -> notify(cond_backward))

    r_forward = Ref(_FB(cb_f.handle))
    r_backward = Ref(_FB(cb_f.handle))

    p_f  = convert(Ptr{Void}, r_forward)
    p_f  = convert(Ptr{Void}, r_backward)

    @schedule begin
      try
        while true
           wait(cond_forward)
           cond_forward = Condition()
           _entry_forward(r_forward[])
        end
      catch
        rethrow()
      finally
        Base.close(cb_f)
      end
    end

    @schedule begin
      try
        while true
           wait(cond_backward)
           cond_backward = Condition()
           _entry_backward(r_backward[])
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


create_info() = NativeOpInfo(fb_entry, fb_entry, infer_entry, list_entry, list_entry)
# pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
# mx._Native(name = :test, info = pstring)
end
