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

  function NativeOpInfo(forward :: Function, backwards :: Function, infer_shape :: Function, list_outputs :: Function, list_arguments :: Function)
    c_wrapper_fb = cfunction(_wrapper_fb, Void, (Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Cint}, Ptr{Void}))
    c_wrapper_infer = cfunction(_wrapper_infer, Void, (Cint, Ptr{Cint}, Ptr{Ptr{Cuint}}, Ptr{Void}))
    const c_wrapper_list = cfunction(_wrapper_list, Void, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Void}))

    p_f  = pointer_from_objref(forward)
    p_b  = pointer_from_objref(backwards)
    p_is = pointer_from_objref(infer_shape)
    p_lo = pointer_from_objref(list_outputs)
    p_la = pointer_from_objref(list_arguments)
    new(c_wrapper_fb, c_wrapper_fb, c_wrapper_infer, c_wrapper_list,
        c_wrapper_list, p_f, p_b, p_is, p_lo, p_la)
  end
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

###
# Infer and list are called in sync.
###
function _wrapper_infer(size :: Cint, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, jf :: Ptr{Void})
  entry = unsafe_pointer_to_objref(jf) :: Function
  entry(size, ndims, shapes)
  return nothing
end

function _wrapper_list(data :: Ptr{Ptr{Cstring}}, jf :: Ptr{Void})
  entry = unsafe_pointer_to_objref(jf) :: Function
  entry(data)
  return nothing
end

create_info() = NativeOpInfo(fb_entry, fb_entry, infer_entry, list_entry, list_entry)
# pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
# mx._Native(name = :test, info = pstring)
