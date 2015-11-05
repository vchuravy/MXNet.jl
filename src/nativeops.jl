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

###
# Each _wrapper_... is the entry point for the c function call that converts the opaque
# pointer into the right julia function.
###

function _wrapper_fb(size :: Cint, data :: Ptr{Ptr{Cfloat}}, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, tags :: Ptr{Cint}, jf :: Ptr{Void})
  julia_function = unsafe_pointer_to_objref(jf) :: Function
  julia_function(Int(size), data, ndims, shapes, tags)
  return nothing
end

function _wrapper_infer(size :: Cint, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, jf :: Ptr{Void})
  julia_function = unsafe_pointer_to_objref(jf) :: Function
  julia_function(Int(size), ndims, shapes)
  return nothing
end

function _wrapper_list(data :: Ptr{Ptr{Ptr{Cchar}}}, jf :: Ptr{Void})
  julia_function = unsafe_pointer_to_objref(jf) :: Function
  julia_function(data)
  return nothing
end

###
# Test entry functions
###

function list_entry(a :: Ptr{Ptr{Ptr{Cchar}}})
  data = ByteString[""]
  ref = Base.cconvert(Ptr{Ptr{Cchar}}, data)
  ptr = Base.unsafe_convert(Ptr{Ptr{Cchar}}, ref)
  unsafe_store!(a, ptr,1)
end

function fb_entry(num_tensor :: Cint, in :: Ptr{Ptr{Cfloat}}, x :: Ptr{Cint}, y :: Ptr{Cuint}, z :: Ptr{Cint})
end

function infer_entry(num_tensor :: Cint, tensor_dims :: Ptr{Cint}, tensor_shapes :: Ptr{Ptr{Cuint}})
end

create_info() = NativeOpInfo(fb_entry, fb_entry, infer_entry, list_entry, list_entry)
# pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
# mx._Native(name = :test, info = pstring)
