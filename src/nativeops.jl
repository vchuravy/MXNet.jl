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
# Each _wrapper_... is the entry point for the c function call.
# It receives an opaque pointer to the correct entry function and
# the function parameters. It then wraps the function parameters in the
# correct immutable T <: _MXNET_DATA and creates an _Async object from it.
# _Async reimplements and extends Base.SingleAsyncWork to also pass some data
# and a Condition along. That condition is then used to synchronize
# the _wrapper and _entry functions. If we wouldn't synchronize them the
# _wrapper functions would return before the _entry functions have been run.
# Returning illegal state to MXNet.
###
abstract _MXNET_DATA

# Based on Base.SingleAsyncWork
immutable _Async{T <: _MXNET_DATA}
  data :: T

  handle :: Ptr{Void}
  cb :: Function
  cond :: Condition

  function _Async(data :: T, cb::Function)
    this = new(data, Libc.malloc(Base._sizeof_uv_async), cb, Condition())
    Base.associate_julia_struct(this.handle, this)
    Base.preserve_handle(this)

    async_cb = cfunction(cb, Void, (Ptr{Void},))

    err = ccall(:uv_async_init,Cint,(Ptr{Void},Ptr{Void},Ptr{Void}),Base.eventloop(),this.handle,async_cb::Ptr{Void})

    this
  end
end

Base._uv_hook_close(t::_Async) = (uv.handle = C_NULL; unpreserve_handle(uv); nothing)

immutable _FB <: _MXNET_DATA
  size :: Cint
  data :: Ptr{Ptr{Cfloat}}
  ndims :: Ptr{Cint}
  shapes :: Ptr{Ptr{Cuint}}
  tags :: Ptr{Cint}
end

function _wrapper_fb(size :: Cint, data :: Ptr{Ptr{Cfloat}}, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, tags :: Ptr{Cint}, jf :: Ptr{Void})
  entry = unsafe_pointer_to_objref(jf) :: Function
  cb_data = _FB(size, data, ndims, shapes, tags)
  work = _Async{_FB}(cb_data, entry) # We have to specify T here otherwise inference will run and seqfault.
  ccall(:uv_async_send, Void, (Ptr{Void},), work.handle)
  wait(work.cond)
  return nothing
end

immutable _INFER <: _MXNET_DATA
  size :: Cint
  ndims :: Ptr{Cint}
  shapes :: Ptr{Ptr{Cuint}}
end

function _wrapper_infer(size :: Cint, ndims :: Ptr{Cint}, shapes :: Ptr{Ptr{Cuint}}, jf :: Ptr{Void})
  entry = unsafe_pointer_to_objref(jf) :: Function
  cb_data = _INFER(size, ndims, shapes)
  work = _Async{_INFER}(cb_data, entry)
  ccall(:uv_async_send, Void, (Ptr{Void},), work.handle)
  wait(work.cond)
  return nothing
end

immutable _LIST <: _MXNET_DATA
  result :: Ptr{Ptr{Ptr{Cchar}}}
end

function _wrapper_list(data :: Ptr{Ptr{Ptr{Cchar}}}, jf :: Ptr{Void})
  entry = unsafe_pointer_to_objref(jf) :: Function
  cb_data = _LIST(data)
  work = _Async{_LIST}(cb_data, entry)
  ccall(:uv_async_send, Void, (Ptr{Void},), work.handle)
  wait(work.cond)
  return nothing
end

###
# Entry functions
# These functions are now executed in the main Julia thread.
###

function list_entry(handle :: Ptr{Void})
  work = Base.@handle_as handle _Async{_LIST}
  try
    data = ByteString[""]
    ref = Base.cconvert(Ptr{Ptr{Cchar}}, data)
    ptr = Base.unsafe_convert(Ptr{Ptr{Cchar}}, ref)
    unsafe_store!(work.data.result, ptr,1)
  catch
  finally
    notify(work.cond)
  end
  nothing
end

function fb_entry(handle :: Ptr{Void})
  work = Base.@handle_as handle _Async{_FB}
  try
    data = work.data
    # do data conversion
    # call appropriate julia function
    # store result
  catch
  finally
    notify(work.cond)
  end
  nothing
end

function infer_entry(handle :: Ptr{Void})
  work = Base.@handle_as handle _Async{_INFER}
  try
    data = work.data
    # do data conversion
    # call appropriate julia function
    # store result
  catch
  finally
    notify(work.cond)
  end
  nothing
end

create_info() = NativeOpInfo(fb_entry, fb_entry, infer_entry, list_entry, list_entry)
# pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
# mx._Native(name = :test, info = pstring)
