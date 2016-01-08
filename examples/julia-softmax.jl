using MXNet

import MXNet.mx.Native: Operator, list_arguments, list_outputs, infer_shape,
                        forward, backward, need_top_grad
immutable JuliaSoftmax <: Operator end

list_arguments(:: JuliaSoftmax) = ["data", "label"]
list_outputs(:: JuliaSoftmax) = ["output"]
need_top_grad(:: JuliaSoftmax) = false

function infer_shape(::JuliaSoftmax, in_shape)
  data_shape = in_shape[1]
  label_shape = (in_shape[1][1],)
  output_shape = in_shape[1]
  return [data_shape, label_shape], [output_shape]
end

function forward(::JuliaSoftmax, in_data, out_data)
  x = in_data[1]
  y = out_data[1]

  @mx.nd_as_jl ro=x rw=y begin
    y[:] = exp(x - maximum(x, 1))
    y /= sum(y, 1)
  end
end

#TODO: Correct gradient
function backward(::JuliaSoftmax, out_grad, in_data, out_data, in_grad)
  label = in_data[2]
  y = out_data[1]
  dx = in_grad[1]

  @mx.nd_as_jl ro=(label, y) rw=dx begin
    dx[:] = y
  end
end

#define mlp
data = mx.Variable("data")
fc1 = mx.FullyConnected(data = data, name="fc1", num_hidden=128)
act1 = mx.Activation(data = fc1, name="relu1", act_type="relu")
fc2 = mx.FullyConnected(data = act1, name="fc2", num_hidden=64)
act2 = mx.Activation(data = fc2, name="relu2", act_type="relu")
fc3 = mx.FullyConnected(data = act2, name="fc3", num_hidden=10)

# Setup Native operator
mysoftmax = JuliaSoftmax()
info = mx.Native.NDArrayOpInfo(mysoftmax)
pstring = bytestring("0x", hex(reinterpret(UInt, pointer_from_objref(info))))
mlp = mx._Native(name = "softmax", info = pstring, data=fc3)

model = mx.FeedForward(mlp, context = mx.cpu())
optimizer = mx.SGD(lr = 0.1, momentum = 0.9, weight_decay = 0.00001)

include("mnist/mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(100)
mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch =20 )

