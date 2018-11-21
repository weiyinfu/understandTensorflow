import func
import graph
import operations as op
import optimizer

const = op.const
log = op.log
exp = op.exp
array2tensor = op.array2tensor
add = op.add
matmul = op.matmul
equal = op.equal
reshape = op.reshape
Tensor = op.Tensor
abs = op.abs
placeholder = op.placeholder
variable = op.variable
argmax = op.Argmax

sigmoid = func.sigmoid
cross_entropy = func.cross_entropy
reduce_mean = func.reduce_mean
reduce_sum = func.reduce_sum
softmax = func.softmax
sparse_cross_entropy = func.sparse_cross_entropy

get_value = graph.get_value
Graph = graph.Graph
control_dependency = graph.control_dependency

Optimizer = optimizer.Optimizer
