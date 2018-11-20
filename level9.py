import level0
import level1
import level2
import level3
import level4

const = level0.get_const

log = level1.log
exp = level1.exp
array2tensor = level1.array2tensor
add = level1.add
matmul = level1.matmul
equal = level1.equal
reshape = level1.reshape
Tensor = level1.Tensor
abs = level1.abs

sigmoid = level2.sigmoid
cross_entropy = level2.cross_entropy
reduce_mean = level2.reduce_mean
reduce_sum = level2.reduce_sum
softmax = level2.softmax

get_value = level3.get_value
Graph = level3.Graph
control_dependency = level3.control_dependency

sparse_cross_entropy = level4.sparse_cross_entropy
placeholder = level4.placeholder
variable = level4.variable
argmax = level4.Argmax
Optimizer = level4.Optimizer
