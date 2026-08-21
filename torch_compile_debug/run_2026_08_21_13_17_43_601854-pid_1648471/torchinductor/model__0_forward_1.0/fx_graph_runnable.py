
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/tensorplay-torchinductor-debug.nBJfto'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.13.0+cu130
# torch cuda version: 13.0
# torch git version: cf30153c4c131c8164ee7798e5022d810682e2cb


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103):
        convolution = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add = torch.ops.aten.add.Tensor(primals_4, 1e-05)
        sqrt = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_3, -1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        relu = torch.ops.aten.relu.default(add_1);  add_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem = _low_memory_max_pool_with_offsets[0]
        getitem_1 = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        convolution_1 = torch.ops.aten.convolution.default(getitem, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_2 = torch.ops.aten.add.Tensor(primals_9, 1e-05)
        sqrt_1 = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(primals_10, -1)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(primals_11, -1);  primals_11 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        relu_1 = torch.ops.aten.relu.default(add_3);  add_3 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, primals_12, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_4 = torch.ops.aten.add.Tensor(primals_14, 1e-05)
        sqrt_2 = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(primals_15, -1)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5 = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        add_6 = torch.ops.aten.add.Tensor(add_5, getitem);  add_5 = None
        relu_2 = torch.ops.aten.relu.default(add_6);  add_6 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, primals_17, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_7 = torch.ops.aten.add.Tensor(primals_19, 1e-05)
        sqrt_3 = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(primals_20, -1)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3 = torch.ops.aten.relu.default(add_8);  add_8 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_9 = torch.ops.aten.add.Tensor(primals_24, 1e-05)
        sqrt_4 = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(primals_25, -1)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_10 = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        add_11 = torch.ops.aten.add.Tensor(add_10, relu_2);  add_10 = None
        relu_4 = torch.ops.aten.relu.default(add_11);  add_11 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, primals_27, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_12 = torch.ops.aten.add.Tensor(primals_29, 1e-05)
        sqrt_5 = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_13 = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        relu_5 = torch.ops.aten.relu.default(add_13);  add_13 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, primals_32, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_14 = torch.ops.aten.add.Tensor(primals_34, 1e-05)
        sqrt_6 = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(primals_33, -1)
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
        mul_19 = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_4, primals_37, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_16 = torch.ops.aten.add.Tensor(primals_39, 1e-05)
        sqrt_7 = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(primals_38, -1)
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(primals_40, -1)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(primals_41, -1);  primals_41 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_17 = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        add_18 = torch.ops.aten.add.Tensor(add_15, add_17);  add_15 = add_17 = None
        relu_6 = torch.ops.aten.relu.default(add_18);  add_18 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, primals_42, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_19 = torch.ops.aten.add.Tensor(primals_44, 1e-05)
        sqrt_8 = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(primals_43, -1)
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(primals_45, -1)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_20 = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        relu_7 = torch.ops.aten.relu.default(add_20);  add_20 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, primals_47, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_21 = torch.ops.aten.add.Tensor(primals_49, 1e-05)
        sqrt_9 = torch.ops.aten.sqrt.default(add_21);  add_21 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_22 = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        add_23 = torch.ops.aten.add.Tensor(add_22, relu_6);  add_22 = None
        relu_8 = torch.ops.aten.relu.default(add_23);  add_23 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, primals_52, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_24 = torch.ops.aten.add.Tensor(primals_54, 1e-05)
        sqrt_10 = torch.ops.aten.sqrt.default(add_24);  add_24 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(primals_53, -1)
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_25 = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        relu_9 = torch.ops.aten.relu.default(add_25);  add_25 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, primals_57, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_26 = torch.ops.aten.add.Tensor(primals_59, 1e-05)
        sqrt_11 = torch.ops.aten.sqrt.default(add_26);  add_26 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(primals_58, -1)
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_27 = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_8, primals_62, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_28 = torch.ops.aten.add.Tensor(primals_64, 1e-05)
        sqrt_12 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(primals_63, -1)
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_29 = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        add_30 = torch.ops.aten.add.Tensor(add_27, add_29);  add_27 = add_29 = None
        relu_10 = torch.ops.aten.relu.default(add_30);  add_30 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_10, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_31 = torch.ops.aten.add.Tensor(primals_69, 1e-05)
        sqrt_13 = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(primals_70, -1)
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_32 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
        relu_11 = torch.ops.aten.relu.default(add_32);  add_32 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_11, primals_72, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_33 = torch.ops.aten.add.Tensor(primals_74, 1e-05)
        sqrt_14 = torch.ops.aten.sqrt.default(add_33);  add_33 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(primals_75, -1)
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_34 = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
        add_35 = torch.ops.aten.add.Tensor(add_34, relu_10);  add_34 = None
        relu_12 = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, primals_77, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_36 = torch.ops.aten.add.Tensor(primals_79, 1e-05)
        sqrt_15 = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(primals_80, -1)
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_37 = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
        relu_13 = torch.ops.aten.relu.default(add_37);  add_37 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, primals_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_38 = torch.ops.aten.add.Tensor(primals_84, 1e-05)
        sqrt_16 = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48 = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_39 = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_12, primals_87, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_40 = torch.ops.aten.add.Tensor(primals_89, 1e-05)
        sqrt_17 = torch.ops.aten.sqrt.default(add_40);  add_40 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_51 = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(primals_88, -1)
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        sub_17 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_41 = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
        add_42 = torch.ops.aten.add.Tensor(add_39, add_41);  add_39 = add_41 = None
        relu_14 = torch.ops.aten.relu.default(add_42);  add_42 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_14, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_43 = torch.ops.aten.add.Tensor(primals_94, 1e-05)
        sqrt_18 = torch.ops.aten.sqrt.default(add_43);  add_43 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_54 = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(primals_93, -1)
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        sub_18 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_44 = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
        relu_15 = torch.ops.aten.relu.default(add_44);  add_44 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_15, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_45 = torch.ops.aten.add.Tensor(primals_99, 1e-05)
        sqrt_19 = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_57 = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(primals_100, -1)
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_46 = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
        add_47 = torch.ops.aten.add.Tensor(add_46, relu_14);  add_46 = None
        relu_16 = torch.ops.aten.relu.default(add_47);  add_47 = None
        mean = torch.ops.aten.mean.dim(relu_16, [-1, -2], True)
        view = torch.ops.aten.view.default(mean, [2, 512]);  mean = None
        permute = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_103, view, permute);  primals_103 = permute = None
        le = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        return (addmm, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, convolution, getitem, getitem_1, convolution_1, relu_1, convolution_2, relu_2, convolution_3, relu_3, convolution_4, relu_4, convolution_5, relu_5, convolution_6, convolution_7, relu_6, convolution_8, relu_7, convolution_9, relu_8, convolution_10, relu_9, convolution_11, convolution_12, relu_10, convolution_13, relu_11, convolution_14, relu_12, convolution_15, relu_13, convolution_16, convolution_17, relu_14, convolution_18, relu_15, convolution_19, view, le)

def load_args(reader):
    buf0 = reader.storage(None, 37632)
    reader.tensor(buf0, (64, 3, 7, 7), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 24576)
    reader.tensor(buf1, (2, 3, 32, 32), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 256)
    reader.tensor(buf2, (64,), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256)
    reader.tensor(buf3, (64,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 256)
    reader.tensor(buf4, (64,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 256)
    reader.tensor(buf5, (64,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 147456)
    reader.tensor(buf6, (64, 64, 3, 3), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 256)
    reader.tensor(buf7, (64,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 256)
    reader.tensor(buf8, (64,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 256)
    reader.tensor(buf9, (64,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 256)
    reader.tensor(buf10, (64,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 147456)
    reader.tensor(buf11, (64, 64, 3, 3), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 256)
    reader.tensor(buf12, (64,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 256)
    reader.tensor(buf13, (64,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 256)
    reader.tensor(buf14, (64,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 256)
    reader.tensor(buf15, (64,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 147456)
    reader.tensor(buf16, (64, 64, 3, 3), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 256)
    reader.tensor(buf17, (64,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 256)
    reader.tensor(buf18, (64,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 256)
    reader.tensor(buf19, (64,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 256)
    reader.tensor(buf20, (64,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 147456)
    reader.tensor(buf21, (64, 64, 3, 3), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 256)
    reader.tensor(buf22, (64,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 256)
    reader.tensor(buf23, (64,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 256)
    reader.tensor(buf24, (64,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 256)
    reader.tensor(buf25, (64,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 294912)
    reader.tensor(buf26, (128, 64, 3, 3), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 512)
    reader.tensor(buf27, (128,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 512)
    reader.tensor(buf28, (128,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 512)
    reader.tensor(buf29, (128,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 512)
    reader.tensor(buf30, (128,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 589824)
    reader.tensor(buf31, (128, 128, 3, 3), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 512)
    reader.tensor(buf32, (128,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 512)
    reader.tensor(buf33, (128,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 512)
    reader.tensor(buf34, (128,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 512)
    reader.tensor(buf35, (128,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 32768)
    reader.tensor(buf36, (128, 64, 1, 1), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 512)
    reader.tensor(buf37, (128,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 512)
    reader.tensor(buf38, (128,), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 512)
    reader.tensor(buf39, (128,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 512)
    reader.tensor(buf40, (128,), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 589824)
    reader.tensor(buf41, (128, 128, 3, 3), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 512)
    reader.tensor(buf42, (128,), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 512)
    reader.tensor(buf43, (128,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 512)
    reader.tensor(buf44, (128,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 512)
    reader.tensor(buf45, (128,), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 589824)
    reader.tensor(buf46, (128, 128, 3, 3), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 512)
    reader.tensor(buf47, (128,), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 512)
    reader.tensor(buf48, (128,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 512)
    reader.tensor(buf49, (128,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 512)
    reader.tensor(buf50, (128,), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 1179648)
    reader.tensor(buf51, (256, 128, 3, 3), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 1024)
    reader.tensor(buf52, (256,), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 1024)
    reader.tensor(buf53, (256,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 1024)
    reader.tensor(buf54, (256,), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 1024)
    reader.tensor(buf55, (256,), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 2359296)
    reader.tensor(buf56, (256, 256, 3, 3), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 1024)
    reader.tensor(buf57, (256,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 1024)
    reader.tensor(buf58, (256,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 1024)
    reader.tensor(buf59, (256,), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 1024)
    reader.tensor(buf60, (256,), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 131072)
    reader.tensor(buf61, (256, 128, 1, 1), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 1024)
    reader.tensor(buf62, (256,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 1024)
    reader.tensor(buf63, (256,), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 1024)
    reader.tensor(buf64, (256,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 1024)
    reader.tensor(buf65, (256,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 2359296)
    reader.tensor(buf66, (256, 256, 3, 3), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 1024)
    reader.tensor(buf67, (256,), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 1024)
    reader.tensor(buf68, (256,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 1024)
    reader.tensor(buf69, (256,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1024)
    reader.tensor(buf70, (256,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 2359296)
    reader.tensor(buf71, (256, 256, 3, 3), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 1024)
    reader.tensor(buf72, (256,), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 1024)
    reader.tensor(buf73, (256,), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 1024)
    reader.tensor(buf74, (256,), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 1024)
    reader.tensor(buf75, (256,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 4718592)
    reader.tensor(buf76, (512, 256, 3, 3), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 2048)
    reader.tensor(buf77, (512,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 2048)
    reader.tensor(buf78, (512,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 2048)
    reader.tensor(buf79, (512,), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 2048)
    reader.tensor(buf80, (512,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 9437184)
    reader.tensor(buf81, (512, 512, 3, 3), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 2048)
    reader.tensor(buf82, (512,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 2048)
    reader.tensor(buf83, (512,), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 2048)
    reader.tensor(buf84, (512,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 2048)
    reader.tensor(buf85, (512,), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 524288)
    reader.tensor(buf86, (512, 256, 1, 1), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 2048)
    reader.tensor(buf87, (512,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 2048)
    reader.tensor(buf88, (512,), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 2048)
    reader.tensor(buf89, (512,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 2048)
    reader.tensor(buf90, (512,), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 9437184)
    reader.tensor(buf91, (512, 512, 3, 3), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 2048)
    reader.tensor(buf92, (512,), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 2048)
    reader.tensor(buf93, (512,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 2048)
    reader.tensor(buf94, (512,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 2048)
    reader.tensor(buf95, (512,), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 9437184)
    reader.tensor(buf96, (512, 512, 3, 3), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 2048)
    reader.tensor(buf97, (512,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 2048)
    reader.tensor(buf98, (512,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 2048)
    reader.tensor(buf99, (512,), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 2048)
    reader.tensor(buf100, (512,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 6144)
    reader.tensor(buf101, (3, 512), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 12)
    reader.tensor(buf102, (3,), is_leaf=True)  # primals_103
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)