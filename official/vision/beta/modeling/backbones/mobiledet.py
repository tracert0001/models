# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains definitions of MobileNet Networks."""

from typing import Optional, Dict, Any, Tuple

# Import libraries
import dataclasses
import tensorflow as tf
from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers
from official.vision.beta.modeling.backbones import mobilenet

layers = tf.keras.layers


#  pylint: disable=pointless-string-statement

"""
Architecture: https://arxiv.org/abs/1704.04861.

"MobileDets: Searching for Object Detection Architectures for 
Mobile Accelerators" Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, 
Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, 
Bo Chen
"""

MD_CPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetCPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_residual', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, None, False),
        # _inverted_bottleneck_no_expansion
        ('invertedbottleneck', 3, 1, 8, 'hard_swish', 0.25, 1., False, True),
        ('invertedbottleneck', 3, 2, 16, 'hard_swish', 0.25, 4., False, True),
        ('invertedbottleneck', 3, 2, 32, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, True),
        ('invertedbottleneck', 5, 2, 72, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 5, 1, 72, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, True),
        ('invertedbottleneck', 5, 2, 104, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 5, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 5, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 144, 'hard_swish', 0.25, 8., False, True),
    ]
}

MD_DSP_BLOCK_SPECS = {
    'spec_name': 'MobileDetDSP',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias', 'is_output'],
    'block_specs': []
}

MD_EdgeTPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetEdgeTPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias', 'is_output'],
    'block_specs': []
}

MD_GPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetGPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias', 'is_output'],
    'block_specs': []
}

