# from __future__ import division

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D
# import numpy as np


# def parse_cfg(cfg_file: str):
#     file = open(cfgfile, 'r')
#     lines = file.read().split('\n')
#     lines = [x for x in lines if len(x) > 0]
#     lines = [x for x in lines if x[0] != '#']
#     lines = [x.strip() for x in lines]

#     block = {}
#     blocks = []

#     for line in lines:
#         if line[0] == '[':
#             if len(block) != 0:
#                 blocks.append(block)
#                 block = {}
#             block['type'] = line[1:-1].strip()
#         else:
#             key, value = line.split('=')
#             block[key.strip()] = value.strip()

#     blocks.append(block)

#     return blocks


# def create_module(blocks):
#     net_info = blocks[0]
#     layers = []
#     prev_filters = 3  # Image has 3 filters initially corresponding to RGB channels
#     output_filters = []

#     for index, block in enumerate(blocks[1:]):
#         if block['type'] == 'convolutional':
#             activation = x['activation']
#             try:
#                 batch_normalize = int(x['batch_normalize'])
#                 bias = False
#             except:
#                 batch_normalize = 0
#                 bias = True

#             filters = int(x['filters'])
#             padding = int(x['pad'])
#             kernel_size = int(x['size'])
#             stride = int(x['stride'])

#             if padding:
#                 pad = (kernel_size - 1) // 2
#             else:
#                 pad = 0

#             if activation == 'leaky':
#                 activation = LeakyReLU()

#             conv = Conv2D(filters, kernel_size, (stride, stride),
#                           activation=activation)
#             layers.append(conv)

#             if batch_normalize:
#                 layers.append(BatchNormalization())

#         elif block['type'] == 'upsample':
#             stride = block['stride']
#             upsample = UpSampling2D(size=(stride, stride))
#             layers.append(upsample)

#         elif block['type'] == 'route':
#             pass
#         elif block['type'] == 'shortcut':
#             pass
#         elif block['type'] == 'yolo':
#             pass
#         else:
#             raise Exception(f'Unknown block type: {block["type"]}')
