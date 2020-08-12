from models.yolo.blocks import conv_block, res_block


def darknet53(inputs):
    x = conv_block(inputs, filters=32, kernel_size=3)
    x = _res_block_body(x, 64, 1)
    x = _res_block_body(x, 128, 2)
    x = _res_block_body(x, 256, 8)
    route_1 = x
    x = _res_block_body(x, 512, 8)
    route_2 = x
    x = _res_block_body(x, 1024, 4)
    return route_1, route_2, x


def _res_block_body(inputs, filters, res_blocks_count):
    x = conv_block(inputs, filters=filters, kernel_size=3, down_sample=True)
    for i in range(res_blocks_count):
        x = res_block(x, filters // 2, filters)
    return x
