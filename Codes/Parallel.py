import multiprocessing as mp
import numpy as np
from Polytran_and_resample import Grid_aff_interpolation
from Basefunction import bilinear_interpolation
from tqdm import tqdm


class parallel_struct:
    '''
    设计并行输入的数据结构，包括格网、块、影像
    '''
    def __init__(self, grids=None,chunk=None, data=None):
        self.grids = grids
        self.chunk = chunk
        self.data = data


def Initial_parallel_struct(grids, chunk, data):
    '''
    初始化数据结果内容
    :param grids: 格网，包含新旧影像一个格网的四个点坐标、仿射变换参数
    :param chunk: 用来存储影像块的地址
    :param data:  整幅影像
    :return:
    '''
    struct = []
    for i in range(len(grids)):
        ts = parallel_struct(grids[i], chunk[i], data)
        struct.append(ts)
    return struct


def Initial_chunk(grids):
    '''
    初始化影像块，确定分块个数
    :param grids: 输入的格网
    :return: 块列表
    '''
    chunk = []
    for i in range(len(grids)):
        chunk.append(0)
    return chunk

def extra_chunk(structs):

    chunk = []
    for i in range(len(structs)):
        chunk.append(structs[i].chunk)
    return chunk


def cal_single_chunk(struct):
    '''
    计算单个格网的数值
    :param struct:
    :return:
    '''
    y_dis = abs(struct.grids.grid_new.point3.y - struct.grids.grid_new.point1.y)
    x_dis = abs(struct.grids.grid_new.point2.x - struct.grids.grid_new.point1.x)
    startx = struct.grids.grid_new.point1.x
    starty = struct.grids.grid_new.point1.y

    a1, a2, a3, b1, b2, b3 = struct.grids.cof[0:6]
    a1, a2, a3, b1, b2, b3 = float(a1), float(a2), float(a3), float(b1), float(b2), float(b3)

    new = np.zeros((int(y_dis) + 1, int(x_dis) + 1), dtype=np.int16)
    for i in range(int(y_dis) + 1):
        for j in range(int(x_dis) + 1):
            temppt = Grid_aff_interpolation(a1, a2, a3, b1, b2, b3, i + starty, j + startx)
            new[i][j] = bilinear_interpolation( temppt, struct.data )
    b = np.array(list(new))
    struct.chunk = b
    return b



def cal_chunks(grids , data):
    '''
    计算所有格网的数值
    :param grids:
    :param data:
    :return:
    '''
    chunk = Initial_chunk(grids)
    struct = Initial_parallel_struct(grids, chunk, data)

    # 分配三个进程池
    p = mp.Pool(4)
    m = p.map(cal_single_chunk, struct)
    # print(list((tqdm(p.map(cal_single_chunk, struct), total=1000, desc='监视进度'))))
    return m
    
