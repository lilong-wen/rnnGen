from struct import unpack

class POT:
    '''POT 解码器'''
    def __init__(self, Z, set_name):
        self.Z = Z
        self._fp = Z.open(set_name)

    def __iter__(self):
        size = unpack('H', self._fp.read(2))[0]  # Sample size
        tag = {}  # 记录字符与笔画
        sizes = []
        tag_id = 0
        while size:
            sizes.append(size)
            tag_code = self._fp.read(4).decode(
                'gb18030').strip('\x00')  # 字符解码
            stroke_num = unpack('H', self._fp.read(2))[0]  # 笔画数
            strokes = {k: [] for k in range(stroke_num)}
            k = 0
            while k <= stroke_num:
                xy = unpack('2h', self._fp.read(4))
                if xy == (-1, 0):
                    k += 1
                elif xy == (-1, -1):
                    tag.update({tag_id: {tag_code: strokes}})  # 更新字典
                    tag_id += 1
                    size = self._fp.read(2)
                    if size == b'':  # 判断是否解码完成
                        ... # print('解码结束！')
                    else:
                        size = unpack('H', size)[0]  # Sample size
                    break
                else:
                    strokes[k].append(xy)  # 记录笔迹坐标
            yield tag, sizes
