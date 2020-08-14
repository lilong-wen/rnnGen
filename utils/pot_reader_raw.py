import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import cv2
import struct
import glob

class POT_RAW:

    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(self.path)
        self.counter = 0

    def drawStroke(self, img, pts, xmin, ymin, x_shift, y_shift):
        pt_length = len(pts)
        stroke_start_tag = False
        for i in range(1, pt_length):
            if pts[i][0] == -1 and pts[i][1] == 0:
                stroke_start_tag = True
                continue
            if stroke_start_tag:
                stroke_start_tag = False
                continue
            x_delta, y_delta = -xmin+x_shift, -ymin+y_shift

            cv2.line(img,
                     (pts[i-1][0]+x_delta, pts[i-1][1]+y_delta),
                     (pts[i][0]+x_delta, pts[i][1]+y_delta),
                     color=(0, 0, 0),
                     thickness=5)
        return img

    def read_from_pot_dir(self, pot_dir):

        def one_file(f):

            while True:
                # 文件头，交代了该sample所占的字节数以及label以及笔画数
                header = np.fromfile(f, dtype='uint8', count=8)
                if not header.size:
                    break
                sample_size = header[0] +(header[1]<<8)
                tagcode = header[2] + (header[3]<<8) + (header[4]<<16) + (header[5]<<24)
                stroke_num = header[6] + (header[7]<<8)

                # 以下是参考官方POTView的C++源码View部分的Python解析代码
                traj = []
                xmin, ymin, xmax, ymax = 100000, 100000, 0, 0
                for i in range(stroke_num):
                    while True:
                        header = np.fromfile(f, dtype='int16', count=2)
                        x, y = header[0], header[1]
                        traj.append([x, y])

                        if x == -1 and y == 0:
                            break
                        else:
                            # 个人理解此处的作用是找到描述该字符的采样点的xmin,ymin,xmax,ymax
                            # 但此处若采用源码的逻辑if x < xmin: xmin = x, else if x > xmax: xmax = x会出现了bug
                            # 如果points中x或y是递减的，由于不会执行else判断，会导致xmax或ymax始终为0
                            if x < xmin: xmin = x
                            if x > xmax: xmax = x
                            if y < ymin: ymin = y
                            if y > ymax: ymax = y
                # 最后还一个标志文件结尾的(-1, -1)
                header = np.fromfile(f, dtype='int16', count=2)

                # 根据得到的采样点重构出样本
                x_shift, y_shift = 5, 5 # 画线是有thickness的，所以上下左右多padding几格
                canva = np.ones((ymax-ymin+2*y_shift, xmax-xmin+2*x_shift),
                                dtype=np.uint8)*255
                pts = np.array(traj)
                img = self.drawStroke(canva, pts, xmin, ymin, x_shift, y_shift)

                yield img, tagcode

        for file_name in os.listdir(pot_dir):
            if file_name.endswith('.pot'):
                file_path = os.path.join(pot_dir, file_name)
                with open(file_path, 'rb') as f:
                    for img, tagcode in one_file(f):
                        yield img, tagcode


    def draw_all(self):
        for image, tagcode in tqdm(self.read_from_pot_dir(pot_dir=self.path)):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
            im = Image.fromarray(image)
            image_dir = os.path.join(os.path.dirname(self.path),
                                     os.path.basename(self.path) + "_img/")
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

            pick_file = os.path.dirname(self.path) + "/*.pkl"

            if len(glob.glob(pick_file)) != 0:
                char_dict_file = open(glob.glob(pick_file)[0], 'rb')
                char_dict = pickle.load(char_dict_file)
            else:
                self.make_alphabet()
                char_dict_file = open(glob.glob(pick_file)[0], 'rb')
                char_dict = pickle.load(char_dict_file)

            dir_name = image_dir + '%0.5d'%char_dict[tagcode_unicode]
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            im.convert('RGB').save(dir_name+'/' + str(self.counter) + '.png')
            self.counter += 1

    def make_alphabet(self):
        char_set = set()
        for _, tagcode in tqdm(self.read_from_pot_dir(pot_dir=self.path)):
            # tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
            char_set.add(tagcode_unicode)
        char_list = list(char_set)
        char_dict = dict(zip(sorted(char_list), range(len(char_list))))
        alphabet_length = len(char_dict)

        alphabet_path = os.path.join(os.path.dirname(self.path),
                                     'alphabet_'+str(alphabet_length)+'.pkl')
        with open(alphabet_path, 'wb') as f:
            pickle.dump(char_dict, f)

        print('alphabet length: ', alphabet_length)
