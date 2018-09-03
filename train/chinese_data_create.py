__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import io
import sys
from PIL import Image,ImageDraw,ImageFont

f = open("chinese\\code3000.txt",'r')

# characters = {}

#从最原始的文件中导出数据
#------------------------------------------------------
# count=0
# while True:
#     str = f.read(1)
#     if str=='':
#         break
#     if str not in characters.keys():
#         characters[str]=count
#         count = count + 1
#     str = f.read(1)
#     if str=='':
#         break
# f.close()
#
# f = open("chinese\\code_map.txt",'w+')
# for key in characters:
#     f.write(key)
#     f.write(':')
#     s = format("%d"%(characters[key]))
#     f.write(s)
#     f.write('\n')
# f.close()
#-------------------------------------------------------------

width = 28
height = 28
font = ImageFont.truetype('simsun.ttc',int(width-width/28*10))
def create_a_pic(chinese_str,str_index,pic_index):
    file_path = format("data\\%d_%d.jpg"%(pic_index,str_index))
    img = Image.new('RGB',(width,height),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((int(width/7),int(height/7)),chinese_str,(0,0,0),font=font)
    img.save(file_path,'jpeg')

def get_code_map():
    characters = {}
    with open("chinese\\code_map1.txt",'r') as f:
        while True:
            line = f.readline()
            line = line.replace('\n','')
            if line=='':
                break
            ss = line.split(':')
            characters[int(ss[1])]=ss[0]
        f.close()
    return  characters,int(width),int(height)

if __name__ == "__main__":
    count = 0
    characters,w,h = get_code_map()
    for key in characters:
        create_a_pic(characters[key],key,count)
        count += 1
    print("data creat success,data nums(%d)"%(len(characters)))


