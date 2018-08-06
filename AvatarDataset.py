# -*- coding: utf-8 -*-
import os, glob, random
import numpy as np
from PIL import Image

jobs = [
    '초보자', '전사', '마법사', '궁수', '도적', '해적', '기사단',
    '아란', '에반', '레지스탕스', '메르세데스', '팬텀', '루미너스', '카이저',
    '엔젤릭버스터', '제로', '은월', '키네시스', '카데나', '일리움', '아크'
]

def load_data():
    filelist = glob.glob('./avatars/*.png')
    data = (
        np.array([np.array(Image.open(fname)) for fname in filelist])[:, :, :, 0],
        [int(label.replace('./avatars/', '').split('-')[0]) for label in filelist]
    )
    dataset = []
    for i in range(len(data[1])):
        dataset.append((data[0][i], data[1][i]))
    random.shuffle(dataset)
    test = dataset[:1500]
    train = dataset[1500:]
    test = (np.array([data[0] for data in test]), [data[1] for data in test])
    train = (np.array([data[0] for data in train]), [data[1] for data in train])
    return train, test
    
if __name__ == '__main__':
    print(load_data())
