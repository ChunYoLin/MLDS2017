import re
import json
import collections
import numpy as np

def build_vocab():
    with open("./MLDS_hw2_data/training_label.json") as train:
        train_json = json.load(train)
        words = []
        for i in range(len(train_json)):
            for caption in train_json[i]["caption"]:
                caption = 'BOS ' + caption
                caption = re.sub('\.', ' EOS', caption)
                for word in caption.split():
                    words.append(word)
        count = [['UNK', -1]]
        count.extend(collections.Counter(words))
        
build_vocab()
