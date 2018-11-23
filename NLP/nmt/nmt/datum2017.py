import numpy as np
import re

from TensorflowExercises.NLP.nmt.nmt.corpus_process import text_process,slide_corpus,vocab


def gen_corpus(paths_list, out_dir,name,lang,**kwargs):
    if len(paths_list)!=len(lang):
        print('The number of lists in paths_list should be same as language number.')
        exit(1)
    n=len(paths_list)
    text_list=['' for _ in range(n)]
    for i in range(n):
        text=''
        for path in paths_list[i]:
            with open(path, 'r', encoding='utf8') as f:
                text += f.read()
            if text[-1]!='\n':text+='\n'
        text=text_process(text,lang[i],**kwargs)
        with open(out_dir+'\{0}.{1}'.format(name,lang[i]),'w',encoding='utf8') as f:
            f.write(text)
        text_list[i]=text
    return text_list


def main(paths_list,name,out_dir,slide_ratios,vocab_size_list,lang):
    text_list=gen_corpus(paths_list,out_dir,name,lang)
    text_list_en=text_list[0].splitlines(keepends=True)
    text_list_zh=text_list[1].splitlines(keepends=True)

    #文件较大时，使用numpy.random.shuffle容易内存溢出，使用索引重建python列表可避免
    shuffle_index=np.random.permutation(list(range(len(text_list_en))))
    names = [name + '_' + n for n in ['train.zh', 'dev.zh', 'test.zh']]
    slide_corpus(text_list_zh, shuffle_index, slide_ratios, out_dir, names)
    del text_list_zh
    names = [name + '_' + n for n in ['train.en', 'dev.en', 'test.en']]
    slide_corpus(text_list_en, shuffle_index, slide_ratios, out_dir, names)
    vocab(text_list[0], out_dir,name, 'en', vocab_size_list)
    vocab(text_list[1], out_dir,name, 'zh', vocab_size_list)

if __name__ == '__main__':
    num=[i+1 for i in range(20)]
    name = "datum1~20"
    file_dir='E:\corpus\datum2017'
    paths_list=[[],[]]
    for i in num:
        paths_list[0].append(file_dir+r'\Book{0}_en.txt'.format(i))
        paths_list[1].append(file_dir+r'\Book{0}_cn.txt'.format(i))

    lang=['en','zh']
    out_dir = 'E:\corpus'
    slide_ratios=[0.8, 0.1]
    vocab_size_list=[20000,25000,30000]
    main(paths_list,name,out_dir,slide_ratios,vocab_size_list,lang)
    # gen_corpus([paths_list[0]],out_dir,name,['en'])
    # with open(out_dir+'\{0}.zh'.format(name), 'r', encoding='utf8') as f:
    #     text_zh=f.read()
    # with open(out_dir+'\{0}.en'.format(name), 'r', encoding='utf8') as f:
    #     text_en=f.read()
    # vocab(text_en,out_dir,name,'en',[25000,30000])
    # vocab(text_zh,out_dir,name,'zh',[25000,30000])
    # names=[name+'_'+n for n in ['train.zh','dev.zh','test.zh']]
    # slide_corpus(text_list_zh, slide_ratios, out_dir, names)