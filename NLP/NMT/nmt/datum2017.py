import numpy as np
import re
from nltk.book import FreqDist

from TensorflowExercises.NLP.nmt.nmt.corpus_process import text_process

def zh_preprocess(text):
    text=re.sub('，',',',text)
    text=re.sub('。','.',text)
    text=re.sub(r' +',r' ',text)
    text=re.sub(r'\n ',r'\n',text)
    text=re.sub(r'[‘’“”]',r'&quot;',text)

    return text

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
        text=text_process(text,lang[i],**kwargs)
        with open(out_dir+'\{0}.{1}'.format(name,lang[i]),'w',encoding='utf8') as f:
            f.write(text)
        text_list[i]=text
    return text_list

def slide_corpus(text_list, slide_ratios, out_dir, names):
    if len(names)!=len(slide_ratios)+1:
        print('wrong num of output paths!')
        exit(1)
    sum=np.cumsum(slide_ratios)
    if sum[np.logical_or(sum<0,sum>1 )].size:
        print('wrong slide_ratio!')
        exit(1)
    l=len(text_list)
    slide_num=[0]+[int(l*sum[i]) for i in range(len(sum))]+[l]
    shuffled_list=np.random.permutation(text_list)
    for i in range(len(slide_num)-1):
        with open(out_dir+'\\'+names[i], 'w', encoding='utf8') as f:
            f.writelines(shuffled_list[slide_num[i]:slide_num[i+1]])

def vocab(text, out_dir,name, lang,vocab_size_list):
    if not out_dir:
        print("Must provide a output path.")
        exit(1)

    text = text.replace('\n', ' ')
    b = text.split(' ')
    fd = FreqDist(b)
    u = ['<unk>', '<s>', '</s>','&quot;','.',',']
    fd.pop('',None)
    for k in u:
        fd.pop(k,None)
    for n in vocab_size_list:
        words = fd.most_common(n)
        with open(out_dir+r'\vocab_{0}_{1}.{2}'.format(name,n,lang), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(u + [w[0] for w in words]))

def main(paths_list,name,out_dir,slide_ratios,vocab_size_list,lang):
    text_list=gen_corpus(paths_list,out_dir,name,lang,zh=zh_preprocess)
    text_list_en=text_list[0].splitlines(keepends=True)
    text_list_zh=text_list[1].splitlines(keepends=True)
    names = [name + '_' + n for n in ['train.zh', 'dev.zh', 'test.zh']]
    slide_corpus(text_list_zh, slide_ratios, out_dir, names)
    names=[name + '_' + n for n in ['train.en','dev.en','test.en']]
    slide_corpus(text_list_en, slide_ratios, out_dir, names)
    vocab(text_list[0], out_dir,name, 'en', vocab_size_list)
    vocab(text_list[1], out_dir,name, 'zh', vocab_size_list)

if __name__ == '__main__':
    num=[1,2]
    name = "datum1~2"
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
    # gen_corpus(path,out_dir,name,'zh')
    # with open(out_dir+'\{0}.zh'.format(name), 'r', encoding='utf8') as f:
    #     text_zh=f.read()
    # with open(out_dir+'\{0}.en'.format(name), 'r', encoding='utf8') as f:
    #     text_en=f.read()
    # vocab(text_en,out_dir,name,'en',[25000,30000])
    # vocab(text_zh,out_dir,name,'zh',[25000,30000])
    # names=[name+'_'+n for n in ['train.zh','dev.zh','test.zh']]
    # slide_corpus(text_list_zh, slide_ratios, out_dir, names)