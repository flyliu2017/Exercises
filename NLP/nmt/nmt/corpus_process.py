import jieba
import numpy as np
import re
from collections import Counter

def en_preprocess(text):
    text=re.sub(r'([;,()[\]?:]|[.]{2,}|\*+)',r' \1 ',text)
    text=re.sub(r'["”“]',r' &quot; ',text)
    text=re.sub(r'[`]','\'',text)
    text=re.sub(r'(^|\s)‘([^\n]*?)’(?=$|\s|\.)',r'\1 &quot2; \2 &quot2; ',text)
    text=re.sub(r'(^|\s)\'([^\n]*?)\'(?=$|\s|\.)',r'\1 &quot2; \2 &quot2; ',text)
    text=re.sub(r'can\'t’',r'can &apos;t',text)
    text=re.sub(r'(n\'(?=t))|(?<=\w)\'(?=([sdm]|ve|ll|re)?\s)|’',r' &apos;',text)
    text=re.sub(r'(?<=\s)\'',r'&quot2; ',text)
    text=re.sub(r'((^|\s)[^\s.]+)\.(?=$|\s)',r'\1 .',text)
    text=re.sub(r'(?<=\d) , (?=\d)',r',',text)
    text=re.sub(r' +',r' ',text)
    text = re.sub(r'\n ', r'\n', text)

    return text

def zh_preprocess(text,segment=True):
    text=re.sub('，',',',text)
    text=re.sub('。','.',text)
    if segment:
        text=' '.join(jieba.cut(text))
    text=re.sub(r' +',r' ',text)
    text=re.sub(r'\n +',r'\n',text)
    text=re.sub(r'[“”]',r'&quot;',text)
    text=re.sub(r'[‘’]',r'&quot2;',text)

    return text

def text_process(text,lang,func_dict,**kwargs):
    dict = {'zh': zh_preprocess, 'en': en_preprocess}
    dict.update(func_dict)
    return dict[lang](text,**kwargs[lang])

def gen_corpus_for_mixed_text(path, out_dir, name, lang='all'):
    with open(path, 'r', encoding='utf8') as f:
        text = f.readlines()
    en_text = ''.join(text[::2])
    zh_text = ''.join(text[1::2])
    en_text=en_preprocess(en_text)
    zh_text=zh_preprocess(zh_text)
    if lang=='all' or lang=='en':
        with open(out_dir+'\{0}.en'.format(name),'w',encoding='utf8') as f:
            f.write(en_text)
    if lang=='all' or lang=='zh':
        with open(out_dir+'\{0}.zh'.format(name), 'w', encoding='utf8') as f:
            f.write(zh_text)

    return en_text,zh_text

def slide_corpus(text_list, shuffle_index,slide_ratios, out_dir, names):
    if len(names)!=len(slide_ratios)+1:
        print('wrong num of output paths!')
        exit(1)
    sum=np.cumsum(slide_ratios)
    if sum[np.logical_or(sum<0,sum>1 )].size:
        print('wrong slide_ratio!')
        exit(1)
    l=len(text_list)
    slide_num=[0]+[int(l*sum[i]) for i in range(len(sum))]+[l]

    #使用numpy.random.shuffle容易内存溢出，使用索引重建python列表可避免
    shuffle_list=[text_list[i] for i in shuffle_index]
    for i in range(len(slide_num)-1):
        with open(out_dir+'\\'+names[i], 'w', encoding='utf8') as f:
            f.writelines(shuffle_list[slide_num[i]:slide_num[i+1]])

def vocab(text, out_dir,name, lang,vocab_size_list):
    if not out_dir:
        print("Must provide a output path.")
        exit(1)

    text = text.replace('\n', ' ')
    b = text.split(' ')
    counter = Counter(b)
    u = ['<unk>', '<s>', '</s>','&quot;','&quot2;','&apos;','.',',']
    counter.pop('',None)
    for k in u:
        counter.pop(k,None)
    for n in vocab_size_list:
        words = counter.most_common(n)
        with open(out_dir+r'\vocab_{0}_{1}.{2}'.format(name,n,lang), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(u + [w[0] for w in words]))

def main(path,name,out_dir,slide_ratios,vocab_size_list):
    en_text,zh_text=gen_corpus_for_mixed_text(path, out_dir, name)
    text_list_en=en_text.splitlines(keepends=True)
    text_list_zh=zh_text.splitlines(keepends=True)

    # 文件较大时，使用numpy.random.shuffle容易内存溢出，使用索引重建python列表可避免
    shuffle_index=np.random.permutation(list(range(len(text_list_en))))
    names = [name + '_' + n for n in ['train.zh', 'dev.zh', 'test.zh']]
    slide_corpus(text_list_zh,shuffle_index, slide_ratios, out_dir, names)
    names=[name + '_' + n for n in ['train.en','dev.en','test.en']]
    slide_corpus(text_list_en,shuffle_index, slide_ratios, out_dir, names)
    vocab(en_text, out_dir,name, 'en', vocab_size_list)
    vocab(zh_text, out_dir,name, 'zh', vocab_size_list)

if __name__ == '__main__':

    name = "Subtitles"
    path = r'E:\corpus\UM-Corpus\data\Bilingual\{0}\Bi-{0}.txt'.format(name)
    out_dir = 'E:\corpus'
    slide_ratios=[0.8, 0.1]
    vocab_size_list=[20000,25000,30000]
    d = {'zh': zh_preprocess, 'en': en_preprocess}

    main(path,name,out_dir,slide_ratios,vocab_size_list)