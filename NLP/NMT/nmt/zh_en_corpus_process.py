import jieba


def gen_corpus(path, name, lang='all'):
    with open(path, 'r', encoding='utf8') as f:
        text = f.readlines()
        en_text = text[::2]
        zh_text = text[1::2]
    if lang=='all' or lang=='en':
        with open('E:\corpus\{0}.en'.format(name),'w',encoding='utf8') as f:
            f.writelines(en_text)
    if lang=='all' or lang=='zh':
        with open('E:\corpus\{0}.zh'.format(name), 'w', encoding='utf8') as f:
            for line in zh_text:
                f.write(' '.join(jieba.cut(line)))


# name='Subtitles'
name='Education'
path = r'E:\corpus\UM-Corpus\data\Bilingual\{0}\Bi-{0}.txt'.format(name)
gen_corpus(path, name, 'en')

