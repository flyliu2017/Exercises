from nltk.book import FreqDist


def zh_vocab(vocab_size_list):
    path=r"E:\pyProject\TensorflowExercises\NLP\nmt\nmt\data\\"
    text=''
    for fn in ['train.zh', 'dev.zh', 'test.zh']:
        with open(path + fn, 'r', encoding='utf-8') as f:
            text += f.read()
    text = text.replace('\n', '')
    b = text.split(' ')
    fd = FreqDist(b)
    fd.pop('')
    u = ['<unk>', '<s>', '</s>']
    for n in vocab_size_list:
        words=fd.most_common(n)
        with open(path + 'vocab{0}.zh'.format(n), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(u + [w[0] for w in words]))

def vocab(input_dir,out_dir,vocab_size_list,suffix=''):
    if not suffix:
        print("Must provide a file suffix.")
        exit(1)
    text=''
    try:
        with open(input_dir, 'r', encoding='utf-8') as f:
            text += f.read()
    except Exception as e:
        print(e)
        exit(1)
    text = text.replace('\n', '')
    b = text.split(' ')
    fd = FreqDist(b)
    if fd.get(''):fd.pop('')
    u = ['<unk>', '<s>', '</s>']
    for n in vocab_size_list:
        words=fd.most_common(n)
        with open(out_dir + 'vocab{0}.{1}'.format(n,suffix), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(u + [w[0] for w in words]))

path='E:\corpus\Education.en'
vsl=[20000,30000,40000]
od=r'E:\pyProject\TensorflowExercises\NLP\nmt\nmt\data\\'
vocab(path,od,vsl,'en')
