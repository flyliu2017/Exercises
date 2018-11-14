import jieba

dir = r'E:\corpus\cnews\\'
file_list = [dir+ name for name in ['cnews.train.txt','cnews.val.txt','cnews.test.txt']]
write_list = [dir+ name for name in ['train_token.txt','val_token.txt','test_token.txt']]

def tokenFile(file_path, write_path):
    with open(write_path, 'w',encoding='utf8') as w:
        with open(file_path, 'r',encoding='utf8') as f:
            for line in f.readlines():
                token_sen = jieba.cut(line.split('\t')[1])
                w.write('__label__'+line.split('\t')[0] + '\t' + ' '.join(token_sen))
    print(file_path + ' has been token and token_file_name is ' + write_path)

for file_path, write_path in zip(file_list, write_list):
    tokenFile(file_path, write_path)