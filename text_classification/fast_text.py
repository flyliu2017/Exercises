import fastText

dir = 'E:\corpus\cnews\\'
file_list = [dir+ name for name in ['train_token.txt','val_token.txt','test_token.txt']]

classifier=fastText.train_supervised(dir+'train_token.txt',epoch=25,dim=300)
classifier.save_model(dir+'cnews_model')
result=classifier.test(dir+'test_token.txt')
print('P@1:%s, R@1:%s, n:%s' % (result[1],result[2],result[0]))
result=classifier.test(dir+'train_token.txt')
print('P@1:%s, R@1:%s, n:%s' % (result[1],result[2],result[0]))