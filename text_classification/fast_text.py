import fastText

dir = 'E:\corpus\cnews\\'
file_list = [dir+ name for name in ['train_token.txt','val_token.txt','test_token.txt']]

classifier=fastText.train_supervised(dir+'train_token.txt',
                                     epoch=20,
                                     dim=100,
                                     lr=1,
                                     wordNgrams=3)
classifier.save_model(dir+'cnews_model_400')
result=classifier.test(dir+'test_token.txt')
print('P@1:%s, R@1:%s, n:%s' % (result[1],result[2],result[0]))
result=classifier.test(dir+'train_token.txt')
print('P@1:%s, R@1:%s, n:%s' % (result[1],result[2],result[0]))