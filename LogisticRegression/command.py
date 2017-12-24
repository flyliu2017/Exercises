import os
# os.chdir(os.getcwd()+'/lr/LogisticRegression/')
import sys
# print(os.getcwd()+'/lr/LogisticRegression')
# sys.path.append(os.getcwd()+'/lr/LogisticRegression/')

ps0='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=192.168.1.157:2000 --worker_hosts=192.168.1.157:2001,192.168.1.159:2002,192.168.1.159:2003 --job_name=ps --task_index=0 --sync=1'
worker0='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=192.168.1.157:2000 --worker_hosts=192.168.1.157:2001,192.168.1.159:2002,192.168.1.159:2003 --job_name=worker --task_index=0 --sync=1'
worker1='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=192.168.1.157:2000 --worker_hosts=192.168.1.157:2001,192.168.1.159:2002,192.168.1.159:2003 --job_name=worker --task_index=1 --sync=1'
worker2='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=192.168.1.157:2000 --worker_hosts=192.168.1.157:2001,192.168.1.159:2002,192.168.1.159:2003 --job_name=worker --task_index=2 --sync=1'
#
# c='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=127.0.0.1:2000 --worker_hosts=127.0.0.1:2001,127.0.0.1:2002 --job_name=ps --task_index=0 & \
# python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=127.0.0.1:2000 --worker_hosts=127.0.0.1:2001,127.0.0.1:2002 --job_name=worker --task_index=0 & \
# python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --ps_hosts=127.0.0.1:2000 --worker_hosts=127.0.0.1:2001,127.0.0.1:2002 --job_name=worker --task_index=1'
c=ps0+' & '+worker0
single='python3 /root/PycharmProjects/tf/lr/LogisticRegression/TwoExamsDistributed.py --worker_hosts=192.168.1.157:2001 --job_name=worker --task_index=0 --sync=1'

os.system(c)
# os.system(single)