import os
import paramiko
from paramiko import SSHClient
from scp import SCPClient

server_list = [('124.70.7.137', 22), ('124.70.32.29', 22), ('119.3.165.88', 22)]
abs_path =os.path.split(os.path.abspath(__file__))[:-1][0]
file = abs_path + '\\Log\\PEMS07_experiment2_qlearning_2_batch32\\'
if not os.path.exists(file):
    os.mkdir(file)
server_path = '/usr/zkx/GNN-NAS-RL/Log/PEMS07_experiment2_qlearning_2_batch32/logger.pkl'
for ip, port in server_list:
    if ip == '124.70.7.137':
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=ip,
                    port=port,
                    username='root',
                    password='Hadoop0201')
        scp = SCPClient(ssh.get_transport())
        scp.get(server_path, file, recursive=True)