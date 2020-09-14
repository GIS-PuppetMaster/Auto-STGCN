import os
import paramiko
from paramiko import SSHClient
from scp import SCPClient

server_list = [('124.70.7.137', 22), ('124.70.32.29', 22), ('119.3.165.88', 22)]
abs_path = os.path.split(os.path.abspath(__file__))[:-1][0]
# file = abs_path + '\\Config\\PEMS03\\'
# server_path = '/usr/zkx/GNN-NAS-RL/Config/PEMS03/'
file = abs_path + '\\Config\\'
server_path = '/usr/zkx/GNN-NAS-RL/Config/'
for ip, port in server_list:
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip,
                port=port,
                username='root',
                password='Hadoop0201')
    scp = SCPClient(ssh.get_transport())
    message = scp.put(file, server_path, recursive=True)
    ssh.close()
