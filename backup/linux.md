### gpu上建立容器
```powershell
nvidia-docker run -it \   （这边也可以使用docker run -it）
docker run -it -u root \
--gpus all  \
-p 1034:1034 -p 2735:2735 \
-e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
--ulimit memlock=-1 \
--name whk_vllm_091_0728 \
-v /data/:/data/ \
-v /home/model/:/home/model/ \
ee0767a44255 bash

#挂载命令
mkdir -p /nfs-data ; mount -t nfs -o vers=3,timeo=600,nolock 10.170.23.193:/ /nfs-data
``` 

### 初始化设置
```python
vim ~/.bashrc

export HISTSIZE=1000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export https_proxy=http://10.155.192.138:8080
``` 

### 修改pip源
```powershell
vim ~/.pip/pip.conf
[global]
index-url = http://7.223.199.227/pypi/simple
trusted-host = 7.223.199.227
timeout = 120

#pip install torch==2.5.1  --default-timeout=1000 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
``` 

### 多机启动
```powershell
ray：# 指定通信网卡，使用ifconfig查看，找到和主机IP一致的网卡名
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MM_ALL_REDUCE_OP_THRESHOLD=1000000
export HCCL_OP_EXPANSION_MODE="AIV"
export NUMEXPR_MAX_THREADS=192

# 将其中一个节点设为头节点
ray start --head --num-gpus=8
# 在其他节点执行
ray start --address='7.216.55.58:6379' --num-gpus=8
``` 


### 远程链接容器
```powershell
#配置ssh
#第一步config文件
vi /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes
#第二步建立/run/sshd
mkdir /run/sshd

#第三步确认没有sshd时候设置passwd
passwd #这个时候别有sshd
#第四步开启sshd
/usr/sbin/sshd
ssh 7.242.105.173 -p 8035 #来确认是否链接成功

#解决上面报错，生成对应ssh
ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -P '' -q
ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -P '' -q
ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -P '' -q

#不需要ssh命令
git config --global http.sslVerify false

#支持自动迁移代码
 from torch_npu.contrib import transfer_to_npu
``` 

### 打patch包
```powershell
cp -r vllm patch/vllm
cd patch
git init
git add .
git commit  -m "init"
修改代码
git add .
git commit -m "xxxx"
git format-patch -1
修改patch名称
``` 

### 查看cann包
```powershell
#查看cann包版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
#去除所有进程
ps -ef | grep python| grep -v grep | awk '{print $2}' | xargs kill -9
#pytorch中查看日志的两行命令
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
``` 

### 可视化数据
```powershell
# 使用 pip3 (推荐)
pip3 install visidata

# Ubuntu/Debian
sudo apt-get install visidata

# macOS (使用Homebrew)
brew install visidata


vd + csv文件
``` 