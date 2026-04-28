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

### Linux命令

```shell
# 查看可执行文件的依赖库
ldd /usr/bin/python3
ldd /bin/ls

# 查看动态库的依赖
ldd /usr/lib/x86_64-linux-gnu/libc.so.6

# 批量查看目录下所有程序的依赖
find /usr/bin -type f -executable -exec ldd {} \; 2>/dev/null

# 按名称查找库文件
find /usr -name "libc.so.6" -type f 2>/dev/null
find / -name "*.so" -type f 2>/dev/null

# 在标准库目录中查找
find /usr/lib /usr/lib64 -name "libpthread*" -type f

#objdump - 目标文件分析 
# 查看动态库依赖
objdump -p /usr/bin/python3 | grep NEEDED

# 查看共享库的符号表
objdump -T /lib/x86_64-linux-gnu/libc.so.6

# 查看动态段信息
objdump -x /path/to/binary | grep -E "(NEEDED|SONAME)"

# 查看动态库依赖
readelf -d /usr/bin/bash | grep "Shared library"

# 查看所有动态段信息
readelf -d /path/to/program

# 查看符号表
readelf -s /usr/lib/libm.so.6 | grep sqrt

# 查看运行中程序加载的库
lsof -p <pid> | grep "\.so"

# 查看哪个进程使用了特定库
lsof /usr/lib/libc.so.6
``` 

| 场景     | 命令示例                                                      |               |
| ------ | --------------------------------------------------------- | ------------- |
| GPU 状态 | nvidia-smi / nvidia-smi dmon -s pucvmet                   |               |
| 显存占用   | gpustat -cpP1 或 nvtop                                     |               |
| 看模型大小  | du -h /data/models/Qwen2-7B\*                             |               |
| 压测接口   | wrk -t4 -c100 -d30s -s post.lua <http://ip:8000/generate> |               |
| 实时日志   | tail -f /var/log/messages                                 | grep llm\_srv |
| 端口冲突   | lsof -i:8000                                              |               |
| 批量杀进程  | pkill -f uvicorn                                          |               |
| 防火墙    | ufw allow 8000/tcp                                        |               |
| 永久挂载   | echo "/dev/sdb1 /data ext4 defaults 0 0" >> /etc/fstab    |               |

