节点间ssh免密（两个节点都需要操作）


ssh-keygen -t rsa -b 4096 -C "wuyiqiang@example.com"
ssh-copy-id root@10.1.30.43

vim ~/.ssh/config

Host a100-44
    HostName 10.1.30.44
    User root
    IdentityFile ~/.ssh/id_rsa