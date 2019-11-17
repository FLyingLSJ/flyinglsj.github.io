---
layout: post
title: v2ray 配置
date: 2019-10-10
tag: 服务器 FQ
---


```javascript
apt-get install curl 安装 curl
bash <(curl -L -s https://install.direct/go.sh)

vi /etc/v2ray/config.json
sudo systemctl start v2ray 启动 v2ray
systemctl restart v2ray 重启 v2ray
```



参考：

<https://www.v2ray.com/chapter_00/install.html>

<https://toutyrater.github.io/prep/start.html>

<https://intmainreturn0.com/v2ray-config-gen/#> 配置文件网站

[https://github.com/233boy/v2ray/wiki/V2Ray%E6%90%AD%E5%BB%BA%E8%AF%A6%E7%BB%86%E5%9B%BE%E6%96%87%E6%95%99%E7%A8%8B](https://github.com/233boy/v2ray/wiki/V2Ray搭建详细图文教程) 一键配置

 http://port.ping.pe/74.82.204.227:27692   ping 网络

 https://www.atrandys.com/2018/290.html  添加多个用户

```
{
    "log": {
        "access": "/var/log/v2ray/access.log",
        "error": "/var/log/v2ray/error.log",
        "loglevel": "warning"
    },
    "inbound": {
        "port": 8899,
        "protocol": "vmess",
        "settings": {
            "clients": [
                {
                    "id": "fe66584e-d7bc-562d-01df-7dd87205bb1a",
                    "level": 1,
                    "alterId": 101
                }
            ]
        },
        "streamSettings": {
            "network": "kcp"
        },
        "detour": {
            "to": "vmess-detour-185623"
        }
    },
    "outbound": {
        "protocol": "freedom",
        "settings": {}
    },
    "inboundDetour": [
        {
            "protocol": "vmess",
            "port": "10000-10010",
            "tag": "vmess-detour-185623",
            "settings": {},
            "allocate": {
                "strategy": "random",
                "concurrency": 5,
                "refresh": 5
            },
            "streamSettings": {
                "network": "kcp"
            }
        }
    ],
    "outboundDetour": [
        {
            "protocol": "blackhole",
            "settings": {},
            "tag": "blocked"
        }
    ],
    "routing": {
        "strategy": "rules",
        "settings": {
            "rules": [
                {
                    "type": "field",
                    "ip": [
                        "0.0.0.0/8",
                        "10.0.0.0/8",
                        "100.64.0.0/10",
                        "127.0.0.0/8",
                        "169.254.0.0/16",
                        "172.16.0.0/12",
                        "192.0.0.0/24",
                        "192.0.2.0/24",
                        "192.168.0.0/16",
                        "198.18.0.0/15",
                        "198.51.100.0/24",
                        "203.0.113.0/24",
                        "::1/128",
                        "fc00::/7",
                        "fe80::/10"
                    ],
                    "outboundTag": "blocked"
                }
            ]
        }
    }
}
```

