#FROM ghcr.io/zhayujie/chatgpt-on-wechat:latest
FROM ghcr.io/xiao3344/chatgpt-on-wechat-master:master

ENTRYPOINT ["/entrypoint.sh"]