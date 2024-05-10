# Setup

## Redis
Channels layers requires redis to be running.
```shell
docker run --name redis -p 6379:6379 -d redis
```

## Train the model

```
python train.py
```
