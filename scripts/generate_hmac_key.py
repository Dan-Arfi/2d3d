import hmac, hashlib, random
key = bytes(random.getrandbits(8) for _ in range(32))
print(key.hex())
mac = hmac.new(key, digestmod=hashlib.sha256)
# f = open('checkpoints_shapenet_v3/best.pt', 'rb')
# for chunk in iter(lambda: f.read(1024*1024), b''):
#     mac.update(chunk)
# f.close()
# print(mac.hexdigest())
with open('checkpoints_shapenet_v3/best.pt', 'rb') as f:
    for chunk in iter(lambda: f.read(1024*1024), b''):
        mac.update(chunk)

print(mac.hexdigest())
