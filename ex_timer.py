import time

start = time.time()
print(start)

for i in range(10000000):
    y = 1

end = time.time()
print(end)
print('diff:', end - start)
