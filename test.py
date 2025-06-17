# path = '/home/yko/carla1/WorldOnRails-release/Result_TTC_TIT/NSGAII_RAIN_5000_1.8_all/NSGAII_RAIN_5000_1.8_13/file_storage.txt'
path = './file_storage.txt'
f = open(path,'r')
rows = f.readlines()
arr = []
for i,row in enumerate(rows):
    row = row.split(" ")
    row = list(map(float,row[:18]))
    if float(row[13]) < 0:
        print(i," ",row)
        arr.append(row)
print(len(arr))