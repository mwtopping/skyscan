

done = []

with open('./data/utc2025apr17.dat', 'r') as f:
    while True:
        line = f.readline().strip()
        line2 = f.readline().strip()
        line3 = f.readline().strip()


        if line == "":
            break

        satnum = line3.split(' ')[1]
        if satnum in done:
            continue
        done.append(satnum)
        print(line)
        print(line2)
        print(line3)

