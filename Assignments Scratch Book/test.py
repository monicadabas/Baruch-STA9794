file1 = "/Users/monicadabas/Desktop/MS Course Work/Spring 2017/Big Data- STA 9497/2017-STAT-9794-Monica-Dabas/AssignmentA_DataScrubberAndAnalyzer/noise.txt"
file2 = "/Users/monicadabas/Desktop/MS Course Work/Spring 2017/Big Data- STA 9497/2017-STAT-9794-Monica-Dabas/Assignments Scratch Book/noise_scratch.txt"
with open(file1,"r") as file1:
    lines_scrub = file1.readlines()
with open(file2,"r") as file2:
    lines_scratch = file2.readlines()

print("Noise in Scrub: {}".format(len(lines_scrub)))
print("Noise in Scratch: {}".format(len(lines_scratch)))

d = {}
for line in lines_scratch:
    if line in d:
        d[line] += 1
    else:
        d[line] = 1

dup = 0
for key in d:
    if d[key] > 1:
        print(key, ",",d[key])
        dup += 1

print("Duplicate indexes: {}".format(dup))

count = [0]*12

for line in lines_scratch:
    line.strip("\n")
    if int(line) < 10000:
        count[0] += 1
    elif 9999 < int(line) < 20000:
        count[1] += 1
    elif 19999 < int(line) < 20002:
        count[2] += 1
    elif 20001 < int(line) < 30002:
        count[3] += 1
    elif 30001 < int(line) < 40001:
        count[4] += 1
    elif 40000 < int(line) < 50001:
        count[5] += 1
    elif 50000 < int(line) < 60000:
        count[6] += 1
    elif 59999 < int(line) < 70000:
        count[7] += 1
    elif 69999 < int(line) < 80000:
        count[8] += 1
    elif 79999 < int(line) < 80003:
        count[9] += 1
    elif 80002 < int(line) < 90003:
        count[10] += 1
    else:
        count[11] += 1

print("Scratch noise per category: {}".format(count))

count1 = [0]*12
for line in lines_scrub:
    line.strip("\n")
    if int(line) < 10000:
        count1[0] += 1
    elif 9999 < int(line) < 20000:
        count1[1] += 1
    elif 19999 < int(line) < 20002:
        count1[2] += 1
    elif 20001 < int(line) < 30002:
        count1[3] += 1
    elif 30001 < int(line) < 40001:
        count1[4] += 1
    elif 40000 < int(line) < 50001:
        count1[5] += 1
    elif 50000 < int(line) < 60000:
        count1[6] += 1
    elif 59999 < int(line) < 70000:
        count1[7] += 1
    elif 69999 < int(line) < 80000:
        count1[8] += 1
    elif 79999 < int(line) < 80003:
        count1[9] += 1
    elif 80002 < int(line) < 90003:
        count1[10] += 1
    else:
        count1[11] += 1

print("Scrub noise per category: {}".format(count1))

Scratch_not_in_scrub = []
for i in lines_scratch:
    if i not in lines_scrub:
        Scratch_not_in_scrub.append(i)

scrub_not_in_scratch = []
for i in lines_scrub:
    if i not in lines_scratch:
        scrub_not_in_scratch.append(i)

print("Noise not captured by Scrub: {}".format(Scratch_not_in_scrub))
print(len(Scratch_not_in_scrub))
print("Extra noise captured by Scrub: {}".format(scrub_not_in_scratch))
print(len(scrub_not_in_scratch))