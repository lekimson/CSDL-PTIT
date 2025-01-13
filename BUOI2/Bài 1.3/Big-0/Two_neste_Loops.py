#đây là chương trình python về 2 vòng lặp lồng nhau 
def printPairs(n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            print(f"({i}, {j})")


printPairs(3)
