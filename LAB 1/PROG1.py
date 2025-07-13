import random
from collections import Counter
import statistics

# 1. Count pairs with sum equal to 10
def countsum10(input_list):
    count = 0
    seen = set()
    for number in input_list:
        complement = 10 - number
        if complement in seen:
            count += 1
        seen.add(number)
    return count

# 2. Find the range of a list (max - min)
def findrange(input_list):
    if len(input_list) < 3:
        return "Range determination is not possible"
    return max(input_list) - min(input_list)

# 3. Matrix exponentiation (A^m)
def matrixpower(matrix, m):
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        return "Matrix is not square"
    result = [[int(i == j) for j in range(n)] for i in range(n)]  
    for _ in range(m):
        result = multiplymatrices(result, matrix)
    return result

def multiplymatrices(A, B):
    n, p, m = len(A), len(B), len(B[0])
    result = [[0]*m for _ in range(n)]
    for x in range(n):
        for y in range(m):
            for z in range(p):
                result[x][y] += A[x][z] * B[z][y]
    return result

# 4. Count highest occurring character in string
def mostcommonchar(input_string):
    filtered = [c.lower() for c in input_string if c.isalpha()]
    count = Counter(filtered)
    if not count:
        return None, 0
    most_common = count.most_common(1)[0]
    return most_common[0], most_common[1]

# 5. Mean, Median, Mode of 25 random numbers between 1 and 10
def generaterandomstats():
    nums = [random.randint(1, 10) for _ in range(25)]
    mean = statistics.mean(nums)
    median = statistics.median(nums)
    mode = statistics.mode(nums)
    return nums, mean, median, mode

# Main Program
if __name__ == "__main__":
    # Q1
    list_q1 = [2, 7, 4, 1, 3, 6]
    pairs_count = countsum10(list_q1)
    print("1. Count of pairs with sum 10:", pairs_count)

    # Q2
    list_q2 = [5, 3, 8, 1, 0, 4]
    range_result = findrange(list_q2)
    print("2. Range of the list:", range_result)

    # Q3
    matrix_q3 = [
        [2, 0],
        [1, 3]
    ]
    m = 2
    matrix_pow = matrixpower(matrix_q3, m)
    print("3. Matrix raised to power", m, ":", matrix_pow)

    # Q4
    input_string = "hippopotamus"
    char, count = mostcommonchar(input_string)
    print("4. Most common alphabet character:", char)
    print("   Occurrence count:", count)

    # Q5.
    nums, mean, median, mode = generaterandomstats()
    print("5. Random Numbers:", nums)
    print("   Mean:", mean)
    print("   Median:", median)
    print("   Mode:", mode)
