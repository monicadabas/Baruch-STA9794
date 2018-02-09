import string
import math

# Unique characters in a string


def unique_chars(st):
    s = sorted(st.lower())
    print(s)
    for i in range(1, len(s)):
        if s[i] == " ":
            continue
        elif s[i] == s[i-1]:
            return False

    return True


# If we do not want to consider punctuation as a char we remove it first

def unique_chars_punc(st):
    translator = str.maketrans('', '', string.punctuation) # creates a table to
    # replace characters in 1st arg by those in second arg (index by index) and remove those in third arg
    s = st.translate(translator) # maps the string to the table created above
    s = sorted(s.lower())
    for i in range(1, len(s)):
        if s[i] == " ":
            continue
        elif s[i] == s[i-1]:
            return False

    return True


# Are two strings permutation of each other- no spaces in strings


def permutation_no_space(s1, s2):
    if len(s1) != len(s2):
        return False

    dict_char = {}
    for i in s1:
        i = i.lower()
        if i in dict_char:
            dict_char[i] += 1
        else:
            dict_char[i] = 1

    for i in s2:
        i = i.lower()
        if i in dict_char:
            dict_char[i] -= 1
        else:
            return False

    for key in dict_char.keys():
        if dict_char[key] != 0:
            return False

    return True


# Add %20 in place on space and get the length of string at end. Do it inplace.
# Strings are immutable in python so cannot be done inplace. Hence, changed string to list


def change_str_len(s):
    s_list = [i for i in s]

    i = 0
    while i < len(s_list):
        if s_list[i] == " ":
            s_list[i] = "%"
            s_list.insert(i+1, 2)
            s_list.insert(i+2, 0)
            i += 3
        else:
            i += 1

    return s_list, len(s_list)


# Check if a string is a permutation of a palindrome - it may have spaces


def palindrome_permutation(s):
    l = len(s)
    dict_char = {}
    for i in range(len(s)):
        if s[i] != " ":
            c = s[i].lower()
            if c in dict_char:
                dict_char[c] += 1
            else:
                dict_char[c] = 1
        else:
            l -= 1
    # count of odd values in dict. It should be 1 in case of l == odd else 0
    num_odd_vals = 0

    for key in dict_char.keys():
        if dict_char[key] % 2 != 0:
            num_odd_vals += 1

    if (l % 2 == 0 and num_odd_vals == 0) or (l % 2 != 0 and num_odd_vals == 1):
        return True
    else:
        return False


# Check if two strings can be identical with only one insert/ remove/ replace


def one_way(s1, s2):
    len_diff = len(s1)-len(s2)

    if abs(len_diff) > 1:
        return False

    if len_diff == -1:
        s1, s2 = s2, s1

    s1_index, s2_index = 0, 0

    while s1_index < len(s1) and s2_index < len(s2) and abs(s1_index - s2_index) < 2:
        if s1[s1_index] == s2[s2_index]:
            s1_index += 1
            s2_index += 1
        else:
            if abs(len_diff) == 0:
                s1_index += 1
                s2_index += 1
            else:
                s1_index += 1

    if s2_index == len(s2):
        return True
    return False


# Compress a string aaaa = a4 and return the smaller string, assuming only alphabets in string


def compress_string(s1):
    s = s1.lower()

    s_list = [s[0], 1]

    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            s_list[-1] += 1
        else:
            s_list.append(s[i])
            s_list.append(1)

    if len(s) <= len(s_list):
        return s1

    s_list = [str(a) for a in s_list]
    return "".join(s_list)


# Zero matrix. Return a matrix with zero in all the rows and columns that has a zero in it

def zero_matrix(m):
    r = len(m)
    c = len(m[0])

    rows, cols = set(), set()

    for i in range(r):
        for j in range(c):
            if m[i][j] == 0:
                rows.add(i)
                cols.add(j)

    print(rows, cols)

    for i in rows:
        m[i][:] = [0]*c

    for i in range(r):
        for j in cols:
            m[i][j] = 0

    return m


# Using only 1 isSubstring method (to find if a string is in another string) check if s1 is a rotation of s2


def str_rotation(s1, s2):

    if len(s1) != len(s2):
        return False

    s2 += s2
    if s1 in s2:
        return True

    return False


# In place square matrix rotation by 90 degrees


def matrix_rotation(M):
    m = len(M)
    for d in range(int(m/2)):
        t = d
        b = m - 1 - d

        for i in range(t, b):
            M[t][t+i], M[t+i][b], M[b][b-i], M[b-i][t] = M[b-i][t], M[t][t+i], M[t+i][b], M[b][b-i]

    return M







