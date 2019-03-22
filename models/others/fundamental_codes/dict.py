

string1, string2, string3, string4 = 'aa', 'Trondheim', 'Hammer Dance', ''

not_null = string1 and string2 or string3 or string4

print(not_null)



yes_votes = 42572654
no_votes = 43132495

percentage = float(yes_votes) / float((yes_votes + no_votes))
print('{:-9} YES votes  {:2.2%}'.format(yes_votes, percentage))


for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x * x, x * x * x))
