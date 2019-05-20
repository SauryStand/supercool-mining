
'''
Python 的星号表达式可以用来解决这个问题。比如，你在学习一门课程，在学期末的时候，
 你想统计下家庭作业的平均成绩，但是排除掉第一个和最后一个分数。如果只有四个分数，你可能就直接去简单的手动赋值，
 但如果有 24 个呢？这时候星号表达式就派上用场了：
 only support python3
'''
# def drop_first_last(grades):
#     first, *middle, last = grades
#     return avg(middle)
# record = ('Dave', 'dave@example.com', '773-555-1212', '847-555-1212')
# name, email, *phone_numbers = record
# print(phone_numbers)
#
# '''
# 星号表达式也能用在列表的开始部分。
# 比如，你有一个公司前 8 个月销售数据的序列， 但是你想看下最近一个月数据和前面 7 个月的平均值的对比。你可以这样做：
# '''
# *trailing_qtrs, current_qtr = sales_record
# trailing_avg = sum(trailing_qtrs) / len(trailing_qtrs)
# return avg_comparison(trailing_avg, current_qtr)
#
# *trailing, current = [10, 8, 7, 1, 9, 5, 10, 3]
#
# print(current == 3)

# python里面的索引的特征是包含起点，但是不包含结束的索引值，-1表示最后一个元素，
# 但是-1是结尾的index，所以含义就是取原始数据的除最后一个元素之外的值
a = [1,5,2,4,7,8]
a[-1]