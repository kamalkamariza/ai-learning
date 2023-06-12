import numpy as np

array_list = [1,2,3]
array_numpy = np.array([1,2,3])

'''
    Addition

    List append the item into the array
    Numpy adds to every item in the array
    based on the shape of numpy array, it can be 1 or need to match the shape of original array
    np.array([1,2,3]) will fail adding np.array([1,2]) different shape (3,) and (2,)
'''
array_list = array_list + [4]
array_numpy = array_numpy + np.array([4])
array_numpy = array_numpy + np.array([1,1,1])

print(array_list)
print(array_numpy)

'''
    Multiplication

    List will repeat the list multiplier times
    Numpy array will multiply every item with multiplier
'''
array_list = array_list * 2
array_numpy = array_numpy * 2

print(array_list)
print(array_numpy)

'''
    Dot Product
'''

list_a = np.array([1,2])
list_b = np.array([2,4])

dot_prod = np.sum(list_a * list_b)
print(dot_prod)
dot_prod = np.dot(list_a, list_b)
print(dot_prod)

'''
    Matrix
'''

matrix_list = [[1,2], [3,4]]
print(matrix_list)
matrix_numpy = np.array([[1,2], [3,4]])
print(matrix_numpy)