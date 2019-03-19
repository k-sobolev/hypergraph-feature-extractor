import numpy as np


def BuildAdjacencyForEdgeComplex(A):
    ''' Calculates adjacency matrix for 0-simplices and
    adjacency matrix for 1-simplices for edge complex of given graph.

    Args:
        A: adjacency matrix of a given graph

    Returns:
        matrix_0: adjacency matrix for 0-simplices
        matrix_1: adjacency matrix for 1-simplices

    '''

    n = A.shape[0]

    number_0 = 0
    number_1 = 0
    numeration_dict_0 = {}
    numeration_dict_1 = {}

    for i in range(n):
        found = False
        for j in range(i + 1, n):
            if A[i, j] == 1:
                numeration_dict_1[(i, j)] = number_1
                number_1 += 1

                found = True

        if found:
            numeration_dict_0[i] = number_0
            number_0 += 1

    matrix_0 = np.zeros((number_0, number_0))
    for i1 in numeration_dict_0.keys():
        for i2 in numeration_dict_0.keys():
            matrix_0[numeration_dict_0[i1], numeration_dict_0[i2]] = A[i1, i2]

    matrix_1 = np.zeros((number_1, number_1))
    for i1, j1 in numeration_dict_1.keys():
        for i2, j2 in numeration_dict_1.keys():
            if (i1, j1) != (i2, j2):
                if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
                    matrix_1[numeration_dict_1[(i1, j1)], numeration_dict_1[(i2, j2)]] = 1

    return matrix_0, matrix_1


def BuildAdjacencyForTriangleComplex(A, want_2=True):
    ''' Calculates adjacency matrix for 0-simplices, adjacency matrix for 1-simplices
    and adjacency matrix for 2-simplices for triangle complex of given graph.

    Args:
        A: adjacency matrix of a given graph
        want_2: whether or not you want to calcualte matrix_2

    Returns:
        matrix_0: adjacency matrix for 0-simplices
        matrix_1: adjacency matrix for 1-simplices
        matrix_2: adjacency matrix for 2-simplices

    '''

    n = A.shape[0]

    number_0 = 0
    number_1 = 0
    number_2 = 0
    numeration_dict_0 = {}
    numeration_dict_1 = {}
    numeration_dict_2 = {}

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if A[i, j] == 1 and A[i, k] == 1 and A[j, k] == 1:
                    numeration_dict_2[(i, j, k)] = number_2
                    number_2 += 1

                    if not (i, j) in numeration_dict_1:
                        numeration_dict_1[(i, j)] = number_1
                        number_1 += 1
                    if not (i, k) in numeration_dict_1:
                        numeration_dict_1[(i, k)] = number_1
                        number_1 += 1
                    if not (j, k) in numeration_dict_1:
                        numeration_dict_1[(j, k)] = number_1
                        number_1 += 1

                    if not i in numeration_dict_0:
                        numeration_dict_0[i] = number_0
                        number_0 += 1
                    if not j in numeration_dict_0:
                        numeration_dict_0[j] = number_0
                        number_0 += 1
                    if not k in numeration_dict_0:
                        numeration_dict_0[k] = number_0
                        number_0 += 1

    matrix_0 = np.zeros((number_0, number_0))
    for i1 in numeration_dict_0.keys():
        for i2 in numeration_dict_0.keys():
            if i1 != i2:
                if tuple(sorted((i1, i2))) in numeration_dict_1:
                    matrix_0[numeration_dict_0[i1], numeration_dict_0[i2]] = 1

    matrix_1 = np.zeros((number_1, number_1))
    for i1, j1 in numeration_dict_1.keys():
        for i2, j2 in numeration_dict_1.keys():
            if (i1, j1) != (i2, j2):
                union = tuple(sorted(set((i1, j1, i2, j2))))
                if len(union) == 3 and not union in numeration_dict_2:
                    matrix_1[numeration_dict_1[(i1, j1)], numeration_dict_1[(i2, j2)]] = 1

    if not want_2:
        return matrix_0, matrix_1

    matrix_2 = np.zeros((number_2, number_2))
    for i1, j1, k1 in numeration_dict_2.keys():
        for i2, j2, k2 in numeration_dict_2.keys():
            if (i1, j1, k1) != (i2, j2, k2):
                union = tuple(sorted(set((i1, j1, k1, i2, j2, k2))))
                if len(union) == 4:
                    matrix_2[numeration_dict_2[(i1, j1, k1)], numeration_dict_2[(i2, j2, k2)]] = 1

    return matrix_0, matrix_1, matrix_2