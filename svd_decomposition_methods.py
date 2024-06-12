import math
import copy
import random
import numpy as np


def scalar_mult_vec(vec1, vec2):
    n = len(vec1)
    s = 0
    for i in range(n):
        s += vec1[i] * vec2[i]
    return s


def norm_vec(vec):
    n = len(vec)
    s = 0
    for i in range(n):
        s += vec[i] ** 2
    norm = s ** 0.5
    return norm


def norm_vec_sq(vec):
    n = len(vec)
    s = 0
    for i in range(n):
        s += vec[i] ** 2
    norm = s
    return norm


def sum_vec(vec1, vec2):
    n = len(vec1)
    vec = [0] * n
    for i in range(n):
        vec[i] = vec1[i] + vec2[i]
    return vec


def sub_vec(vec1, vec2):
    n = len(vec1)
    vec = [0] * n
    for i in range(n):
        vec[i] = vec1[i] - vec2[i]
    return vec


def mult_vec_num(num, vec):
    n = len(vec)
    vec1 = [0] * n
    for i in range(n):
        vec1[i] = num * vec[i]
    return vec1


def mult_matr_matr(matr1, matr2):
    n1 = len(matr1)
    m1 = len(matr1[0])
    m2 = len(matr2[0])
    matr = [[0] * m2 for i in range(n1)]
    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                matr[i][j] += matr1[i][k] * matr2[k][j]
    return matr


def sub_matr_matr(matr1, matr2):
    n = len(matr1)
    matr = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            matr[i][j] = matr1[i][j] - matr2[i][j]
    return matr


def norm_matr(matr):
    n = len(matr)
    s = 0
    for i in range(n):
        for j in range(n):
            s += matr[i][j] ** 2
    norm = s ** 0.5
    return norm


def print_matr(matr):
    n = len(matr)
    m = len(matr[0])
    for i in range(n):
        for j in range(m):
            print("{value:4.20f}".format(value=matr[i][j]), end="  ")
        print()


def mult_matr_vec(matr, vec):
    n = len(vec)
    m = len(matr[0])
    vec1 = [0] * n
    for i in range(n):
        for j in range(m):
            vec1[i] += matr[i][j] * vec[j]
    return vec1


def mult_matr_num(num, matr):
    n = len(matr)
    matr1 = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            matr1[i][j] = num * matr[i][j]
    return matr1


def transp_matr(matr):
    n = len(matr)
    m = len(matr[0])
    matr1 = [[0] * n for i in range(m)]
    for i in range(m):
        for j in range(n):
            matr1[i][j] = matr[j][i]
    return matr1


def generate_matrix(n, m, v1, v2):
    a = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            a[i][j] = random.uniform(v1, v2)
    return a


def generate_symm_matrix(n, v1, v2):
    a = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            a[i][j] = random.uniform(v1, v2)
            if i != j:
                a[j][i] = a[i][j]
    return a


def det_matr(matr):
    n = len(matr)
    if n == 2:
        return matr[0][0] * matr[1][1] - matr[1][0] * matr[0][1]
    else:
        d = 0
        for i in range(n):
            d += matr[0][i] * alg_add(matr, 0, i)
        return d


def alg_add(matr, i, j):
    n = len(matr)
    matr1 = []
    for k in range(n):
        if k != i:
            matr1.append([])
            for q in range(n):
                if q != j:
                    matr1[-1].append(matr[k][q])
    return (-1) ** (i + j) * det_matr(matr1)


def inv_matr(matr):
    n = len(matr)
    matr_inv = [[0] * n for i in range(n)]
    c = 1 / det_matr(matr)
    for i in range(n):
        for j in range(n):
            matr_inv[j][i] = c * alg_add(matr, i, j)
    return matr_inv


def gauss_method(a, b):
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            c = - a[j][i] / a[i][i]
            for k in range(i, n):
                if k == i:
                    a[j][k] = 0
                else:
                    a[j][k] += c * a[i][k]
            b[j] += c * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(n - 1, i, -1):
            x[i] -= x[j] * a[i][j]
        x[i] /= a[i][i]
    return x


def func(p):
    n = len(p)
    p = [1] + p
    return lambda x: (-1) ** n * sum([x ** i * p[n - i] * (1 if i == n else -1) for i in range(n, -1, -1)])


def div_half_method(a, b, f):
    s = 0.1
    d = 0.0001
    res = []
    x_last = a
    x = x_last
    while x <= b:
        x = x_last + s
        if f(x) * f(x_last) < 0:
            x_left = x_last
            x_right = x
            x_mid = (x + x_last) / 2
            while abs(f(x_mid)) >= d:
                if f(x_left) * f(x_mid) < 0:
                    x_right = x_mid
                else:
                    x_left = x_mid
                x_mid = (x_left + x_right) / 2
            res.append(x_mid)
        x_last = x
    return res


def gershgorin_rounds(a):
    left = -100000
    right = 100000
    n = len(a)
    for i in range(n):
        s = 0
        for j in range(n):
            if i != j:
                s += abs(a[i][i])
        b1 = a[i][i] - s
        b2 = a[i][i] + s
        if i == 0:
            left = b1
            right = b2
        elif b1 < left:
            left = b1
        elif b2 > right:
            right = b2
    return [left, right]


def danilevsky_method(a):
    n = len(a)
    m = n - 1

    b = [[0] * n for i in range(n)]
    for i in range(n):
        b[i][i] = 1
    for j in range(n):
        if j != m - 1:
            b[m - 1][j] = -a[m][j] / a[m][m - 1]
    b[m - 1][m - 1] = 1 / a[m][m - 1]

    b_mul = copy.deepcopy(b)

    c = [[0] * n for i in range(n)]
    for i in range(n):
        c[i][m - 1] = a[i][m - 1] * b[m - 1][m - 1]
    for i in range(n - 1):
        for j in range(n):
            if j != m - 1:
                c[i][j] = a[i][j] + a[i][m - 1] * b[m - 1][j]

    b_inv = [[0] * n for i in range(n)]
    for i in range(n):
        b_inv[i][i] = 1
    for j in range(n):
        b_inv[m - 1][j] = a[m][j]

    d = [[0] * n for i in range(n)]
    for i in range(m - 1):
        for j in range(n):
            d[i][j] = c[i][j]
    for j in range(n):
        for k in range(n):
            d[m - 1][j] += a[m][k] * c[k][j]
    d[m][m - 1] = 1

    for k in range(2, n):
        b = [[0] * n for i in range(n)]
        for i in range(n):
            b[i][i] = 1
        for j in range(n):
            if j != m - k:
                b[m - k][j] = -d[m - k + 1][j] / d[m - k + 1][m - k]
        b[m - k][m - k] = 1 / d[m - k + 1][m - k]

        b_mul = mult_matr_matr(b_mul, b)

        b_inv = inv_matr(b)
        d = mult_matr_matr(b_inv, d)
        d = mult_matr_matr(d, b)

    b = copy.deepcopy(b_mul)
    p = d[0][:]
    g = gershgorin_rounds(a)

    ls = div_half_method(g[0], g[1], func(p))

    vectors = []
    for l in ls:
        y = [1]
        for i in range(1, n):
            y.append(l ** i)
        y = y[::-1]
        y = mult_matr_vec(b, y)
        norm = norm_vec(y)
        for i in range(n):
            y[i] /= norm
        vectors.append(y)
    return ls, vectors


def krylov_method(a):
    n = len(a)
    y = [[0] * n for _ in range(n + 1)]
    y[0][0] = 1
    for i in range(1, n + 1):
        y[i] = mult_matr_vec(a, y[i - 1])
    m = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            m[i][j] = y[n - 1 - j][i]
    p = gauss_method(m, y[n])
    g = gershgorin_rounds(a)
    ls = div_half_method(g[0], g[1], func(p))
    p = p[::-1]
    q = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j == 0:
                q[j][i] = 1
            else:
                q[j][i] = ls[i] * q[j - 1][i] - p[n - j]
    x = []
    for i in range(n):
        xi = [0] * n
        for j in range(n):
            xi = sum_vec(xi, mult_vec_num(q[j][i], y[n - 1 - j]))
        x.append(xi)
    vectors = []
    for xi in x:
        norm = norm_vec(xi)
        for i in range(n):
            xi[i] /= norm
        vectors.append(xi)
    return ls, vectors


def qr_decomposition(matrix):
    q = []
    r = [[0] * len(matrix[0]) for _ in range(len(matrix[0]))]

    for j in range(len(matrix[0])):
        v = matrix[j]
        for i in range(len(q)):
            rij = scalar_mult_vec(q[i], matrix[j])
            r[i][j] = rij
            v = sub_vec(v, mult_vec_num(rij, q[i]))
        r[j][j] = math.sqrt(scalar_mult_vec(v, v))
        q.append(mult_vec_num(1 / r[j][j], v))

    return q, r


def svd_decomposition_eigenvalues_danilevsky(a):
    a_transp_a = mult_matr_matr(transp_matr(a), a)

    eigenvalues, eigenvectors = danilevsky_method(a_transp_a)

    singular_values = [math.sqrt(eigenvalue) for eigenvalue in eigenvalues]

    left_singular_vectors = mult_matr_matr(a, eigenvectors)
    for i in range(len(left_singular_vectors)):
        for j in range(len(left_singular_vectors[0])):
            left_singular_vectors[i][j] /= singular_values[j]

    right_singular_vectors = eigenvectors

    return left_singular_vectors, singular_values, transp_matr(right_singular_vectors)


def svd_decomposition_eigenvalues_krylov(a):
    a_transp_a = mult_matr_matr(transp_matr(a), a)

    eigenvalues, eigenvectors = krylov_method(a_transp_a)

    singular_values = [math.sqrt(eigenvalue) for eigenvalue in eigenvalues]

    left_singular_vectors = mult_matr_matr(a, eigenvectors)
    for i in range(len(left_singular_vectors)):
        for j in range(len(left_singular_vectors[0])):
            left_singular_vectors[i][j] /= singular_values[j]

    right_singular_vectors = eigenvectors

    return left_singular_vectors, singular_values, transp_matr(right_singular_vectors)


def svd_decomposition_qr(matrix, num_iterations=100):
    m, n = len(matrix), len(matrix[0])

    u = [[random.random() for _ in range(m)] for _ in range(m)]
    v = [[random.random() for _ in range(n)] for _ in range(n)]

    for _ in range(num_iterations):
        u, _ = qr_decomposition(mult_matr_matr(matrix, v))
        v, _ = qr_decomposition(mult_matr_matr(transp_matr(matrix), u))
    sigma = mult_matr_matr(transp_matr(u), mult_matr_matr(matrix, v))
    return u, sigma, transp_matr(v)


def power_iteration(matrix, num_iterations):
    b_k = np.random.rand(matrix.shape[1])
    for _ in range(num_iterations):
        b_k1 = np.dot(matrix, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k


def find_eigvals_and_vecs(matrix, num_eigenvalues, num_iterations):
    eigenvalues = []
    eigenvectors = []
    matrix1 = copy.deepcopy(matrix)
    for _ in range(num_eigenvalues):
        eigenvector = power_iteration(matrix1, num_iterations)
        eigenvalue = np.dot(np.dot(eigenvector.T, matrix1), eigenvector)
        np.dot(eigenvector.T, eigenvector)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        matrix1 -= eigenvalue * np.outer(eigenvector, eigenvector)
    return np.array(eigenvalues), np.array(eigenvectors).T


def gram_schmidt_process(vectors):
    orthogonal_vectors = []
    for v in vectors.T:
        for u in orthogonal_vectors:
            v -= np.dot(u, v) * u
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-6:
            v /= v_norm
            orthogonal_vectors.append(v)
    return np.array(orthogonal_vectors).T


def svd_decomposition_iterations(matrix):
    matrix = np.array(matrix)
    n = matrix.shape[0]

    eigvals_A_A_T , eigvecs_A_A_T = find_eigvals_and_vecs(np.dot (matrix, matrix.T), n, 100)
    eigvals_A_T_A , eigvecs_A_T_A = find_eigvals_and_vecs(np.dot (matrix. T, matrix), n, 100)

    sing_vals = np.sqrt(eigvals_A_T_A)
    u = gram_schmidt_process(eigvecs_A_A_T.T).T
    s = np.zeros((n, n))
    np.fill_diagonal(s, sing_vals)
    vt = np.zeros((n, n))
    for i, sing_val in enumerate(sing_vals):
        vt[i] = 1 / sing_val * np.dot(matrix.T, eigvecs_A_A_T.T[i])
    return u, s, vt


matrix = generate_symm_matrix(5, 0, 5)
print("Matrix:")
print_matr(matrix)
print()

u, s, vt = svd_decomposition_eigenvalues_danilevsky(matrix)
# u, s, vt = svd_decomposition_eigenvalues_krylov(matrix)

sigma = [[0] * len(s) for _ in range(len(s))]
for i in range(len(s)):
    sigma[i][i] = s[i]

print("Result:")
print("U:")
print_matr(u)
print()
print("Sigma:")
print_matr(sigma)
print()
print("V*:")
print_matr(vt)
print()

res = mult_matr_matr(mult_matr_matr(u, sigma), vt)
print("Result after multiplication")
print_matr(res)
print()

print("Error rate")
print(norm_matr(sub_matr_matr(matrix, res)))
print()


# matrix = generate_matrix(5, 5, 0, 5)
# print("Matrix:")
# print_matr(matrix)
# print()
#
# u, sigma, vt = svd_decomposition_qr(matrix)
#
# print("Result:")
# print("U:")
# print_matr(u)
# print()
# print("Sigma:")
# print_matr(sigma)
# print()
# print("V*:")
# print_matr(vt)
# print()
#
# res = mult_matr_matr(mult_matr_matr(u, sigma), vt)
# print("Result after multiplication")
# print_matr(res)
# print()
#
# print("Error rate")
# print(norm_matr(sub_matr_matr(matrix, res)))
# print()


# matrix = generate_matrix(5, 5, 0, 5)
# print("Matrix:")
# print_matr(matrix)
# print()
#
# u, sigma, vt = svd_decomposition_iterations(matrix)
#
# print("Result:")
# print("U:")
# print_matr(u)
# print()
# print("Sigma:")
# print_matr(sigma)
# print()
# print("V*:")
# print_matr(vt)
# print()
#
# res = mult_matr_matr(mult_matr_matr(u, sigma), vt)
# print("Result after multiplication")
# print_matr(res)
# print()
#
# print("Error rate")
# print(norm_matr(sub_matr_matr(matrix, res)))
# print()

