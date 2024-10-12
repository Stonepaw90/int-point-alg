import numpy as np
import pandas as pd
import sympy
import math
import streamlit as st
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def latex_matrix_sum(name, m1, m2, m3):

    m1 = digit_fix(m1)
    m2 = digit_fix(m2)
    m3 = digit_fix(m3)
    latex_string = name + " = " + "\\begin{bmatrix}  ("
    for i in range(len(m1)):
        latex_string += str(round(m1[i],4)) + ") - (" + str(round(m2[i],4)) + ") - (" + str(round(m3[i],4)) + ") \\\\ ("
    latex_string = latex_string[:-1] + " \\end{bmatrix} = \\begin{bmatrix}"
    new_thing = m1 - m2 - m3
    new_thing = digit_fix(new_thing)
    for i in new_thing:
        if i%1 == 0:
            latex_string += str(int(round(i,4))) + " \\\\ "
        else:
            latex_string += str(round(i,4)) + " \\\\ "
    latex_string = latex_string[:-2] + "  \\end{bmatrix}"
    st.latex(latex_string)


def latex_matrix(name, matrix_for_me, vec=True):
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        if vec:
            latex_string += str(matrix_for_me[i]) + " \\\\ "
        else:
            latex_string += str(matrix_for_me[i]) + " & "
        if len(shape_tuple) > 1:
            if ((i + 1) % shape_tuple[1] == 0):
                latex_string = latex_string[:-3] + " \\\\ "
    latex_string = latex_string[:-3] + "  \\end{bmatrix}"
    return latex_string

def digit_fix(subs):
    for i,j in enumerate(subs):
        if j%1 == 0:
            subs[i] = int(j)
        else:
            subs[i] = j.round(4)
            if subs[i] < 0.0001 and subs[i] > -0.0001:
                subs[i] = 0
    return(subs)



def feasible_point(A, b):
    # finds the center of the largest sphere fitting in the convex hull
    norm_vector = np.linalg.norm(A, axis=1)
    A_ = np.hstack((A, norm_vector[:, None]))
    c = np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
    return res.x[:-1]


def hs_intersection(A, b):
    interior_point = feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    return hs


def plt_halfspace(a, b, bbox, ax):
    if a[1] == 0:
        ax.axvline(b / a[0])
    else:
        x = np.linspace(bbox[0][0], bbox[0][1], 100)
        ax.plot(x, (b - a[0] * x) / a[1])


def add_bbox(A, b, xrange, yrange):

    A = np.vstack((A, [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ]))
    b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
    return A, b


def solve_convex_set(A, b, bbox, ax=None):
    A_, b_ = add_bbox(A, b, *bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs


def plot_convex_set(A, b, bbox, ax=None):
    # solve and plot just the convex set (no lines for the inequations)
    points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])
    ax.fill(points[:, 0], points[:, 1], 'gray')
    return points, interior_point, hs


def plot_inequalities(A, b, bbox, ax=None):
    # solve and plot the convex set,
    # the inequation lines, and
    # the interior point that was used for the halfspace intersections
    points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
    # ax.plot(*interior_point, 'o')
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    return points, interior_point, hs



def is_neg(x, strict=True):

    if isinstance(x, np.ndarray):
        if strict:
            return np.any(x <= 0)
        else:
            return np.any(x < 0)

    if strict:
        return any([i <= 0 for i in x])
    else:
        return any([i < 0 for i in x])

def round_vector(vec,n):
    #vec must be a np.ndarray or list
    if type(vec) is np.ndarray:
        return vec.round(n)
    else:
        return [round(i) for i in vec]

def round_list(listc, make_tuple=False):
    #Use only when your list has different things in it.
    #Useful when list is ints, floats, strings, and np.arrays.
    for i in range(len(listc)):
        if type(listc[i]) is str or type(listc[i]) is tuple or type(listc[i]) is None:
            pass
        elif type(listc[i]) is list or type(listc[i]) is np.ndarray:
            listc[i] = round_vector(listc[i], 4)
            if make_tuple:
                if len(listc[i]) > 1:
                    listc[i] = tuple(listc[i])
                else:
                    listc[i] = float(listc[i][0])
        else: #float or int
            listc[i] = round(listc[i], 4)
    return listc


def diagonal_matrix(x):
    string = f"\\begin{{bmatrix}}"
    x_l = len(x)
    for i in range(x_l):
        string = string + "0 &" * i + str(x[i][i]) + "  &  " + "0 & " * (x_l - i - 1)
        string = string[:-3] + "\\\\ "
    string = string + "\\end{bmatrix}"
    return string


def constraint_string(rowc, b_val):
    # rowc is a list. Like [4.0, 3.0, 1, 2]
    # b is the <= list. Like [11, 4]
    # returns 4x + 3y < 11
    if rowc[0] % 1 == 0:
        rowc = [int(rowc[0]), rowc[1]]
    if rowc[1] % 1 == 0:
        rowc = [rowc[0], int(rowc[1])]
    legstring = ""
    if rowc[0] != 0:
        legstring += f"{str() if rowc[0] == 1 else str(rowc[0])}x"
        if rowc[1] > 0:
            legstring += " + " + f"{str() if rowc[1] == 1 else str(rowc[1])}y"
        if rowc[1] < 0:
            legstring += " - " + f"{str() if rowc[1] == 1 else str(-rowc[1])}y"
        if rowc[1] == 0:
            pass
    else:
        legstring += f"{str() if rowc[1] == 1 else str(rowc[1])}y"
    if b_val % 1 == 0:
        legstring += " = " + str(int(b_val))
    else:
        legstring += " = " + str(b_val)
    return legstring

def lt(x):
    return (sympy.latex(sympy.Matrix(x)))

def empty_vec(n):
    pd.DataFrame(np.zeros((n, 1)), columns=['Value'], index=[f'Con{i + 1}' for i in range(n)])


MATRIX_11_7 = np.array([[1.5, 1], [1, 1], [0, 1]])
A_DEFAULT = pd.DataFrame(MATRIX_11_7, index=[f'Con{i + 1}' for i in range(3)],
                  columns=[f'Var{i + 1}' for i in range(2)])
b_DEFAULT = pd.DataFrame([16, 12, 10], columns=['Value'], index=[f'Con{i + 1}' for i in range(3)])
c_DEFAULT = pd.DataFrame([4, 3], columns=['Value'], index=[f'Var{i + 1}' for i in range(2)])
x_DEFAULT = pd.DataFrame([3.0, 3.0], columns=['Value'], index=[f'Var{i + 1}' for i in range(2)])
y_DEFAULT = pd.DataFrame([2.0, 2.0, 1.0], columns=['Value'], index=[f'Con{i + 1}' for i in range(3)])