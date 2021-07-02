#!/usr/bin/env python
# coding: utf-8

# # This is a program that runs the primal dual interior point algorithm given a function and a starting point  (and some other variables)
# ### by Abraham Holleran

import numpy as np
import pandas as pd
import sympy
import math
import streamlit as st
from functools import reduce

st.set_page_config(layout="wide")


variable_dict = {"shortcut": False, "show_symbo": False, "show_numeric": False, "show_all_numeric": False, "feasible": False, "pos": False}

# Carefully put your variables, functions, and constraints here.
#st.sidebar.button("Re Run")
st.sidebar.header("Parameters")
st.sidebar.write(r"""$\alpha$: Step size Multiplier.""")
alpha = st.sidebar.number_input(r"""""", value = 0.8, step=0.01,min_value = 0.0, max_value = 0.999, help = r"""Each dual variable is reduced by no more than a factor of $1 - \alpha$.""")
#st.sidebar.markdown("""---""")
st.sidebar.write(r"$\beta$: Backtracking multiplier.")
beta = st.sidebar.number_input(r"""""", value = 0.9, step=0.01,min_value = 0.0, max_value = 0.999, help = r"If a constraint is violated, the step size is multiplied by $\beta$.")
#st.sidebar.markdown("""---""")
st.sidebar.write("""$\epsilon$: Stopping criterion.""")
epsilon = st.sidebar.number_input(r"""""", value = 0.001, step=0.001, format="%f", min_value = 0.00001, help = r"""Stop the algorithm once $\lambda||d^x|| < \epsilon$.""")
#st.sidebar.markdown("""---""")
st.sidebar.write("""$\gamma$: Duality gap.""")
gamma = st.sidebar.number_input(r"""""", value = 0.1, step=0.01, help = r"""The complimentary slackness constraint violation $\mu$ is multiplied by $\gamma$ each iteration.""")
#st.sidebar.markdown("""---""")
variable_dict["shortcut"] = st.sidebar.checkbox(label="""For example 9: Use ratio test (not backtracking). If the ratio test is used, when the step
 lambda_max violates the constraint, it is reduced to satisfy the contraint with equality.""")
st.title("Primal-dual Interior Point Algorithm for Convex Programs")
st.markdown("### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:")
#st.header("By Abraham Holleran")
#st.write(
#    "Written from the book [Linear and Convex Optimization](https://www.wiley.com/go/veatch/convexandlinearoptimization) under the supervision of the author, Dr. Michael Veatch.")
st.write("Use this website to interact with the Primal-dual Interior Point Algorithm from [Section 15.4](https://www.wiley.com/go/veatch/convexandlinearoptimization) to solve Examples 15.9 and 15.10.")

st.write("Select a problem in the dropdown, then enter the initial conditions below and adjust the parameters on the left. The problem is re-solved after any change. "
         "After solving, you can look at the equations or the numerical steps.")
option = st.selectbox('Which problem do you want to optimize?', ('Example 9 (1 variable)', 'Example 10 (2 variables)'))
if option.split(' ')[1] == "10":
    option = 1
elif option.split(' ')[1] == "9":
    option = 2
if option == 1:
    #st.latex(r'''\text{max } 10 + 10x_1 - 8x_2 - 4e^{x_1}-e^{x_1-x_2}\\ \text{s.t.  } x_2 - x_1^{0.5} \leq 0 \\ -x_2 + x_1^{1.5} \leq 0 ''')
    st.latex(r"""\begin{aligned}
    &\text{max } 10 + 10x_1 - 8x_2 - 4e^{x_1}-e^{x_1-x_2}& \\
    &\text{s.t.  } x_2 - x_1^{0.5} \leq 0& \\
    &-x_2 + x_1^{1.5} \leq 0& \end{aligned}""")
    x1, x2, mu = sympy.symbols('x1 x2 mu', real=True)  # Sympy requires that variables be initiated
    X = sympy.Matrix([x1, x2])
    y1, y2 = sympy.symbols('y1 y2', real=True)
    Y = sympy.Matrix([y1, y2])
    all_vars = sympy.Matrix([x1, x2, y1, y2])
    f, g1, g2 = sympy.symbols('f g1 g2', cls=sympy.Function)  # Sympy requires that functions be initiated
    s1, s2 = sympy.symbols('s1 s2', real=True)  # one s_i for each g_i, b_i
    f = 10 + 10 * x1 - 8 * x2 - 4 * sympy.exp(x1) - sympy.exp(x1 - x2)
    g1 = x2 - x1 ** (0.5)  # Setting up constraints
    g2 = -x2 + x1 ** (1.5)
    g = sympy.Matrix([g1, g2])
    b = sympy.Matrix([0, 0])
    alist = ["k", "mu", "x1", "x2", "y1", "y2", "f(x)", "lambda", "d^x", """l*||d^x||"""]
    st.write("Please write your (feasible) initial point.")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    #while not (variable_dict["feasible"] and variable_dict["pos"]):
    variable_dict["feasible"] = False
    variable_dict["pos"] = False
    col1.write(r"""$x_1$""")
    x1_input = col1.number_input(value = 0.5, label = "", key = "x1")
    col2.write(r"""$x_2$""")
    x2_input = col2.number_input(value = 0.6, label = "", key = "x2")
    col3.write(r"""$y_1 \geq 0$""")
    y1_input = col3.number_input(value = 5.0, label = "", min_value = 0.0, key = "y1")
    col4.write(r"""$y_2 \geq 0$""")
    y2_input = col4.number_input(value = 10.0, label = "", min_value = 0.0, key = "y2")
    col5.write(r"""$\mu > 0$""")
    mu_input = col5.number_input(value = 1.0, label = "", min_value=0.001, key = "mu")
    mu_value = float(mu_input)
    point = [x1_input,x2_input, y1_input, y2_input]


elif option == 2:
    st.latex(r"""\begin{aligned}
    &\text{max  } 10x-e^x& \\
    &\text{s.t.     } x \leq 2&
    \end{aligned} """)
    x, mu = sympy.symbols('x mu', real=True)
    X = sympy.Matrix([x])
    y = sympy.symbols('y', real=True)
    Y = sympy.Matrix([y])
    all_vars = sympy.Matrix([x, y])
    f, g1 = sympy.symbols('f g1 ', cls=sympy.Function)
    s1 = sympy.symbols('s1', real=True)  # one s_i for each g_i, b_i
    f = 10 * x - sympy.exp(x)
    g1 = x
    g = sympy.Matrix([g1])
    b = sympy.Matrix([2])
    alist = ["k", "mu", "x", "y", "f(x)", "lambda", "d^x", "l*||d^x||"]
    st.write("Please write your (feasible) initial point.")
    col11, col12, col13 = st.beta_columns(3)
    point = [1, 1]
    s = [1]
    error_found = not all([i >= 0 for i in s])
    col11.write(r"""$x$""")
    x_input = col11.number_input(value = 1.0, label = "", key = "x")
    col12.write(r"""$y \geq 0$""")
    y_input = col12.number_input(value = 0.5, label = "", key = "y", min_value=0.001)
    col13.write(r"""$\mu > 0$""")
    mu_input = col13.number_input(value = 2.0, label = "", min_value = 0.0, key = "mu2")
    point = [float(x_input), float(y_input)]
    mu_value = float(mu_input)

s = sympy.Matrix([b[i] - g[i].subs([*zip(X, point[:len(X)])]).evalf() for i in range(len(g))])
input_point = point.copy()
assert all([i >= 0 for i in
            s]), f"The initial point does not satisfy the constraints. They have negative slacks [s1, s2] of {[*s]}."
assert all([i >= 0 for i in point[len(Y):]]), f"Choose positive y."

gradient = lambda f, v: sympy.Matrix([f]).jacobian(v)

H = sympy.hessian(f, X)
Z = sympy.zeros(len(X))
thing = [y_i * sympy.hessian(g_i, X) for y_i, g_i in zip(Y, g)]
sum_M = reduce((lambda x, y: x + y), thing)  # This takes the sum of the list "thing"
# sum(thing) doesn't work for sympy.Dense matrices"
Q = -H + sum_M

m = sympy.Matrix([mu / y_i for y_i in Y])
RHSB = b - g - m
J = g.jacobian(X)
RHST = - J.T * Y + gradient(f, X).T
RHS = sympy.Matrix([RHST, RHSB])
st.latex(r'''\text{We need to solve for } \textbf{d}^x, \textbf{d}^y \text{ using (15.14): } \begin{bmatrix}
\textbf{Q} & \textbf{J}(\textbf{x})^T\\
\textbf{J}(\textbf{x}) & -\textbf{S}
\end{bmatrix}\begin{bmatrix}
\textbf{d}^x \\ \textbf{d}^y
\end{bmatrix}
=
\begin{bmatrix}
\nabla f(\textbf{x}) - \textbf{J}(\textbf{x})^T\textbf{y} \\
\textbf{b} - \textbf{g}(\textbf{x})-\textbf{m}
\end{bmatrix} ''')

# The following cell prints \\(\begin{bmatrix}
# \textbf{d}^x \\ \textbf{d}^y
# \end{bmatrix}\\).)
S = sympy.diag(*[(b_i - g_i) / y_i for b_i, g_i, y_i in zip(b, g, Y)])
LHS = sympy.Matrix([[Q, J.T], [J, -S]])

solv = LHS.LUsolve(RHS)
k = 0
done = False
data = []
if option == 2:
    st.write(f"Example 9 only: method for choosing step size is ", "a ratio test." if variable_dict["shortcut"] else "backtracking. (default)")

while not done and k < 14:
    solv_eval = solv.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    f_eval = f.subs([*zip(X, point[:len(X)])])
    l_max1 = min(1, min([y_i / -dy_i if dy_i < 0 else 1 for y_i, dy_i in zip(point[-len(Y):], solv_eval[-len(Y):])]))
    if option == 2 and "shortcut" in variable_dict and variable_dict["shortcut"]:
        l_max = min(l_max1, (2 - point[0]) / solv_eval[0])
    else:
        l_max = l_max1
    assert (all([y_i + l_max * dy_i >= 0 for y_i, dy_i in
                 zip(point[-len(Y):], solv_eval[-len(Y):])])),\
        "Iterations found negative y"
    l = l_max
    all_constraints_satisfied = False
    iter = 0
    while not all_constraints_satisfied:
        violation = False
        iter += 1
        assert iter < 20, "The program is running too many iterations. Try changing your initial point or lowering your alpha or beta parameters."
        test_x = [i + l * j for i, j in zip(point[:len(X)], solv_eval[:len(X)])]
        for g_i, b_i in zip(g, b):
            g_eval = g_i.subs([*zip(all_vars[:len(X)], test_x)])
            if g_eval > b_i:
                violation = True
        if violation:
            l *= beta
        else:
            all_constraints_satisfied = True
    l *= alpha
    dnorm = math.sqrt(sum(map(lambda i: l * i * l * i, solv_eval[:len(X)])))
    mu_scientific = "{:2.1E}".format(mu_value)
    dpowerx = [round(float(io),4) for io in solv_eval[:len(X)]]
    if len(dpowerx) ==1:
        dpowerx = dpowerx[0]
    else:
        dpowerx = tuple(dpowerx)
    value_list = [k, mu_scientific, *[round(float(i), 4) for i in point], round(f_eval, 4), round(l, 4),
                  dpowerx, round(dnorm,4)]
    data.append(value_list)
    point = [i + l * j for i, j in zip(point, solv_eval)]
    mu_value *= gamma
    k += 1
    if dnorm < epsilon:
        done = True

last_list = [k, "-", *[round(float(i), 4) for i in point], round(f.subs([*zip(X, point[:len(X)])]), 4), "-",
                  "-", "-"]
data.append(last_list)
df = pd.DataFrame(data, columns=alist)
st.markdown("""
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
""", unsafe_allow_html=True)
# st.dataframe(df)
st.table(df)
st.write("""We stopped after iteration """, str(k), """ as $\lambda \mid \mid \\textbf{d}^x\mid \mid <
 \epsilon$, indeed, """, str(round(dnorm, 4)),
         """$<$""", str(epsilon), ".")
if st.button("Show equations."):
    columns = st.beta_columns(2)
    col_help2 = 0

    matrix_list = [gradient(f, X).T, H, None, Q, J.T * Y, RHSB]
    matrix_string = ["\\nabla f(\\textbf{x})", "\\nabla^2 f(\\textbf{x}) ", None, "Q",
                     "J(\\textbf{x})^T\\textbf{y}", "\\textbf{b}-\\textbf{g}-\\textbf{m}"]

    for i,j in zip(matrix_list, matrix_string):
        with columns[col_help2 % 2]:
            col_help2 += 1
            if not i:
                for g_var in range(len(g)):
                    underscore = "_"
                    my_string = f"\\nabla^2 g{underscore +str(g_var+1) if option == 1 else str()} (\\textbf{{x}}) ="
                    st.latex(my_string + sympy.latex(sympy.hessian(g[g_var], X)))

            else:
                 st.latex(j + " = " + sympy.latex(i))
    st.write("Equation 15.14 is:")
    if option == 2:
        st.latex(sympy.latex(LHS) + sympy.latex(sympy.Matrix(["d^x", "d_y"])) + "= " + sympy.latex(RHS))
        st.write("With solution:")
        st.latex(sympy.latex(sympy.Matrix(["d^x", "d_y"])) + " = " + sympy.latex(solv))
    else:
        st.latex(
            sympy.latex(LHS) + sympy.latex(sympy.Matrix(["d_1^x", "d_2^x", "d_1^y", "d_2^y"])) + "= " + sympy.latex(
                RHS))

col4, col5 = st.beta_columns(2)

col_help = 0


def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2):
    global col_help
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        if round(matrix_for_me[i], 4)%1 == 0:
            latex_string += str(int(round(matrix_for_me[i], 4))) + " & "
        else:
            latex_string += str(round(matrix_for_me[i],4)) + " & "
        if ((i + 1) % shape_tuple[1] == 0):
            latex_string = latex_string[:-3] + " \\\\ "
    latex_string = latex_string[:-3] + "  \\end{bmatrix}"
    try:
        if col_bool:
            if col_help % 2 == 0:
                with col_use1:
                    st.latex(latex_string)
            else:
                with col_use2:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("Something broke")

    col_help += 1

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

def digit_fix(subs):
    for i,j in enumerate(subs):
        if j%1 == 0:
            subs[i] = int(j)
        else:
            subs[i] = j.round(4)
            if subs[i] < 0.0001 and subs[i] > -0.0001:
                subs[i] = 0
    return(subs)

if st.button("Details of one iteration."):
    st.latex("\\text{We solve (15.14) at the point } (\\textbf{x}, \\textbf{y}) = (\\textbf{x}_0, \\textbf{y}_0).")
    col6, col7 = st.beta_columns(2)
    col_help = 0
    mu_value = mu_input
    point = input_point
    matrix_list = [gradient(f, X).T, H, None, Q, J.T * Y]
    matrix_string = ["\\nabla f(\\textbf{x})", "\\nabla^2 f(\\textbf{x}) ", None, "Q",
                     "J(\\textbf{x})^T\\textbf{y}"]
    for i in range(len(matrix_list)):
        if i == 2:
            for j in range(len(g)):
                g_subs = sympy.hessian(g[j], X).subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
                if option == 1:
                    latex_matrix("\\nabla^2 g_" + str(j + 1) + " (\\textbf{x}) ", g_subs, True, col6, col7)
                else:
                    latex_matrix("\\nabla^2 g(\\textbf{x}) ", g_subs, True, col6, col7)

        else:
            subss = matrix_list[i].subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
            latex_matrix(matrix_string[i], subss, True, col6, col7)
    b_eval = b.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    g_eval = g.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    m_eval = m.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    latex_matrix_sum("\\textbf{b}-\\textbf{g}-\\textbf{m}", b_eval, g_eval, m_eval)
    LHS_subs = LHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    LHS_subs = digit_fix(LHS_subs)
    st.write("The coefficient matrix on the left of (15.14) is")
    st.latex(sympy.latex(LHS_subs))
    st.write("and the right hand side is")
    RHS_subs = RHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    RHS_subs = digit_fix(RHS_subs)
    st.latex(sympy.latex(RHS_subs))
    st.write("The solution to this is:")
    solv_temp = LHS_subs.LUsolve(RHS_subs)
    solv_temp = digit_fix(solv_temp)
    if option == 1:
        st.latex(sympy.latex(sympy.Matrix(["d_1^x", "d_2^x", "d_1^y", "d_2^y"])) + "= " + sympy.latex(solv_temp))
    else:
        st.latex(sympy.latex(sympy.Matrix(["d^x", "d_y"])) + "= " + sympy.latex(solv_temp))

if st.button(f"Details of all remaining {k - 1} iterations."):

    df1 = df.drop(columns=['k', 'l*||d^x||', 'lambda', 'f(x)', "d^x"])
    for index, df_row in df1.iterrows():
        if index == 0:
            mu_value = float(mu_input)
            continue
        mu_value *= gamma
        point = list(df_row[1:])
        st.latex(
            f"\\text{{We solve (15.14) numerically at the next point, }} (\\textbf{{x}}_{index}, \\textbf{{y}}_{index}).")
        col8, col9 = st.beta_columns(2)

        col_help = 0
        matrix_list = [gradient(f, X).T, H, None, Q, J.T * Y]
        matrix_string = ["\\nabla f(\\textbf{x})", "\\nabla^2 f(\\textbf{x}) ", None, "Q",
                         "J(\\textbf{x})^T\\textbf{y}"]
        for i in range(len(matrix_list)):
            if i == 2:
                for j in range(len(g)):
                    g_subs = sympy.hessian(g[j], X).subs([*zip(all_vars, point), (mu, mu_value)]).evalf(4)
                    if option ==1:
                        latex_matrix("\\nabla^2 g_" + str(j + 1) + f" (\\textbf{{x}}) ", g_subs, True, col8,
                                 col9)
                    else:
                        latex_matrix("\\nabla^2 g" + f" (\\textbf{{x}}) ", g_subs, True, col8,
                                     col9)
            else:
                subss = matrix_list[i].subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
                latex_matrix(matrix_string[i], subss, True, col8, col9)
        b_eval = b.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        g_eval = g.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        m_eval = m.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        latex_matrix_sum("\\textbf{b}-\\textbf{g}-\\textbf{m}", b_eval, g_eval, m_eval)



        LHS_subs = LHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        LHS_subs = digit_fix(LHS_subs)
        st.write("The coefficient matrix on the left of (15.14) is")
        st.latex(sympy.latex(LHS_subs))
        st.write("and the right hand side is")
        RHS_subs = RHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        RHS_subs = digit_fix(RHS_subs)
        st.latex(sympy.latex(RHS_subs))
        st.write("The solution to this is:")
        solv_temp = digit_fix(LHS_subs.LUsolve(RHS_subs))
        st.latex(sympy.latex(solv_temp))
        st.markdown("""---""")
