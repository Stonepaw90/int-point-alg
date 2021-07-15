import streamlit as st
import numpy as np
import pandas as pd
import sympy

from st_aggrid import AgGrid#, DataReturnMode, GridUpdateMode, GridOptionsBuilder

import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog


#Source code is here https://github.com/Stonepaw90/int-point-alg




#Thanks to stackoverflow user Pierre D for the many functions to graph the feasible region, inequalities
#https://stackoverflow.com/a/65344728

#Thanks to github user PablocFonseca for putting AgGrid in streamlit
#https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108

#Thanks to streamlit creator ash2shukla for how to write a nice table
#https://discuss.streamlit.io/t/questions-on-st-table/6878/3

#Big thanks to my brother Ben for helping me code the columns in the detailed numeric output
#https://github.com/TheBengineer


#Thanks to Dr. Michael Veatch for polishing this with me, over and over again




constraint_slider = False #Feature I don't want but want to be able to toggle
st.set_page_config(page_title = "Linear Interior Point Algorithm", layout="wide")

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
        ax.plot(x, (b - a[0]*x) / a[1])

def add_bbox(A, b, xrange, yrange):
    A = np.vstack((A, [
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1],
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
    #ax.plot(*interior_point, 'o')
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    return points, interior_point, hs

variable_dict = {"advanced": False, "update 11.26": False, "standard": False, "done": False, "ex 11.7":False}

st.title("Interior Point Algorithm for Linear Programs")
st.markdown('''
### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:
''')
st.write(
    "This website uses [Algorithm 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization) (Primal-dual path following) to solve a linear program in canonical form."
    " If the problem is entered in standard form, it is converted to canonical form. For two-variable problems, the feasible region and solutions are graphed.")
st.header("Standard and canonical form notation")
st.markdown("The canonical form problem has $m$ constraints and $n$ variables:")
col = st.beta_columns(2)
with col[0]:
    st.write("Primal")
    st.latex(r"""\begin{aligned}
    &\text{max } c^Tx& \\
    &\text{s.t.  } Ax = b & \\
    &x \geq 0& \end{aligned}""")
with col[1]:
    st.write("Dual")
    st.latex(r"""\begin{aligned}
    &\text{min } b^Ty& \\
    &\text{s.t.  } A^Ty -w = c & \\
    &w \geq 0& \end{aligned}""")
st.markdown(
    "The standard form problem has $m$ constraints and $n^′$ variables. Call the $m \\times n^′$ coefficient matrix $\\bar{{A}}$ , etc.:")
st.latex(r"""\begin{aligned}
    &\text{max } \bar{c}^T\bar{x}& \\
    &\text{s.t.  } \bar{A}\bar{x} \leq b & \\
    &\bar{x} \geq 0& \end{aligned}""")
st.write(
    "When converted to canonical form, the constraints are $\\bar{A}\\bar{x}+s=b$. Here $s$ contains $m$ slack variables and $w$ contains $n = m + n^′$ dual surplus variables. "
    "Strict feasibility for the primal requires $\\bar{x}>0, s>0$. Strict feasibility for the dual requires $w > 0$.")
st.write(
    "Enter the problem, strictly feasible initial solutions, and parameters.")
def lt(x):
    return (sympy.latex(sympy.Matrix(x)))

# You can choose if standard or not standard.
variable_dict['standard'] = st.checkbox("Standard form", value=False)
if variable_dict["standard"]:
    variable_dict["ex 11.7"] = st.checkbox("Load Example 11.7", value=False)
if variable_dict["ex 11.7"]:
    matrix_small = np.array([[1.5, 1], [1, 1], [0, 1]])
    input_dataframe = pd.DataFrame(matrix_small, index=[str(i + 1) for i in range(3)], columns=[str(i + 1) for i in range(2)])
    default_var = ["16 12 10", "4 3", "3.0 3.0", "2.0 2.0 1.0"]
    width = 35
    grid_height = 335/2.2
    matrix_key = "11.7"
    col_n = 2
else:
    default_var = ["","","",""]
    input_dataframe = pd.DataFrame('', index=[str(i + 1) for i in range(10)], columns=[str(i + 1) for i in range(10)])
    width = 15
    matrix_key = "not"
    grid_height = 335
    col_n = 1

st.markdown("## Coefficient matrix A")
st.markdown("Write your matrix in the top-left of this entry grid. The maximum size is 10 x 10.")
st.write(
    "Press enter or tab to confirm your edits.")
col = st.beta_columns(2)
with col[0]:
    response = AgGrid(
        input_dataframe,
        height=grid_height,
        width='100%',
        suppressMenu = True, #This line removes the filter
        editable=True,
        #filter = False,
        sortable=False,
        resizable=True,
        fit_columns_on_grid_load=False,
        key=matrix_key
    )
# Convert Matrix, catching errors. Errors lead to a stop that prints out the matrix and your matrix shape (m_s, n_s).

try:
    messy_matrix = response['data'].replace("nan", "")
    messy_matrix.replace(to_replace="", value=np.nan, inplace=True)
    messy_matrix = messy_matrix.dropna(axis=1, how='all')
    messy_matrix = messy_matrix.dropna(axis=0, how='all')
    matrix_small = np.array(messy_matrix, dtype=float)
    m_s = matrix_small.shape[0]
    n_s = matrix_small.shape[1]
    # If the matrix contains any NaNs, move to the except clause.
    assert not np.isnan(matrix_small).any()
except:  # If any errors with conversion
    st.write("Something is wrong with your matrix. ")
    try:  # Try to give diagnostics
        st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
    except:
        pass
    st.write(" Please ensure dimensions and entries are correct and submit again.")
    st.stop()  # Nice exit of program, with no giant red errors.
if m_s > 0 and n_s > 0:
    # Only get here if no errors. #Non-zero matrix, with no errors! Yay!
    if not variable_dict['standard']:
        if m_s > n_s:
            st.markdown("This program cannot handle redundant constraints in canonical form. "
                        "Please enter a coefficient matrix with the number of rows $\leq$ the number of columns.")
            st.stop()
    st.write("You entered")
    col = st.beta_columns(2)
    with col[0]:
        st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
    st.write("Enter vectors using spaces between entries, e.g., \"1 4.1 3 2.0\".")
    col = st.beta_columns(2)
    # col_help = 0
    with col[0]:
        try:
            b = np.array([float(i) for i in
                      st.text_input(f"Right-hand side b (a {m_s}-vector)", value=default_var[0]).split(" ") if i]) # 2 1
        except:
            st.write("Enter vectors using spaces between entries, e.g., \"1 4.1 3 2.0\".")
            st.stop()
    if variable_dict["standard"]:
        n_full = n_s + m_s
        n_plot = n_s
    else: #n_s is already big!
        n_full = n_s
        n_plot = n_s - m_s
    with col[1]:
        try:
            c = np.array([float(i) for i in
                      st.text_input(f"Objective function coefficients c (a {n_s}-vector)", value=default_var[1]).split(
                          " ") if i]) #1 2 0 0
        except:
            st.write("Enter vectors using spaces between entries, e.g., \"1 4.1 3 2.0\".")
            st.stop()
    st.header("Initial solution")
    col = st.beta_columns(2)
    with col[0]:
        try:
            x = np.array([float(i) for i in
                      st.text_input(f"x (a {n_s}-vector)", value=default_var[2]).split(" ") if i]) #1 0.5 0.5 1.5
        except:
            st.write("Enter vectors using spaces between entries, e.g., \"1 4.1 3 2.0\".")
            st.stop()
    with col[1]:
        try:
            y = np.array([float(i) for i in
                      st.text_input(f"y (a {m_s}-vector)", value=default_var[3]).split(" ") if i]) #2 0.5
        except:
            st.write("Enter vectors using spaces between entries, e.g., \"1 4.1 3 2.0\".")
            st.stop()
    st.header("Parameters")
    col = st.beta_columns(2)
    with col[0]:
        st.write(r"""$\alpha$: Step size parameter.""")
        alpha = st.number_input(r"""""", value=0.9, step=0.01, min_value=0.0, max_value=0.999,
                                help=r"""Ensures each variable is reduced by no more than a factor of $1 - \alpha$. $\hspace{13px} 0 < \alpha < 1$""")
        st.write("""$\gamma$: Duality gap parameter.""")
        gamma = st.number_input(r"""""", value=0.25, step=0.01,
                                help=r"""The complimentary slackness parameter $\mu$ is multiplied by $\gamma$ each iteration such that $\mu \rightarrow 0$. $\hspace{13px} 0 < \gamma < 1$""")
    with col[1]:
        st.write("""$\epsilon$: Optimality tolerance.""")
        epsilon = st.number_input(r"""""", value=0.01, step=0.001, format="%f", min_value=0.00001,
                                  help=r"""Stop the algorithm once $x^Tw< \epsilon$. $\hspace{13px} \epsilon > 0$""")
        st.write("""$\mu$: Initial complementary slackness parameter.""")
        mu = st.number_input("", value=5.0, step=0.1, help = r"""$\mu > 0$""") #0.25
    variable_dict["done"] = st.checkbox("Solve")


def is_neg(x, strict = True):
    if not strict:
        return any([i < 0 for i in x])
    else:
        return any([i <= 0 for i in x])

def round_list(list, make_tuple=False):
    for i in range(len(list)):
        if type(list[i]) is str or type(list[i]) is tuple or type(list[i]) == type(None):
            pass
        elif type(list[i]) is list or type(list[i]) is np.ndarray:
            try:
                for j in range(len(list[i])):
                    list[i][j] = round(list[i][j], 4)
                if make_tuple:
                    if len(list[i]) > 1:
                        list[i] = tuple(list[i])
                    else:
                        list[i] = float(list[i][0])
            except:
                pass
        else:
            list[i] = round(list[i], 4)
    return list


if variable_dict["done"]:  #Once solve is pressed
    # Always run! Ex 11.7, standard, canonical, this is always run.
    # By this point, crucially, our data has been marked as correct. We should still check this.

    if variable_dict["standard"]:
        st.header("After converting to canonical form, the data and initial solutions are:")
        try:
            s = b - matrix_small.dot(x)
            matrix_full = np.concatenate((matrix_small, np.identity(m_s)), axis=1)
            x_full = np.concatenate((x, s))
            c_full = np.concatenate((c, np.zeros(m_s)))
        except:
            st.write("The given vectors have incorrect dimensions.")
            st.stop()
    else:
        st.header("The data and initial solutions are:")
        matrix_full = matrix_small
        x_full = x
        c_full = c
        try:
            if type(matrix_full.dot(x_full) - b) is str:
                pass #Why this code? It checks that matrix_full, x_full, and b are the right size.
        except:
            st.write("The given vectors have incorrect dimensions.")
            st.stop()
        ax = matrix_full.dot(x_full)
        if any([abs(i) > 0.001 for i in (ax - b)]):
            st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(round_list(ax)) + f"\\neq" + lt(b))
            st.stop()
            # matrix_full = np.concatenate((matrix_small, np.identity(m_s)), axis=1)
            # x_full = np.concatenate((x,s))
            # c_full = np.concatenate((c, np.zeros(m_s)))
    try:
        w = matrix_full.T.dot(y) - c_full
    except:
        st.write("The given vectors have incorrect dimensions.")
        st.stop()
    #I don't know why this was so difficult! I'm saving these initial values for later.
    w_initial = list(w)
    x_initial = list(x_full)
    y_initial = list(y)
    mu_initial = mu/2 #This is not an error, as mu_initial will be doubled later
    if variable_dict["standard"]:
        st.latex("A = " + sympy.latex(sympy.Matrix(matrix_full)))
    col = st.beta_columns(5)
    col_helper1 = 0
    var = [sympy.Matrix(round_list(i)) for i in [b, c_full, w, x_full, y]]
    names = ["b", "c", "w", "x", "y"]
    for i in range(5):
        with col[col_helper1 % 5]:
            st.latex(names[i] + "=" + sympy.latex(var[i]))
            col_helper1 += 1
    if is_neg(x_full, False):
        st.markdown("Error: $x < 0$.")
        st.stop()
    if is_neg(w):
        st.markdown("Error: $w\leq0$.")
        st.stop()
    try:
        f = x_full.dot(c_full)
    except:
        st.write("The given vectors have incorrect dimensions.")
        st.stop()
    variable_dict["update 11.26"] = st.checkbox("Use (11.26) to update mu?", value=False)
    if variable_dict["update 11.26"]:
        st.markdown(f"The method for computing $\mu$ is Equation (11.26).")
    else:
        st.markdown("The method for updating $\mu$ each iteration is $\mu^{new} = \gamma \mu$.")
    # if variable_dict["update 11.26"]:
    #    mu = gamma*np.dot(x,w)/len(x)
    # st.write("mu=",mu)
    # elif variable_dict["ex 11.7"]:
    #    mu = 5
iter = 0
data = []



if variable_dict["done"]:  # All branches get here, once data has been verified.
    variable_dict['advanced'] = st.checkbox("Show slacks and dual values", value=True)
    mu_e = "{:2.1E}".format(mu)
    ###ITERATION 0 ROW
    if variable_dict["advanced"]:
        # IN CANONICAL FORM THERE IS NO S TO PRINT! We're already printing x_full.
        if variable_dict["standard"]: #Standard, advanced
            data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full[:n_s], s, y, w], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "s", "y", "w"]
        else: #Canonical, advanced
            data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full, y, w], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "y", "w"]
    else:
        if variable_dict["standard"]:  # Not Advanced, and Standard
            data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full[:n_s]], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]
        else:  # Not advanced, canonical
            data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]
    while np.dot(x_full, w) >= epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        # diagwinv = np.linalg.inv(diagw)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        try:
            dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        except:
            st.latex("AXW^{-1}A^T \\text{ Could not be inverted. This may be due to redundant constraints.}")
            st.stop()
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        betap = min(1, min([alpha * j for j in [-x_full[i] / dx[i] if dx[i] < 0 else 100 for i in range(n_full)]]))
        betad = min(1, min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 100 for i in range(n_full)]]))
        
        x_full += betap * dx
        y += betad * dy
        w += betad * dw
        if variable_dict["update 11.26"]:
            mu = gamma * x_full.dot(w) / (m_s + n_s)
        else:
            mu *= gamma
        mu_e = "{:2.1E}".format(mu)
        iter += 1
        f = x_full.dot(c_full)
        ax = matrix_full.dot(x_full)
        #if not variable_dict["standard"]:
            #if any([abs(i) > 0.001 for i in (ax - b)]):
            #    st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(round_list(ax)) + f"\\neq" + lt(b))
            #    df = pd.DataFrame(data, columns=alist)
            #    st.markdown("""
            #        <style>
            #        table td:nth-child(1) {
            #            display: none
            #        }
            #        table th:nth-child(1) {
            #            display: none
            #        }
            #        </style>
            #        """, unsafe_allow_html=True)
            #    # st.dataframe(df)
            #    st.table(df)
            #    st.stop()
        st.write(matrix_full.dot(x_full))
        if variable_dict["advanced"]:
            if variable_dict["standard"]: #Advanced, standard
                data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full[:n_s], s, y, w], make_tuple=True))                
            else: #Advanced, canonical
                data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full, y, w], make_tuple=True))
        else:
            if variable_dict["standard"]:  # Not Advanced, and Standard
                data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full[:n_s]], make_tuple=True))
            else:  # Not advanced, canonical
                data.append(round_list([iter, mu_e, x_full.dot(w), f, x_full], make_tuple=True))

                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
        if iter >= 15:
            st.write("The program terminated, as after 15 iterations, the duality gap was still more than epsilon.")
            break
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
    st.table(df)
    st.markdown("Note: In this table the $\mu$ in a row is used to compute the next row, while Table 11.2 reports $\mu$ in the row it was used to compute.")
    col_help = 0


def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec=True):
    global col_help
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
    try:
        if col_bool:
            if col_help % 3 == 0:
                with col_use1:
                    st.latex(latex_string)
            elif col_help % 3 == 1:
                with col_use2:
                    st.latex(latex_string)
            else:
                with col_use3:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("Something broke while trying to print a matrix.")
    col_help += 1


def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2, col_use3, vec=True):
    global col_help
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
    try:
        if col_bool:
            if col_help % 3 == 0:
                with col_use1:
                    st.latex(latex_string)
            elif col_help % 3 == 1:
                with col_use2:
                    st.latex(latex_string)
            else:
                with col_use3:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("Something broke while trying to print a matrix.")
    col_help += 1


def diagonal_matrix(x):
    string = f"\\begin{{bmatrix}}"
    x_l = len(x)
    for i in range(x_l):
        string = string + "0 &" * i + str(x[i][i]) + "  &  " + "0 & " * (x_l - i - 1)
        string = string[:-3] + "\\\\ "
    string = string + "\\end{bmatrix}"
    return string


def digit_fix(subs):
    for i, j in enumerate(subs):
        if j % 1 == 0:
            subs[i] = int(j)
        else:
            subs[i] = j.round(4)
            if subs[i] < 0.0001 and subs[i] > -0.0001:
                subs[i] = 0
    return (subs)

def constraint_string(rowc, b_val):
    #rowc is a list. Like [4.0, 3.0, 1, 2]
    #b is the <= list. Like [11, 4]
    #returns 4x + 3y < 11
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


if variable_dict["done"]:
    if n_plot == 2:
        make_plot = st.checkbox("Graph feasible region and iterations.")
        if make_plot:
            col = st.beta_columns(2)
            with col[0]:
                plot_space = st.empty()
            with col[1]:
                boundaries = st.empty()
                legend_print = st.empty()
                if constraint_slider:
                    slider = st.empty()
    w = np.array(w_initial)
    x_full = np.array(x_initial)
    y = np.array(y_initial)
    mu = mu_initial*2
    f = x.dot(c)
    iter = 0
    st.write("Detailed output of all iterations is below.")
    st.write("# ")
    st.write("""---""")
    st.write("# ")
    while np.dot(x_full, w) >= epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        matrix_string = ["X", "W", "XW^{-1}",
                         None, None, "v(\\mu)",
                         "A", "AXW^{-1}A^T",
                         "d^x", "d^y", "d^w"]
        complicated_eq = matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)
        matrix_list = round_list([np.diagflat([round(i, 4) for i in x_full]), np.diagflat([round(i, 4) for i in w]),
                                  diagx.dot(diagwinv).round(4),
                                  #mu * np.ones(n_full), diagx.dot(diagw).dot(np.ones(n_full)), vmu,
                                  None, None, None,
                                  matrix_full, complicated_eq, dx, dy, dw], False)
        st.markdown("### $k= "+str(iter) + "$")
        col = st.beta_columns(3)
        for i in range(len(matrix_string)):
            # col_help += 1
            if i in [3,4]:
                pass
            elif i == 5:
                with col[1]:
                    muone = lt(round_list(mu * np.ones(n_full)))
                    xwone = lt(round_list(diagx.dot(diagw).dot(np.ones(n_full))))
                    vmulatex = lt(round_list(vmu))
                    st.latex("v(\mu) = " + muone + "-"
                             + xwone + "= " + vmulatex)
                col_help = 0
            elif i in [0, 1, 2, 6, 7]:
                with col[col_help % 3]:
                    if i == 6:
                        if m_s < 6:
                            st.latex(matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                            col_help += 2
                        else:
                            with col[1]:
                                st.latex(matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                    elif i == 7:
                        if m_s < 6:
                            st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(
                                sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                            col_help += 1
                        else:
                            with col[1]:
                                st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(
                                    sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                            col_help = 0
                    elif n_full < 6: #Happens with i == 0,1,2
                        st.latex(matrix_string[i] + "=" + diagonal_matrix(matrix_list[i]))
                        col_help += 1
            else:
                latex_matrix(matrix_string[i], matrix_list[i], True, col[0], col[1], col[2])
            if i == 2:
                st.write("Details of (11.21):")
                col = st.beta_columns(3)
                col_help = 2
            if i == 5:
                st.write("Details of (11.23):")
                col = st.beta_columns(3)
                col_help = 0
            if i == 7:
                st.markdown("Solving for *d*:")
                col = st.beta_columns(3)
                col_help = 0

        st.write("The step sizes are")
        optionp = min([alpha * j for j in [-x_full[i] / dx[i] if dx[i] < 0 else 100 for i in range(n_full)]])
        optiond = min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 100 for i in range(n_full)]])
        x_r = [round(i, 4) for i in x_full]
        dx_r = [round(i, 4) for i in dx]
        dw_r = [round(i, 4) for i in dw]
        w_r = [round(i, 4) for i in w]
        betap = min(1, optionp)
        betad = min(1, optiond)

        #betap
        l_string = "\\beta_P = \\text{min}\\{1, 0.9*\\text{min}\\{ "
        for i in range(n_full):
            if dx_r[i] < 0:
                #empty = False
                l_string += "\\frac{" + str(x_r[i]) + "}{" + str(-dx_r[i]) + "},"
        l_string = l_string[:-1] + "\\}\\} = \\text{min}\\{1, " + f"{round(optionp, 4)}" + "\\} = " + f"{round(betap, 4)}"
        st.latex(l_string)

        #betad
        l_string = "\\beta_D = \\text{min}\\{1, 0.9*\\text{min}\\{ "
        for i in range(n_full):
            if dw_r[i] < 0:
                l_string += "\\frac{" + str(w_r[i]) + "}{" + str(-dw_r[i]) + "},"
        l_string = l_string[:-1] + "\\}\\} = \\text{min}\\{1, " + f"{round(optiond, 4)}" + "\\} = " + f"{round(betad, 4)}"
        st.latex(l_string)
        col = st.beta_columns(3)
        with col[0]:
            st.latex("x^{new} =" + lt(digit_fix(x_full)) + "+" + str(round(betap, 4)) + lt(digit_fix(dx)) + " = " + lt(
                digit_fix(x_full + betap * dx)))
        with col[1]:
            st.latex("y^{new} =" + lt(digit_fix(y)) + "+" + str(round(betad, 4)) + lt(digit_fix(dy)) + " = " + lt(
                digit_fix(y + betad * dy)))
        with col[2]:
            st.latex("w^{new} =" + lt(digit_fix(w)) + "+" + str(round(betad, 4)) + lt(digit_fix(dw)) + " = " + lt(
                digit_fix(w + betad * dw)))
        x_full += betap * dx
        y += betad * dy
        w += betad * dw
        mu *= gamma
        iter += 1
        st.write("""---""")
        assert iter <= len(df), "Too many iterations"

    if n_plot == 2:
        if make_plot:
            bbox = boundaries.text_input("Plot area [x1, x2], [y1, y2]", value = "[0,10],[0,10]")
            legend_show = legend_print.checkbox("Show legend?", True)

            try:
                bbox = [float(i.strip("][").split(" ")[0]) for i in bbox.split(",")]
                bbox = [bbox[0:2], bbox[2:]]
                fig = plt.figure(figsize=(7,3), dpi = 80)
                ax = plt.axes()
                if variable_dict['standard']:
                    plot_inequalities(matrix_small, b, bbox, ax=ax)
                else:
                    plot_inequalities(matrix_small[:,:2], b, bbox, ax=ax)
                if constraint_slider:
                    obj = slider.slider("Objective function value", min_value=0.0,
                                        max_value=round(df['Objective'][len(df) - 1] + 5, 1), step=0.1)
                    if obj > 0:
                        ax.plot([0, obj / c[0]], [obj / c[1], 0], "r-")
                go = ax.plot(*df['x'][0][:2], 'go', label = "Initial point")


                for i in range(len(df['x'])-1):
                    bo = ax.plot(*df['x'][i+1][:2], 'bo', label = "Improving Point")
                    ax.plot([df['x'][i][0],df['x'][i+1][0]],[df['x'][i][1],df['x'][i+1][1]], 'k-')
                go = ax.plot(*df['x'][0][:2], 'go', label = "Initial point")
                #ro = ax.plot(*df['x'][i+1][:2], 'ro', label = "Epsilon-optimal Point")
                legend_l = []
                for i in range(m_s):
                    row_con = matrix_small[i]
                    legend_l.append(constraint_string(row_con, b[i]))
                if constraint_slider:
                    if obj > 0:
                        legend_l.append(constraint_string(c[:2], obj))
                legend_l.append("Initial")
                #legend_l.append("Improving")
                #legend_l.append("Epsilon-optimal")
                plt.xlabel("x")
                plt.ylabel("y")
                if legend_show:
                    if variable_dict['ex 11.7']:
                        ax.legend(legend_l, loc = "upper right")
                    else:
                        ax.legend(legend_l)
                plot_space.pyplot(fig)
            except:
                plot_space.header("Plotting failed.")
st.write(iter)
st.write(np.dot(x_full, w))
