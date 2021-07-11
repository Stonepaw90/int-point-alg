import streamlit as st
import numpy as np
import pandas as pd
import sympy

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

st.set_page_config(layout="wide")

variable_dict = {"advanced": False, "update 11.26": False, "standard": False, "done": False, "ex 11.7":False}

st.title("Interior Point Algorithm for Linear Programs")
st.markdown('''
### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:
''')
st.write(
    "This website uses [Algorithm 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization) (Primal-dual path following) to solve a linear program in canonical form."
    " If the problem is entered in standard form, it is converted to canonical form.")
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
# Standard
st.markdown("Write your matrix in the top-left of this entry grid. The maximum size is 10 x 10.")
st.write(
    "Press enter or tab to confirm your edits.")
col = st.beta_columns(2)
with col[0]:
    gb = GridOptionsBuilder.from_dataframe(input_dataframe)
    gb.configure_grid_options(
        editable=True,
        sortable=False,
        #filter = False,
        #enableFilter=False,
        resizable=True,
        defaultWidth=width,
        fit_columns_on_grid_load=False,
        key=matrix_key)
    gridOptions = gb.build()
    response = AgGrid(
        input_dataframe,
        height=grid_height,
        width='100%',
        gridOptions = gridOptions,
        #editable=True,
        #sortable=False,
        #enableFilter=False,
        #resizable=True,
        #defaultWidth=width,
        fit_columns_on_grid_load=False,
        key=matrix_key
    )
    #gridOptions = {
    #    floatingFilter: true
    #    columnDefs:
    #    [{
    #        suppressMenu: true,
    #        floatingFilterComponentParams: {suppressFilterButton:true}
    #     }]
    #}
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
    st.write("You entered")
    col = st.beta_columns(2)
    with col[0]:
        st.latex("A = " + sympy.latex(sympy.Matrix(matrix_small)))
    st.write("Enter vectors using spaces between entries, e.g.,\"1 4.1 3 2.0\".")
    col = st.beta_columns(2)
    # col_help = 0
    with col[0]:
        b = np.array([float(i) for i in
                      st.text_input(f"Right-hand side b (a {m_s}-vector)", value=default_var[0]).split(" ") if i]) # 2 1
    if variable_dict["standard"]:
        n_full = n_s + m_s
    else: #n_s is already big!
        n_full = n_s
    with col[1]:
        c = np.array([float(i) for i in
                      st.text_input(f"Objective function coefficients c (a {n_s}-vector)", value=default_var[1]).split(
                          " ") if i]) #1 2 0 0
    st.header("Initial solution")
    col = st.beta_columns(2)
    with col[0]:
        x = np.array([float(i) for i in
                      st.text_input(f"x (a {n_s}-vector)", value=default_var[2]).split(" ") if i]) #1 0.5 0.5 1.5
    with col[1]:
        y = np.array([float(i) for i in
                      st.text_input(f"y (a {m_s}-vector)", value=default_var[3]).split(" ") if i]) #2 0.5
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


def is_neg(x):
    return any([i <= 0 for i in x])

def round_list(list, make_tuple=False):
    for i in range(len(list)):
        if type(list[i]) is str or type(list[i]) is tuple:
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
        if any([abs(i) > 0.001 for i in (matrix_full.dot(x_full) - b)]):
            st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(round_list(matrix_full.dot(x_full))) + f"\\neq" + lt(b))
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
    var = [sympy.Matrix(i) for i in [b, c_full, w, x_full, y]]
    names = ["b", "c", "w", "x", "y"]
    for i in range(5):
        with col[col_helper1 % 5]:
            st.latex(names[i] + "=" + sympy.latex(var[i]))
            col_helper1 += 1
    if is_neg(x_full):
        st.markdown("Error: $x\leq0$.")
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
    variable_dict['advanced'] = st.checkbox("Show slacks and dual values", value=False)

    ###ITERATION 0 ROW
    if variable_dict["advanced"]:
        # IN CANONICAL FORM THERE IS NO S TO PRINT! We're already printing x_full.
        if variable_dict["standard"]: #Standard, advanced
            data.append(round_list([iter, mu, x_full.dot(w), f, x, s, y, w], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "s", "y", "w"]
        else: #Canonical, advanced
            data.append(round_list([iter, mu, x_full.dot(w), f, x_full, y, w], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "y", "w"]
    else:
        if variable_dict["standard"]:  # Not Advanced, and Standard
            data.append(round_list([iter, mu, x_full.dot(w), f, x], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]
        else:  # Not advanced, canonical
            data.append(round_list([iter, mu, x_full.dot(w), f, x_full], make_tuple=True))
            alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]
    while not np.dot(x_full, w) < epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        # diagwinv = np.linalg.inv(diagw)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        try:
            dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        except:
            st.latex("AXW^{-1}A^T \\text{ Could not be inverted. Perhaps your coefficient matrix is singular.}")
            st.stop()
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        betap = min(1, min([alpha * j for j in [-x_full[i] / dx[i] if dx[i] < 0 else 1000 for i in range(n_full)]]))
        betad = min(1, min([alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 1000 for i in range(n_full)]]))
        x_full += betap * dx
        y += betad * dy
        w += betad * dw
        if variable_dict["update 11.26"]:
            mu = gamma * x_full.dot(w) / (m_s + n_s)
        else:
            mu *= gamma

        iter += 1
        f = x_full.dot(c_full)

        if not variable_dict["standard"]:
            if any([abs(i) > 0.001 for i in (matrix_full.dot(x_full) - b)]):
                st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(round_list(matrix_full.dot(x_full))) + f"\\neq" + lt(b))
                #st.latex(f"Ax \\neq b, \hspace{{8px}} {str(*round_list(matrix_full.dot(x_full)))} \\neq {str(*b)}")
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
                st.stop()

        if variable_dict["advanced"]:
            if variable_dict["standard"]: #Advanced, standard
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full[:n_s], s, y, w], make_tuple=True))
            else: #Advanced, canonical
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full, y, w], make_tuple=True))
        else:
            if variable_dict["standard"]:  # Not Advanced, and Standard
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full[:n_s]], make_tuple=True))
            else:  # Not advanced, canonical
                data.append(round_list([iter, mu, x_full.dot(w), f, x_full], make_tuple=True))

        assert iter < 15, "The program terminated, as after 15 iterations, the duality gap was still more than epsilon."
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
    #st.markdown("Note: Unlike table 11.2, in this table, the $\mu$ used in each row was used to compute that row")
    #st.markdown("Note: Unlike table 11.2, the $\mu$ used on each row of this table was used to compute that row.")
    #st.markdown("Note: In this table each row is used to compute the next row, which differs from table 11.2 which places each $\mu$ on the row it computes.")
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





#if st.button("Detailed output of all iterations.") and variable_dict["done"]:
if variable_dict["done"]:
    w = np.array(w_initial)
    x_full = np.array(x_initial)
    y = np.array(y_initial)
    mu = mu_initial*2


    f = x.dot(c)
    # if variable_dict["update 11.26"]:
    #    mu = gamma * np.dot(x, w) / len(x)
    # else:
    #    mu = 5
    iter = 0
    st.write("Detailed output of all iterations is below.")
    st.write("# ")
    st.write("""---""")
    st.write("# ")
    while not np.dot(x_full, w) < epsilon:
        diagx = np.diagflat(x_full)
        diagw = np.diagflat(w)
        diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape((n_full, n_full))
        vmu = mu * np.ones(n_full) - diagx.dot(diagw).dot(np.ones(n_full))
        dy = np.linalg.inv(matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)).dot(matrix_full).dot(diagwinv).dot(vmu)
        dw = matrix_full.T.dot(dy)
        dx = diagwinv.dot(vmu - diagx.dot(dw))
        #matrix_string = ["\\mathbf{X}", "\\mathbf{W}", "\\mathbf{X}\\mathbf{W}^{-1}",
        #                 "\mu\\mathbf{1}", "\\mathbf{XW1}", "\\mathbf{v}(\\mu)",
        #                 "A", "\\mathbf{AX}\\mathbf{W}^{-1}\\mathbf{A}^T",
        #                 "\\mathbf{d}^x", "\\mathbf{d}^y", "\\mathbf{d}^w"]
        matrix_string = ["X", "W", "XW^{-1}",
                         "\mu1", "XW1", "v(\\mu)",
                         "A", "AXW^{-1}A^T",
                         "d^x", "d^y", "d^w"]
        complicated_eq = matrix_full.dot(diagx).dot(diagwinv).dot(matrix_full.T)
        matrix_list = round_list([np.diagflat([round(i, 4) for i in x_full]), np.diagflat([round(i, 4) for i in w]),
                                  diagx.dot(diagwinv).round(4),
                                  mu * np.ones(n_full), diagx.dot(diagw).dot(np.ones(n_full)), vmu,
                                  matrix_full, complicated_eq, dx, dy, dw], False)
        st.markdown("### $k= "+str(iter) + "$")
        col = st.beta_columns(3)
        for i in range(len(matrix_string)):
            # col_help += 1

            if i in [0, 1, 2, 6, 7]:
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
        l_string = "\\beta_P = \\text{min}\\{1, 0.9*\\text{min}\\} "
        empty = True
        for i in range(n_full):
            if dx_r[i] < 0:
                empty = False
                l_string += "\\frac{" + str(x_r[i]) + "}{" + str(-dx_r[i]) + "},"
        if empty:
            l_string = l_string[:-3] + "\\{ \\} "
        l_string = l_string[:-1] + "\\} = \\text{min}\\{1, " + f"{round(optionp, 4)}" + "\\} = " + f"{round(betap, 4)}"
        st.latex(l_string)
        l_string = "\\beta_D = \\text{min}\\{1, 0.9*\\text{min}\\} "
        empty = True
        for i in range(n_full):
            if dw_r[i] < 0:
                empty = False
                l_string += "\\frac{" + str(w_r[i]) + "}{" + str(-dw_r[i]) + "},"
        if empty:
            l_string = l_string[:-3] + "\\{ \\} "
        l_string = l_string[:-1] + "\\} = \\text{min}\\{1, " + f"{round(optiond, 4)}" + "\\} = " + f"{round(betad, 4)}"
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
