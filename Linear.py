from helpers import *

import matplotlib.pyplot as plt


# Source code is here https://github.com/Stonepaw90/int-point-alg


# Thanks to stackoverflow user Pierre D for the many functions to graph the feasible region, inequalities
# https://stackoverflow.com/a/65344728

# Thanks to github user PablocFonseca for putting AgGrid in streamlit
# https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108

# Thanks to streamlit creator ash2shukla for how to write a nice table
# https://discuss.streamlit.io/t/questions-on-st-table/6878/3

# Big thanks to my brother Ben for helping me code the columns in the detailed numeric output
# https://github.com/TheBengineer


# Thanks to Dr. Michael Veatch for polishing this with me, over and over again

class LinearProgram:
    def __init__(self):
        self.constraint_slider = False
        self.variable_dict = {"advanced": False, "update 11.26": False, "standard": False, "done": False,
                              "ex 11.7": False}


    def print_intro(self):
        st.title("Interior Point Algorithm for Linear Programs")
        st.markdown('''### Coded by [Abraham Holleran](https://github.com/Stonepaw90) :sunglasses:''')
        st.write(
            "This website uses [Algorithm 11.3](https://www.wiley.com/go/veatch/convexandlinearoptimization) (Primal-dual path following) to solve a linear program in canonical form."
            " If the problem is entered in standard form, it is converted to canonical form. For two-variable problems, the feasible region and solutions are graphed.")
        st.header("Standard and canonical form notation")
        st.markdown("The canonical form problem has $m$ constraints and $n$ variables:")
        col = st.columns(2)
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
            "The standard form problem has $m_s$ constraints and $n_s$ variables. Call the $m_s \\times n_s$ coefficient matrix $\\bar{{A}}$, etc.:")
        st.latex(r"""\begin{aligned}
            &\text{max } \bar{c}^T\bar{x}& \\
            &\text{s.t.  } \bar{A}\bar{x} \leq b & \\
            &\bar{x} \geq 0& \end{aligned}""")
        st.write(
            "When converted to canonical form, the constraints are $\\bar{A}\\bar{x}+s=b$. Here $s$ contains $m_s$ slack variables and $w$ contains $n = m_s + n_s$ dual surplus variables. "
            "Strict feasibility for the primal requires $\\bar{x}>0, s>0$. Strict feasibility for the dual requires $w > 0$.")
        st.write("Enter the problem, strictly feasible initial solutions, and parameters.")

    def update_attributes(self):
        self.variable_dict['standard'] = st.toggle("Standard form", value=False)
        if self.variable_dict["standard"]:
            self.variable_dict["ex 11.7"] = st.toggle("Load Example 11.7", value=False)
        if self.variable_dict["ex 11.7"]:
            self.m_s, self.n_s = 3, 2
            self.A = A_DEFAULT
            self.b = b_DEFAULT
            self.c = c_DEFAULT
            self.x = x_DEFAULT
            self.y = y_DEFAULT
        else:
            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                self.m_s = st.number_input(r'$m$: The number of constraints.', min_value=1, value=3)
            with col2:
                self.n_s = st.number_input(r'$n$: The number of variables.', min_value=1, value=2)
            st.markdown(
                        "In canonical form, this program cannot handle redundant constraints. Please ensure that the number of rows $m$ $\leq$ the number of columns $n$.")

            if not self.variable_dict["standard"]:
                if not self.m_s <= self.n_s:
                    st.error(f"In canonical form, ensure $m \leq n$.")

            self.A = pd.DataFrame(np.zeros((self.m_s, self.n_s)), columns=[f'Var{i + 1}' for i in range(self.n_s)],
                                  index=[f'Con{i + 1}' for i in range(self.m_s)])
            self.b = pd.DataFrame(np.zeros((self.m_s, 1)), columns=['Value'],
                                  index=[f'Con{i + 1}' for i in range(self.m_s)])
            self.c = pd.DataFrame(np.zeros((self.n_s, 1)), columns=['Value'],
                                  index=[f'Var{i + 1}' for i in range(self.n_s)])
            self.x = pd.DataFrame(np.zeros((self.n_s, 1)), columns=['Value'],
                                  index=[f'Var{i + 1}' for i in range(self.n_s)])
            self.y = pd.DataFrame(np.zeros((self.m_s, 1)), columns=['Value'],
                                  index=[f'Con{i + 1}' for i in range(self.m_s)])

    def get_parameters(self):
        st.subheader(f'Enter matrix $A$ ({self.m_s} x {self.n_s}):')
        st.text(f'Please enter a matrix of size {self.m_s}x{self.n_s} representing the constraints.')
        self.A = st.data_editor(self.format_dataframe(self.A), key='matrix_A', hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f'Enter right-hand side vector $b$ for {self.m_s} constraints:')
            self.b = st.data_editor(self.format_dataframe(self.b), key='vector_b', hide_index=True)

            st.subheader(f'Enter initial solution $x$ for {self.n_s} variables:')
            self.x = st.data_editor(self.format_dataframe(self.x), key='initial_solution_x', hide_index=True)

        with col2:
            st.subheader(f'Enter objective function coefficients $c$ for {self.n_s} variables:')
            self.c = st.data_editor(self.format_dataframe(self.c), key='vector_c', hide_index=True)

            st.subheader(f'Enter initial solution $y$ for {self.m_s} constraints:')
            self.y = st.data_editor(self.format_dataframe(self.y), key='initial_solution_y', hide_index=True)
        self.flatten_values()

    def flatten_values(self):
        self.A = np.array(self.A)
        self.b = self.b.values.flatten()
        self.x = self.x.values.flatten()
        self.y = self.y.values.flatten()
        self.c = self.c.values.flatten()

    @staticmethod
    def format_dataframe(df):
        """Format DataFrame for display in Streamlit."""
        df_copy = df.copy()
        df_copy.index = [''] * len(df_copy)  # Hide index by replacing with empty strings
        return df_copy

    def get_additional_parameters(self):
        if self.m_s > 0 and self.n_s > 0:
            if not self.variable_dict['standard']:
                if self.m_s > self.n_s:
                    st.markdown("This program cannot handle redundant constraints in canonical form. "
                                "Please enter a coefficient matrix with the number of rows $\leq$ the number of columns.")
                    st.stop()
            st.write("You entered")
            col = st.columns(2)
            with col[0]:
                st.latex("A = " + sympy.latex(sympy.Matrix(self.A)))
            if self.variable_dict["standard"]:
                self.n_full = self.n_s + self.m_s
                self.n_plot = self.n_s
            else:
                self.n_full = self.n_s
                self.n_plot = self.n_s - self.m_s
            st.header("Parameters")
            col = st.columns(2)
            with col[0]:
                st.write(r"""$\alpha$: Step size parameter.""")
                self.alpha = st.number_input(r"$\alpha$", value=0.9, step=0.01, min_value=0.0, max_value=0.999,
                                        help=r"""Ensures each variable is reduced by no more than a factor of $1 - \alpha$. $\hspace{13px} 0 < \alpha < 1$""")
                st.write("""$\gamma$: Duality gap parameter.""")
                self.gamma = st.number_input(r"$\gamma$", value=0.25, step=0.01,
                                        help=r"""The complimentary slackness parameter $\mu$ is multiplied by $\gamma$ each iteration such that $\mu \rightarrow 0$. $\hspace{13px} 0 < \gamma < 1$""")
            with col[1]:
                st.write("""$\epsilon$: Optimality tolerance.""")
                self.epsilon = st.number_input(r"$\epsilon$", value=0.01, step=0.001, format="%f", min_value=0.00001,
                                          help=r"""Stop the algorithm once $x^Tw< \epsilon$. $\hspace{13px} \epsilon > 0$""")
                st.write("""$\mu$: Initial complementary slackness parameter.""")
                self.mu = st.number_input(r"$\mu$", value=5.0, step=0.1, help=r"""$\mu > 0$""")  # 0.25
            self.variable_dict["done"] = st.checkbox("Solve")

    def solve(self):
        if self.variable_dict["done"]:
            if self.variable_dict["standard"]:
                st.header("After converting to canonical form, the data and initial solutions are:")
                self.s = self.b - self.A.dot(self.x)
                self.A = np.concatenate((self.A, np.identity(self.m_s)), axis=1)
                self.x = np.concatenate((self.x, self.s))
                self.c = np.concatenate((self.c, np.zeros(self.m_s)))
                #except:
                #st.write("The given vectors have incorrect dimensions.")
                #st.stop()
            else:
                st.header("The data and initial solutions are:")
                #self.x = self.x
                #self.c_full = self.c
                try:
                    if isinstance(self.A.dot(self.x) - self.b, str):
                        pass
                except:
                    st.write("The given vectors have incorrect dimensions.")
                    st.stop()
                ax = self.A.dot(self.x)
                if any([abs(i) > 0.001 for i in (ax - self.b)]):
                    st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(ax.round(4)) + f"\\neq" + lt(self.b))
                    st.stop()
            try:
                self.w = self.A.T.dot(self.y) - self.c
            except:
                st.write("The given vectors have incorrect dimensions.")
                st.stop()
            self.w_initial = list(self.w)
            self.x_initial = list(self.x)
            self.y_initial = list(self.y)
            self.mu_initial = self.mu / 2
            if self.variable_dict["standard"]:
                st.latex("A = " + sympy.latex(sympy.Matrix(self.A)))
            col = st.columns(5)
            col_helper1 = 0
            var = [sympy.Matrix(i.round(4)) for i in [self.b, self.c, self.w, self.x, self.y]]
            names = ["b", "c", "w", "x", "y"]
            for i in range(5):
                with col[col_helper1 % 5]:
                    st.latex(names[i] + "=" + sympy.latex(var[i]))
                    col_helper1 += 1
            if is_neg(self.x, False):
                st.markdown("Error: $x < 0$.")
                st.stop()
            if is_neg(self.w):
                st.markdown("Error: $w\leq0$.")
                st.stop()
            try:
                self.f = self.x.dot(self.c)
            except:
                st.write("The given vectors have incorrect dimensions.")
                st.stop()
            self.variable_dict["update 11.26"] = st.checkbox("Use (11.26) to update mu?", value=False)
            if self.variable_dict["update 11.26"]:
                st.markdown(f"The method for computing $\mu$ is Equation (11.26).")
            else:
                st.markdown("The method for updating $\mu$ each iteration is $\mu^{new} = \gamma \mu$.")


    def run_iterations(self):
        iter = 0
        data = []

        if self.variable_dict["done"]:
            self.variable_dict['advanced'] = st.checkbox("Show slacks and dual values", value=False)
            mu_e = "{:2.1E}".format(self.mu)

            if self.variable_dict["advanced"]:
                if self.variable_dict["standard"]:
                    data.append(round_list(
                        [iter, mu_e, self.x.dot(self.w), self.f, self.x[:self.n_s], self.s, self.y, self.w],
                        make_tuple=True))
                    alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "s", "y", "w"]
                else:
                    data.append(round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x, self.y, self.w],
                                           make_tuple=True))
                    alist = ["k", "mu", "Gap x^Tw", "Objective", "x", "y", "w"]
            else:
                if self.variable_dict["standard"]:
                    data.append(round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x[:self.n_s]],
                                           make_tuple=True))
                    alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]
                else:
                    data.append(round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x], make_tuple=True))
                    alist = ["k", "mu", "Gap x^Tw", "Objective", "x"]

            while np.dot(self.x, self.w) >= self.epsilon:
                diagx = np.diagflat(self.x)
                diagw = np.diagflat(self.w)
                diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape(
                    (self.n_full, self.n_full))
                vmu = self.mu * np.ones(self.n_full) - diagx.dot(diagw).dot(np.ones(self.n_full))

                try:
                    #self.A = np.array(self.A)
                    dy = np.linalg.inv(
                        self.A.dot(
                            diagx).dot(
                            diagwinv).dot(
                            self.A.T)
                    ).dot(self.A).dot(diagwinv).dot(vmu)
                except:
                    st.latex("AXW^{-1}A^T \\text{ Could not be inverted. This may be due to redundant constraints.}")
                    st.stop()

                dw = self.A.T.dot(dy)
                dx = diagwinv.dot(vmu - diagx.dot(dw))
                betap = min(1, min([self.alpha * j for j in
                                    [-self.x[i] / dx[i] if dx[i] < 0 else 100 for i in range(self.n_full)]]))
                betad = min(1, min([self.alpha * j for j in
                                    [-self.w[i] / dw[i] if dw[i] < 0 else 100 for i in range(self.n_full)]]))

                self.x += betap * dx
                self.y += betad * dy
                self.w += betad * dw

                if self.variable_dict["update 11.26"]:
                    self.mu = self.gamma * self.x.dot(self.w) / (self.m_s + self.n_s)
                else:
                    self.mu *= self.gamma

                mu_e = "{:2.1E}".format(self.mu)
                iter += 1
                self.f = self.x.dot(self.c)
                ax = self.A.dot(self.x)

                if any([abs(i) > 0.001 for i in (ax - self.b)]):
                    st.latex(f"Ax \\neq b, \hspace{{8px}} " + lt(ax.round(6)) + f"\\neq" + lt(self.b))
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
                    st.stop()

                if self.variable_dict["advanced"]:
                    if self.variable_dict["standard"]:
                        data.append(round_list(
                            [iter, mu_e, self.x.dot(self.w), self.f, self.x[:self.n_s], self.s, self.y,
                             self.w], make_tuple=True))
                    else:
                        data.append(
                            round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x, self.y, self.w],
                                       make_tuple=True))
                else:
                    if self.variable_dict["standard"]:
                        data.append(round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x[:self.n_s]],
                                               make_tuple=True))
                    else:
                        data.append(
                            round_list([iter, mu_e, self.x.dot(self.w), self.f, self.x], make_tuple=True))

                if iter >= 15:
                    st.write(
                        "The program terminated, as after 15 iterations, the duality gap was still more than epsilon.")
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
            st.markdown(
                "Note: In this table the $\mu$ in a row is used to compute the next row, while Table 11.2 reports $\mu$ in the row it was used to compute.")
            self.df = df

    def plot_iterations(self):
        if self.variable_dict["done"]:
            if self.n_plot == 2:
                make_plot = st.checkbox("Graph feasible region and iterations.")
                if make_plot:
                    col = st.columns(2)
                    with col[0]:
                        plot_space = st.empty()
                    with col[1]:
                        boundaries = st.container()
                        legend_print = st.empty()
                        if self.constraint_slider:
                            slider = st.empty()

            w = np.array(self.w_initial)
            x = np.array(self.x_initial)
            y = np.array(self.y_initial)
            mu = self.mu_initial * 2
            f = self.x.dot(self.c)
            iter = 0
            st.write("Detailed output of all iterations is below.")
            st.write("# ")
            st.write("""---""")
            st.write("# ")

            while iter < len(self.df):
                diagx = np.diagflat(x)
                diagw = np.diagflat(w)
                diagwinv = np.array([1 / i if i != 0 else 0 for i in np.nditer(diagw)]).reshape(
                    (self.n_full, self.n_full))
                vmu = mu * np.ones(self.n_full) - diagx.dot(diagw).dot(np.ones(self.n_full))
                dy = np.linalg.inv(self.A.dot(diagx).dot(diagwinv).dot(self.A.T)).dot(
                    self.A).dot(diagwinv).dot(vmu)
                dw = self.A.T.dot(dy)
                dx = diagwinv.dot(vmu - diagx.dot(dw))

                matrix_string = ["X", "W", "XW^{-1}", None, None, "v(\\mu)", "A", "AXW^{-1}A^T", "d^x", "d^y", "d^w"]
                complicated_eq = self.A.dot(diagx).dot(diagwinv).dot(self.A.T)
                matrix_list = [np.diagflat([round(i, 4) for i in x]), np.diagflat([round(i, 4) for i in w]),
                               diagx.dot(diagwinv).round(4), None, None, None,
                               self.A.round(4), complicated_eq.round(4), dx.round(4), dy.round(4),
                               dw.round(4)]

                st.markdown("### $k= " + str(iter) + "$")
                col = st.columns(3)
                col_help = 0

                for i in range(len(matrix_string)):
                    if i in [3, 4]:
                        pass
                    elif i == 5:
                        with col[1]:
                            muone = lt((mu * np.ones(self.n_full)).round(4))
                            xwone = lt((diagx.dot(diagw).dot(np.ones(self.n_full))).round(4))
                            vmulatex = lt(vmu.round(4))
                            st.latex("v(\mu) = " + muone + "-" + xwone + "= " + vmulatex)
                        col_help = 0
                    elif i in [0, 1, 2, 6, 7]:
                        with col[col_help % 3]:
                            if i == 6:
                                if self.m_s < 6:
                                    st.latex(
                                        matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                                    col_help += 2
                                else:
                                    with col[1]:
                                        st.latex(
                                            matrix_string[7] + "=" + sympy.latex(sympy.Matrix(complicated_eq.round(4))))
                            elif i == 7:
                                if self.m_s < 6:
                                    st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(
                                        sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                                    col_help += 1
                                else:
                                    with col[1]:
                                        st.latex("(" + matrix_string[7] + ")^{-1}=" + sympy.latex(
                                            sympy.Matrix(np.linalg.inv(complicated_eq).round(4))))
                                    col_help = 0
                            elif self.n_full < 6:
                                st.latex(matrix_string[i] + "=" + diagonal_matrix(matrix_list[i]))
                                col_help += 1
                    else:
                        latex_matrix_string = latex_matrix(matrix_string[i], matrix_list[i], True)
                        col[i % 3].latex(latex_matrix_string)

                    if i == 2:
                        st.write("Details of (11.21):")
                        col = st.columns(3)
                        col_help = 2
                    if i == 5:
                        st.write("Details of (11.23):")
                        col = st.columns(3)
                        col_help = 0
                    if i == 7:
                        st.markdown("Solving for *d*: ")
                        col = st.columns(3)
                        col_help = 0

                st.write("The step sizes are")
                optionp = min(
                    [self.alpha * j for j in [-x[i] / dx[i] if dx[i] < 0 else 100 for i in range(self.n_full)]])
                optiond = min(
                    [self.alpha * j for j in [-w[i] / dw[i] if dw[i] < 0 else 100 for i in range(self.n_full)]])

                x_r = [round(i, 4) for i in x]
                dx_r = [round(i, 4) for i in dx]
                dw_r = [round(i, 4) for i in dw]
                w_r = [round(i, 4) for i in w]
                betap = min(1, optionp)
                betad = min(1, optiond)

                l_string = "\\beta_P = \\text{min}\\{1, 0.9*\\text{min}\\{ "
                for i in range(self.n_full):
                    if dx_r[i] < 0:
                        l_string += "\\frac{" + str(x_r[i]) + "}{" + str(-dx_r[i]) + "},"
                l_string = l_string[
                           :-1] + "\\}\\} = \\text{min}\\{1, " + f"{round(optionp, 4)}" + "\\} = " + f"{round(betap, 4)}"
                st.latex(l_string)

                l_string = "\\beta_D = \\text{min}\\{1, 0.9*\\text{min}\\{ "
                for i in range(self.n_full):
                    if dw_r[i] < 0:
                        l_string += "\\frac{" + str(w_r[i]) + "}{" + str(-dw_r[i]) + "},"
                l_string = l_string[
                           :-1] + "\\}\\} = \\text{min}\\{1, " + f"{round(optiond, 4)}" + "\\} = " + f"{round(betad, 4)}"
                st.latex(l_string)

                col = st.columns(3)
                with col[0]:
                    st.latex(
                        "x^{new} =" + lt(x.round(4)) + "+" + str(round(betap, 4)) + lt(dx.round(4)) + " = " + lt(
                            (x + betap * dx).round(4)))
                with col[1]:
                    st.latex("y^{new} =" + lt(y.round(4)) + "+" + str(round(betad, 4)) + lt(dy.round(4)) + " = " + lt(
                        (y + betad * dy).round(4)))
                with col[2]:
                    st.latex("w^{new} =" + lt(w.round(4)) + "+" + str(round(betad, 4)) + lt(dw.round(4)) + " = " + lt(
                        (w + betad * dw).round(4)))

                x += betap * dx
                y += betad * dy
                w += betad * dw
                if self.variable_dict["update 11.26"]:
                    mu = self.gamma * x.dot(w) / (self.m_s + self.n_s)
                else:
                    mu *= self.gamma
                iter += 1
                st.write("""---""")

            if self.n_plot == 2:
                if make_plot:
                    with boundaries:
                        st.subheader("Enter the plot boundaries:")
                        x1, x2 = st.slider("Select x-axis range", 0.0, 10.0, (0.0, 10.0), key = "xboundaries")
                        y1, y2 = st.slider("Select y-axis range", 0.0, 10.0, (0.0, 10.0), key = "yboundaries")
                    legend_show = legend_print.checkbox("Show legend?", True)

                    try:
                        df = self.df
                        #bbox = [float(i.strip("][").split(" ")[0]) for i in bbox.split(",")]
                        #bbox = [bbox[0:2], bbox[2:]]
                        bbox = [[x1, x2], [y1, y2]]
                        fig = plt.figure(figsize=(7, 3), dpi=80)
                        ax = plt.axes()
                        if self.variable_dict['standard']:
                            plot_inequalities(self.A[:, :2], self.b, bbox, ax=ax)
                        if self.constraint_slider:
                            obj = slider.slider("Objective function value", min_value=0.0,
                                                max_value=round(df['Objective'][len(df) - 1] + 5, 1), step=0.1)
                            if obj > 0:
                                ax.plot([0, obj / c[0]], [obj / c[1], 0], "r-")
                        go = ax.plot(*df['x'][0][:2], 'go', label="Initial point")

                        for i in range(len(df['x']) - 1):
                            bo = ax.plot(*df['x'][i + 1][:2], 'bo', label="Improving Point")
                            ax.plot([df['x'][i][0], df['x'][i + 1][0]], [df['x'][i][1], df['x'][i + 1][1]], 'k-')
                        go = ax.plot(*df['x'][0][:2], 'go', label="Initial point")
                        # ro = ax.plot(*df['x'][i+1][:2], 'ro', label = "Epsilon-optimal Point")
                        legend_l = []
                        for i in range(self.m_s):
                            row_con = self.A[i]
                            legend_l.append(constraint_string(row_con, self.b[i]))
                        if self.constraint_slider:
                            if obj > 0:
                                legend_l.append(constraint_string(self.c[:2], obj))
                        legend_l.append("Initial")
                        # legend_l.append("Improving")
                        # legend_l.append("Epsilon-optimal")
                        plt.xlabel("x")
                        plt.ylabel("y")
                        if legend_show:
                            if self.variable_dict['ex 11.7']:
                                ax.legend(legend_l, loc="upper right")
                            else:
                                ax.legend(legend_l)
                        plot_space.pyplot(fig)
                    except:
                        plot_space.header("Plotting failed.")


    def run(self):
        self.print_intro()
        self.update_attributes()
        self.get_parameters()
        self.get_additional_parameters()
        self.solve()
        self.run_iterations()
        self.plot_iterations()


def run():
    lp = LinearProgram()
    lp.run()

if __name__ == '__page__':
    run()