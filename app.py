import streamlit as st
import Linear
import Convex

st.set_page_config(page_title="Linear Interior Point Algorithm", layout="wide",
                   menu_items = {"About": "A companion app to [Linear and Convex Optimization: A Mathematical Approach](https://www.gordon.edu/michaelveatch/optimization) by Michael Veatch.",
                                 "Report a Bug": "mailto:abraham.holleran@gordon.edu"})


delete_page = st.Page("Linear.py", title="Linear", icon=":material/bedtime:")
create_page = st.Page("Convex.py", title="Convex", icon=":material/show_chart:")

pg = st.navigation([delete_page, create_page])
pg.run()

# Sidebar for page selection
#page = st.sidebar.selectbox("Choose a page", ["Linear", "Convex"])

#if page == "Linear":
#    Linear.run()  # Assuming Linear.py has a function `run()`
#elif page == "Convex":
#    Convex.run()  # Assuming Convex.py has a function `run()`
