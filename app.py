import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from functions import apply_mcar, apply_mar, apply_mnar, generate_synthetic_dataset
from pandas.plotting import parallel_coordinates
import numpy as np
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title='Missingness Prototype', layout='wide')

st.title('Mapping the Missing: Synthetic Data Exploration Tool')

st.markdown("""This interactive tool is designed to help you understand and visualise patterns of missing data in 
datasets. Whether you're a data scientist, a student, or just curious about data analysis, this tool will assist you 
in exploring various missingness mechanisms and their impacts on data.""")

with st.expander("Click here for instructions on how to use this tool"):
    st.markdown("""#### How to Use This Tool:

1. **Upload Your Dataset**:
   - Use the sidebar to upload a CSV file. Please ensure your file is properly formatted with headers for each column.
   - If you do not have a dataset ready, you can use the example dataset provided by clicking on the link in the sidebar.

2. **Select Missingness Type**:
   - Choose one of the three types of missingness mechanisms to apply to your data:
     - **MCAR (Missing Completely at Random):** Data is missing randomly, and the missingness does not depend on any other data points.
     - **MAR (Missing at Random):** Data is missing in a way that relates to the observed data but not the missing data itself.
     - **MNAR (Missing Not at Random):** Missingness is related to the unobserved data, meaning the reason data is missing could depend on the missing data itself.

3. **Specify the Percentage of Missingness**:
   - Adjust the slider to set the percentage of the data you wish to be missing according to the selected mechanism.

4. **Choose the Start Column for Missingness**:
   - Select a specific column or 'All Columns' to apply the missingness. This selection defines where the missingness starts or is applied across all columns.

5. **Generate Missingness**:
   - Click the 'Generate Missingness' button to apply the selected missingness type and percentage to the uploaded or example dataset.

6. **Visualise Missingness Patterns**:
   - After generating missingness, choose from various visualisation methods to explore the missing data patterns:
     - **Heatmap:** Visualize the distribution and frequency of missing values across your dataset.
     - **Bar Chart**: Quantify and compare the number of missing values across different variables.
     - **Box Plots**: Assess the central tendency and dispersion of data, which can reveal the impact of missingness on the dataset's distribution.
     - **Correlation Matrix**: Understand how missingness may be influencing the relationships between variables by observing changes in correlation patterns.
     - **Parallel Coordinates**: Detect patterns and potential correlations across multiple dimensions of the data, aiding in the detection of MNAR (Missing Not at Random) data patterns.

7. **Customise Your View**:
   - Use the multiselect dropdown to choose which visualisations you wish to see displayed below the dataset. You can select or deselect visualisations according to your needs.

#### Tips for Better Insights:
- Analyse the impact of different missingness types on your data analysis.
- Use the visualisations to identify patterns that might indicate biases or potential issues in data collection or handling processes.
- Experiment with different percentages of missingness to see how robust your data analysis methods are against the varying degrees of incompleteness.

Thank you for using the Missingness Prototype. Explore, visualise, and gain deeper insights into your data's missingness patterns!
""")

with st.expander('Learn About Synthetic Data'):
    st.markdown("""Synthetic Data Generation In addition to handling missing data, this tool provides the 
    functionality to generate synthetic data. Synthetic data is artificially created information that closely mirrors 
    the statistical properties of a real dataset without containing any actual sensitive information. This is 
    especially useful in situations where privacy is a concern, or the original data cannot be shared due to 
    confidentiality constraints. It allows for robust testing of data pipelines, machine learning models, 
    and statistical analyses in a secure environment. 

Generating synthetic data can help in several ways:

- Privacy Protection: It protects individual privacy, as the synthetic dataset does not include real individual 
records. 

- Data Availability: Synthetic datasets can be shared freely, helping overcome barriers due to data privacy 
regulations. 

- Model Training: In machine learning, synthetic data can be used to augment real datasets, improving the 
performance of models, especially in cases where the original data may be imbalanced or sparse. 

- Testing and Development: Developers and analysts can use synthetic data to test their systems, algorithms, or analysis workflows 
without the risk of data misuse or exposure. The included synthetic data generation feature allows you to create a 
full dataset based on the statistical characteristics of either the uploaded dataset or a sample dataset provided. 
You can then apply missingness to this synthetic data, just as you would with real data, enabling comprehensive 
analysis and testing in a secure and responsible manner.""")


def apply_missingness(df, start_col, selected_type, selected_percentage):
    if selected_type == 'MCAR':
        if start_col == all_columns_option:
            for column in df.columns:
                df = apply_mcar(df, column, selected_percentage)
        else:
            df = apply_mcar(df, start_col, selected_percentage)
    elif selected_type == 'MAR':
        condition = lambda x: x > 3.0
        condition_column = 'sepal.length'
        if start_col == all_columns_option:
            for column in df.columns:
                if column != condition_column:
                    df = apply_mar(df, condition_column, condition, column, selected_percentage)
        else:
            df = apply_mar(df, condition_column, condition, start_col, selected_percentage)
    elif selected_type == 'MNAR':
        prob_missing = selected_percentage
        prob_continuation = 0.8
        if start_col == all_columns_option:
            for column in df.columns:
                df = apply_mnar(df, column, prob_missing, prob_continuation)
        else:
            df = apply_mnar(df, start_col, prob_missing, prob_continuation)
    return df


def visualize_data(df):
    visualisation_options = ['Heatmap', 'Bar Chart', 'Box Plots', 'Correlation Matrix', 'Parallel Coordinates']
    selected_visualisations = st.multiselect('Select Visualization Methods:', visualisation_options, default=visualisation_options)

    if 'Heatmap' in selected_visualisations:
        fig = px.imshow(df.isnull(), color_continuous_scale='blues')
        st.plotly_chart(fig)

    if 'Bar Chart' in selected_visualisations:
        missing_values = df.isnull().sum()
        fig = px.bar(x=missing_values.index, y=missing_values.values, labels={'x': 'Column', 'y': 'Number of Missing Values'})
        st.plotly_chart(fig)

    if 'Box Plots' in selected_visualisations:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fig = px.box(df, y=numeric_cols)
        st.plotly_chart(fig)

    if 'Correlation Matrix' in selected_visualisations:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='blues', labels=dict(color="Correlation Coefficient"))
        fig.update_xaxes(side="bottom")
        fig.update_layout(margin=dict(l=0), autosize=False)
        st.plotly_chart(fig, use_container_width=True)

    if 'Parallel Coordinates' in selected_visualisations:
        # Ensure that the column for coloring is categorical
        if 'variety' in df.columns and df['variety'].dtype != 'category':
            df['variety'] = df['variety'].astype('category')

        # Check if the 'variety' column exists and is categorical for safety
        if 'variety' in df.columns and df['variety'].dtype == 'category':
            plt.figure(figsize=(8, 4))
            parallel_coordinates(df, 'variety', colormap=plt.get_cmap("Set2"))  # Using a built-in colormap
            plt.title('Parallel Coordinates Plot')
            plt.grid(True)
            plt.legend(title='Class Label')
            st.pyplot(plt)
            plt.clf()


st.sidebar.header('User Input Features')
option = st.sidebar.radio("Pick One:", ["Generate missingness into existing dataset", "Generate fully synthetic dataset"])

uploaded_file = st.sidebar.file_uploader("Upload your CSV file, or use the [Default CSV ]("
                                         "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

if option == "Generate fully synthetic dataset" and df is not None:
    df = generate_synthetic_dataset(df)

if df is not None:
    all_columns_option = "All Columns"
    start_col = st.sidebar.selectbox('Start Column for Missingness', [all_columns_option] + list(df.columns))
    selected_type = st.sidebar.selectbox('Select Missingness Type', ['MCAR', 'MAR', 'MNAR'])
    selected_percentage = st.sidebar.slider('Percentage of Missingness', 0, 100, 25) / 100.0

    if st.sidebar.button('Generate Missingness'):
        df = apply_missingness(df, start_col, selected_type, selected_percentage)
        st.session_state['modified_df'] = df
        st.experimental_rerun()

    # Use the modified dataframe from session state
    df_to_display = st.session_state.get('modified_df', df)
    total_rows = df_to_display.shape[0]
    total_missing = df_to_display.isnull().sum().sum()
    st.markdown(f"**Total Rows in Dataset:** {total_rows}")
    st.markdown(f"**Total Missing Values Across Dataset:** {total_missing}")
    st.write('Modified DataFrame:', df_to_display)

    # Visualization part
    visualize_data(df_to_display)
