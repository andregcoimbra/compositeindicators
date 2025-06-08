from utils import (normalizar_dados, BOD_Calculation, 
                   Entropy_Calculation, EqualWeights, 
                   PCA_Calculation, Minimal_Uncertainty)
import plotly.express as px
import streamlit as st
import pandas as pd
import io

version = "v1.0"

data = pd.DataFrame()
ranking_ic = []

st.set_page_config(
    page_title="S-CI-MaxS",
    page_icon="📉"
)

st.title('📉 Software for building composite indicators with maximum stability')
st.header("Calculate composite indicators. Methods: PCA, BoD, Equal Weights, and Shannon's Entropy")

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"]::after {{
        content: "{version}";
        position: absolute;
        bottom: 10px;
        left: 16px;
        font-size: 0.8em;
        color: gray;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Carregar arquivo Excel
uploaded_file = st.sidebar.file_uploader("Select Excel file", type=["xlsx"])

# Verifique se o arquivo foi carregado
if uploaded_file is not None:
    # Carregar o arquivo Excel em um DataFrame
    df = pd.read_excel(uploaded_file)

    data_missing = df.isnull().sum()
    
    if data_missing.any():
        missing_columns = data_missing[data_missing > 0]
        missing_info = [f"{col}: {count} missing" for col, count in missing_columns.items()]
        st.error(f"Error: Data missing in the following columns: {', '.join(missing_info)}.")
        st.stop()
    
    if len(df) > 300:
        df = df.iloc[:300]
        st.warning("The file has been trimmed to use only the first 300 rows of data.")
    
    # Exibir as primeiras linhas do arquivo
    st.subheader("Data Preview")
    st.dataframe(df.head(), hide_index=True)

    # Selecionar colunas
    number_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns", 
                                              number_columns,
                                              help="Select the columns to be used in the calculation of composite indicators. At least one column must be selected.")

    # Selecionar variável de controle
    control_variable = st.sidebar.selectbox("Select the control variable", 
                                            ["Choose an option"] + number_columns,
                                            help="""Select a control variable to normalize the data.
                                            Min-Max normalization will be minimum-oriented if the correlation is greater than zero; otherwise, it will be maximum-oriented.
                                            If no control variable is selected, minimum-oriented Min-Max normalization will be applied by default.""")

    # Selecionar colunas
    string_columns = df.columns.tolist()
    labels_column = st.sidebar.selectbox("Select label column", 
                                         ["Choose an option"] + string_columns,
                                         help="""Select a column to use as labels for the rows.
                                         If no column is selected, the rows will be labeled as 'DMU 1', 'DMU 2', etc.""")

    # Botão
    calculate_button = st.sidebar.button("Calculate")

    # Novo bloco: campos para min/max de cada coluna selecionada
    st.sidebar.markdown("---")
    with st.sidebar.expander("Setup BoD: Expert Opinion"):
        if selected_columns:
            column_min_max_BoD = {}
            for col in selected_columns:
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.markdown("**"+col+"**")
                min_value = col2.number_input(
                    label="**Min**",
                    value=0.0,
                    format="%.4f",
                    key=f"min_{col}"
                )
                max_value = col3.number_input(
                    label="**Max**",
                    value=1.0,
                    format="%.4f",
                    key=f"max_{col}"
                )
                column_min_max_BoD[col] = (min_value, max_value)
        else:
            column_min_max_BoD = {}
        
        with st.sidebar.expander("Setup Minimal Uncertainty: Expert Opinion"):
            if selected_columns:
                column_min_max_MU = {}
                for col in selected_columns:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    col1.markdown("**"+col+"**")
                    min_value = col2.number_input(
                        label="**Min**",
                        value=0.0,
                        format="%.4f",
                        key=f"min_MI_{col}"
                    )
                    max_value = col3.number_input(
                        label="**Max**",
                        value=1.0,
                        format="%.4f",
                        key=f"max_MI_{col}"
                    )
                    column_min_max_MU[col] = (min_value, max_value)
            else:
                column_min_max_MU = {}

    if calculate_button:
        if not selected_columns:
            st.error("Error: You need to select at least one column to continue!")
        else:
            st.subheader("Results")
            # Mostrar o indicador de carregamento
            with st.spinner('Calculating... Please wait.'):
                # Normalização das colunas selecionadas 
                for column in selected_columns:
                    if (control_variable != "Choose an option") and not df[control_variable].isnull().all():
                        correlation = df[control_variable].corr(df[column])
                        normalization_type = 'Min' if correlation > 0 else 'Max'
                    else:
                        normalization_type = 'Min'
                    
                    data[column] = normalizar_dados(df[column].tolist(), normalization_type)


                # Criar uma aba para cada método
                tabs = st.tabs(["📉 PCA", "📊 Equal Weights", "💹 Shannon's Entropy", "📈 BoD", "🧮 Minimal Uncertainty"])
                methods = ["PCA", "Equal Weights", "Shannon's Entropy", "BoD", "Minimal Uncertainty"]

                for tab, method in zip(tabs, methods):
                    with tab:
                        # Cálculo do método correspondente
                        if method == "PCA":
                            model = PCA_Calculation(data)
                        elif method == "BoD":
                            bounds = [column_min_max_BoD[col] for col in selected_columns if col in column_min_max_BoD]
                            #verificar se bounds estas entre 0 e 1
                            if any(min_val < 0 or max_val > 1 for min_val, max_val in bounds):
                                st.error("Error: Min/Max values must be between 0 and 1.")
                                continue
                            model = BOD_Calculation(data, bounds=bounds)
                        elif method == "Equal Weights":
                            model = EqualWeights(data)
                        elif method == "Shannon's Entropy":
                            model = Entropy_Calculation(data)
                        elif method == "Minimal Uncertainty":
                            bounds = [column_min_max_MU[col] for col in selected_columns if col in column_min_max_MU]
                            #verificar se bounds estas entre 0 e 1
                            if any(min_val < 0 or max_val > 1 for min_val, max_val in bounds):
                                st.error("Error: Min/Max values must be between 0 and 1.")
                                continue
                            model = Minimal_Uncertainty(data, ranking_ic, bounds=bounds)

                        try:
                            result = model.run()
                        except ValueError as e:
                            st.error(f"Error: {str(e)}")
                            continue

                        # Organizar os resultados
                        filtered_df = pd.DataFrame(result)

                        #Ranking dos indicadores compostos para calcular a incerteza mínima
                        if method != "Minimal Uncertainty":
                            ranking_ic.append(filtered_df["ci"].rank(method='min').to_list())

                        if labels_column.strip() != "Choose an option":
                            filtered_df.index = df[labels_column]
                        else:
                            filtered_df.index = ["DMU " + str(i+1) for i in df.index]
                        
                        filtered_df.sort_values(by="ci", ascending=False, inplace=True)

                        # Formatar os pesos
                        filtered_df['weights'] = filtered_df['weights'].apply(lambda x: [f"{i:.3f}" for i in x])

                        # Exibir a tabela
                        st.subheader(f"{method}")
                        st.dataframe(filtered_df)

                        # Gerar um arquivo Excel para download
                        excel_buffer = io.BytesIO()
                        filtered_df.to_excel(excel_buffer, index=False)
                        excel_buffer.seek(0)

                        st.download_button(
                            label=f"Download {method} results (xlsx)",
                            data=excel_buffer,
                            file_name=f"{method}_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Gráfico de Dispersão
                        fig = px.scatter(filtered_df, 
                                         y="ci", 
                                         title=f"{method} - Composite Indicators",
                                         labels={"ci": "CI"})
                        st.plotly_chart(fig)

                        # Histograma
                        fig_hist = px.histogram(filtered_df, x="ci", nbins=20, title=f"{method} - CI Distribution", labels={"ci": "CI"})
                        st.plotly_chart(fig_hist)

                        # Valores extremos
                        min_ci = filtered_df["ci"].min()
                        max_ci = filtered_df["ci"].max()

                        # Exibir valores extremos em estilo formatado
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; gap: 20px;">
                                <div style="flex: 1; background-color:#f1f1f1; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                                    <h3 style="color:#333;">CI - Min. value</h3>
                                    <h2 style="color:#555;">{min_ci:.3f}</h2>
                                </div>
                                <div style="flex: 1; background-color:#f1f1f1; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                                    <h3 style="color:#333;">CI - Max. value</h3>
                                    <h2 style="color:#555;">{max_ci:.3f}</h2>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

else:
    st.warning("Please upload an Excel file to proceed.")
