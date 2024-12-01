import streamlit as st
import pandas as pd
from utils import normalizar_dados, BOD_Calculation, Entropy_Calculation, EqualWeights, PCA_Calculation
import plotly.express as px
import io


data = pd.DataFrame()

st.set_page_config(
    page_title="Composite Indicators",
    page_icon="📉"
)

st.title('📉 Basics Composite Indicators')
st.header("Calculate composite indicators. Methods: PCA, BoD, Equal Weights, and Shannon's Entropy")

# Carregar arquivo Excel
uploaded_file = st.sidebar.file_uploader("Select Excel file", type=["xlsx"])

# Verifique se o arquivo foi carregado
if uploaded_file is not None:
    # Carregar o arquivo Excel em um DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Exibir as primeiras linhas do arquivo
    st.subheader("Data")
    st.write(df.head())

    # Selecionar colunas
    number_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns", number_columns)

    # Selecionar variável de controle
    control_variable = st.sidebar.selectbox("Select the control variable", [""] + number_columns)

    # Selecionar colunas
    string_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    labels_column = st.sidebar.selectbox("Select label column", string_columns)

    # Botão
    calculate_button = st.sidebar.button("Calculate")

    if calculate_button:
        if not selected_columns:
            st.error("Error: You need to select at least one column to continue!")
        else:
            st.subheader("Results")
            # Mostrar o indicador de carregamento
            with st.spinner('Calculating... Please wait.'):
                # Normalização das colunas selecionadas 
                for column in selected_columns:
                    if control_variable and not df[control_variable].isnull().all():
                        correlation = df[control_variable].corr(df[column])
                        normalization_type = 'Min' if correlation > 0 else 'Max'
                    else:
                        normalization_type = 'Min'
                    
                    data[column] = normalizar_dados(df[column].tolist(), normalization_type)


                # Criar uma aba para cada método
                tabs = st.tabs(["📉 PCA", "📈 BoD", "📊 Equal Weights", "💹 Shannon's Entropy"])
                methods = ["PCA", "BoD", "Equal Weights", "Shannon's Entropy"]

                for tab, method in zip(tabs, methods):
                    with tab:
                        # Cálculo do método correspondente
                        if method == "PCA":
                            model = PCA_Calculation(data)
                        elif method == "BoD":
                            model = BOD_Calculation(data)
                        elif method == "Equal Weights":
                            model = EqualWeights(data)
                        elif method == "Shannon's Entropy":
                            model = Entropy_Calculation(data)

                        result = model.run()

                        # Organizar os resultados
                        filtered_df = pd.DataFrame(result, index=df[labels_column])
                        filtered_df = filtered_df.sort_values(by="ci", ascending=False)

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
                        fig = px.scatter(filtered_df, y="ci", title=f"{method} - Composite Indicators", labels={"ci": "CI"})
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
