import streamlit as st
import pandas as pd
from utils import normalizar_dados, BOD_Calculation, Entropy_Calculation, EqualWeights, PCA_Calculation
import plotly.express as px
import io

data = pd.DataFrame()

st.title('📉 Basics Composite Indicators')
st.header('Calculate composite indicators. Methods: PCA, BoD, Equal Weigths and Entropy')

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
    control_variable = st.sidebar.selectbox("Select the control variable", number_columns)

    # Selecionar colunas
    string_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    labels_column = st.sidebar.selectbox("Select label column", string_columns)

    # Escolher o normalização
    # normalization = st.sidebar.radio("Choose normalization", ("MinMax", "Standardize"))

    # Escolher o método
    method = st.sidebar.radio("Choose method", ("PCA", "BoD", "Equal Weights", "Shannon's Entropy"))

    # Botões
    # expert_button = st.sidebar.button("Expert's Opinion")
    calculate_button = st.sidebar.button("Calculate")
    download_button = st.download_button("Download xlsx", data=uploaded_file, file_name="downloaded_file.xlsx")

    # Lógica de exibição do botão "Expert's Opinion"
    # if expert_button:
    #     with st.expander("Expert's Opinion Settings", expanded=True):
    #         st.subheader("Set Weights or Adjust Parameters")

    #         # Exibir os campos para opinião de especialista
    #         for column in selected_columns:
    #             col1, col2 = st.columns(2)  # Cria duas colunas lado a lado
    #             with col1:
    #                 weight_min = st.number_input(
    #                     f"Set min. weight for {column}", 
    #                     min_value=0.0, max_value=1.0, step=0.01, 
    #                     key=f"weight_{column}_min"
    #                 )
    #             with col2:
    #                 weight_max = st.number_input(
    #                     f"Set max. weight for {column}", 
    #                     min_value=0.0, max_value=1.0, step=0.01, 
    #                     key=f"weight_{column}_max"
    #                 )
            
    #         st.markdown(
    #             """
    #             Once you adjust the weights, click "Calculate" to apply these settings to the calculation.
    #             """
    #         )

    # Lógica de exibição de resultados ou ações
    if calculate_button:

        if not selected_columns:
            st.error("Error: You need to select at least one column to continue!")
        else:
        # Mostrar o indicador de carregamento
            with st.spinner('Calculating... Please wait.'):
                #1 - Step
                # if normalization == 'MinMax':
                for column in selected_columns:
                    correlation = df[control_variable].corr(df[column])
                    if correlation > 0:
                        data[column] = normalizar_dados(df[column].tolist(), 'Min')
                    else:
                        data[column] = normalizar_dados(df[column].tolist(), 'Max')

                # elif normalization == 'Standardize':
                #     for column in selected_columns:
                #         data[column] = padronizar_dados(df[column].tolist())
                
                #2 - Step
                if method == "PCA":
                    model = PCA_Calculation(data)
                elif method == "BoD":
                    model = BOD_Calculation(data)
                elif method == "Equal Weights":
                    model = EqualWeights(data)
                elif method == "Shannon's Entropy":
                    model = Entropy_Calculation(data)
                
                result = model.run()

                filtered_df = pd.DataFrame(result, index=df[labels_column])
                filtered_df = filtered_df.sort_values(by="ci", ascending=False)

                # Formatar os valores dentro de cada lista para 3 casas decimais
                filtered_df['weights'] = filtered_df['weights'].apply(lambda x: [f"{i:.3f}" for i in x])
                # filtered_df['ci'] = filtered_df['ci'].apply(lambda x: f"{x:.3f}")

                # Mostrar tabela
                st.subheader("Results table")
                st.dataframe(filtered_df)

                
                # Gerar um arquivo Excel em memória
                excel_buffer = io.BytesIO()
                filtered_df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)  # Necessário para voltar ao início do buffer
                
                # Botão de download
                st.download_button(
                    label="Download xlsx",
                    data=excel_buffer,
                    file_name="filtered_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Gráfico de Dispersão
                fig = px.scatter(filtered_df, y="ci", title="Composite Indicators", labels={"ci": "CI"})
                st.plotly_chart(fig)

                # Histograma
                fig_hist = px.histogram(filtered_df, x="ci", nbins=20, title="Distribuição dos Valores de CI", labels={"ci": "Coeficiente CI"})
                st.plotly_chart(fig_hist)

                # Mostrar valores extremos
                # st.subheader("Extreme values")
                min_ci = filtered_df["ci"].min()
                max_ci = filtered_df["ci"].max()
                # st.write(f"CI min.: {min_ci:.3f}")
                # st.write(f"CI max.: {max_ci:.3f}")

                # Container principal
                with st.container():
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
