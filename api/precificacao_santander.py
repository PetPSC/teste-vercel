import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import warnings
import polars as pl

def inflacao_ajustada(df):
    df = df.loc[df.groupby('Ano')['Mês'].idxmin()]
    df.reset_index(inplace=True)
    df.loc[df['Ano'] == 0,'Inflação acumulada'] = 0
    return df

def monta_parceiro(df_parceiro):



    # Convert numeric columns to proper format for comparison, handling potential NaN values

    # Filter columns containing 'Proposta' in their names while ignoring NaN or non-iterable columns
    proposta_cols = [col for col in df_parceiro.columns if  ('Proposta' in col[1])]

    # Ensure numeric values for 'Proposta' columns
    for col in proposta_cols:
        df_parceiro[col] = df_parceiro[col].fillna(0)
        df_parceiro[col] = pd.to_numeric(df_parceiro[col], errors='coerce')

    # Calculate 'maior lance' column
    df_parceiro[('MAIOR_OFERTA','MAIOR_OFERTA')] = df_parceiro[proposta_cols].max(axis=1)
    df_parceiro[('OFERTANTE','OFERTANTE')] = df_parceiro[proposta_cols].idxmax(axis=1).apply(lambda x: x[0])
    df_parceiro[('MODELO','MODELO')] = df_parceiro.apply(
        lambda row: row[(row[('OFERTANTE','OFERTANTE')], 'Modelo')] 
        if pd.notna(row[('OFERTANTE','OFERTANTE')]) else None
        ,axis=1
    )



    df_parceiro.columns = df_parceiro.columns.droplevel(0)
    df_parceiro=df_parceiro[df_parceiro['MAIOR_OFERTA'] > 0][['NR_CONTRATO','MAIOR_OFERTA', 'OFERTANTE', 'MODELO']]
    df_parceiro.reset_index(inplace=True
                            ,drop=True)
    return df_parceiro

def precifica_carrego(df
                    ,df_cdi
                    ,df_inflacao_filtered
                    ,spread_carrego):
    # Ajustes das taxas, ponto importante a verificar, pois pode ser alterado o padrão no arquivo que santander manda
    #df['PC_TX_PAGO'] = df['PC_TX_PAGO']/10000
    #df['PC_FC_PAGO'] = df['PC_FC_PAGO']/100
    df_cdi = df_cdi.copy()
    df_cdi.rename(columns={'TAXA INTERPOLADA':'CDI_LIQUIDACAO'}, inplace=True)

    #df['DATA_REFERENCIA'] = data_referencia

    df['SPREAD_CARREGO'] = spread_carrego
    

    # df = df[df['NM_SITU_ENTREGA_BEM'] != 'Bem Pendente de Entrega'] # Fazer modelo para precificar
    # df = df[~((df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada') & (df['VL_DEVOLVER'] == 0))]
    
    
    # df['VALOR_DEVOLVER_BRBB'] = np.where(df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada', df['VL_DEVOLVER'], (df['VL_BEM_ATUAL'] * (df['PC_FC_PAGO']/100)) *0.8)

    df['DATA_ULTIMA_AGO'] = df.apply(lambda x : x['DATA_REFERENCIA'] + relativedelta(months=x['PZ_RESTANTE_GRUPO']), axis=1)
    df['DATA_PRIMEIRA_AGO'] = df.apply(lambda x : x['DATA_ULTIMA_AGO'] - relativedelta(months=x['PZ_COMERCIALIZACAO']) ,axis=1)
    df['ULTIMO_AJUSTE'] = df.apply(lambda x: x['DATA_PRIMEIRA_AGO'] + relativedelta(months=x['PZ_COMERCIALIZACAO']-1), axis=1)
    df['MESES_DATABASE_ATE_ULTIMO_AJUSTE'] = df.apply(lambda x: relativedelta(x['ULTIMO_AJUSTE'], x['DATA_REFERENCIA']).months + (relativedelta(x['ULTIMO_AJUSTE'], x['DATA_REFERENCIA']).years*12) if x['DATA_REFERENCIA'] < x['ULTIMO_AJUSTE']  else 0, axis=1)
    df['ANIVERSARIOS'] = df['MESES_DATABASE_ATE_ULTIMO_AJUSTE'].apply(lambda x: 0 if x == 0 else (x // 12 - 1 if x % 12 == 0 else x // 12))
    df['DATA_LIQUIDAÇÃO'] = df['DATA_ULTIMA_AGO'].apply(lambda x: x + relativedelta(days=120))
    df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'] = (df['DATA_LIQUIDAÇÃO'] - df['DATA_REFERENCIA']).dt.days

    
    #Buscando cdi na curva
    maior_dc = df_cdi['DC'].max()
    taxa_maior_dc = df_cdi[df_cdi['DC'] == df_cdi['DC'].max()]['CDI_LIQUIDACAO']
    df = df.merge(df_cdi[['DC', 'CDI_LIQUIDACAO']], right_on='DC', left_on='DIAS_CORRIDOS_DATABASE_LIQUIDACAO', how='left')
    df['CDI_LIQUIDACAO'] = df.apply(lambda x: taxa_maior_dc if x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'] > maior_dc else x['CDI_LIQUIDACAO'],axis=1)
    df['CDI_LIQUIDACAO'] = df['CDI_LIQUIDACAO'].fillna(0)
    df['CDI_CORRECAO'] = df['CDI_LIQUIDACAO'] * 0.8

    #Buscando inflação baseado nos aniversários restantes
    maior_ano = df_inflacao_filtered['Ano'].max()
    df['ANO_MERGE'] = df['ANIVERSARIOS'].apply(lambda x: x if x <= maior_ano else maior_ano)
    df = df.merge(df_inflacao_filtered[['Ano', 'Inflação acumulada']], how='left', left_on='ANO_MERGE', right_on='Ano')
    df.drop(columns=['ANO_MERGE', 'Ano'], inplace=True)
    df.rename(columns={'Inflação acumulada' : 'INFLACAO_ATE_ULTIMA_AGO'}, inplace=True)

    # Melhorar estrategia para pegar cdi
    #df['CDI_AA'] = 0.1125

    # Construção das Taxas 
    df['TAXA_SANTANDER'] = df.apply(
        lambda x: 1000 if x['VL_DEVOLVER'] > 7000 and x['VL_BEM_ATUAL'] > 70000 else (
            700 if x['VL_DEVOLVER'] > 7000 or x['VL_DEVOLVER'] > 5000 else (
                500 if x['VL_DEVOLVER'] > 3000 else (
                    300 if x['VL_DEVOLVER'] > 1500 else 150
                )
            )
        ),
        axis=1)


    #Calculo dos Direitos Creditórios na Data da Última Assembleia e na Data da Liquidação
    #Data Liquidação = Ultima AGO + 120dias
    df['DIR_CRED_ULTIMA_AGO'] = df.apply(lambda x: 0 if x['VALOR_DEVOLVER_BRBB']== 0 else x['VALOR_DEVOLVER_BRBB'] * (1 + x['INFLACAO_ATE_ULTIMA_AGO']), axis=1)
    df['DIR_CRED_DATA_LIQUIDACAO'] = df['DIR_CRED_ULTIMA_AGO'] * (1+ df['CDI_CORRECAO'])**(120/252)

    

    #Oferta BRBB
    df['VP_CARREGO_DESAGIO'] = df['DIR_CRED_DATA_LIQUIDACAO'] / ((1+df['CDI_LIQUIDACAO'])*(1+df['SPREAD_CARREGO']))**(df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']/365)
    df['OFERTA_BRBB_CARREGO'] = (df['VP_CARREGO_DESAGIO'] - df['TAXA_SANTANDER'])/1.03
    df['TAXA_INLIRA_CARREGO'] = df['OFERTA_BRBB_CARREGO'] * 0.03
    df['DESEMBOLSO_TOTAL_CARREGO'] = df['OFERTA_BRBB_CARREGO'] + df['TAXA_SANTANDER'] + df['TAXA_INLIRA_CARREGO']
    df['TIR_CARREGO_BRBB_DESEMBOLSO'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['DESEMBOLSO_TOTAL_CARREGO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1, axis=1)
    #df['TIR_CARREGO_BRBB_OFERTA'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['OFERTA_BRBB_CARREGO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1, axis=1)

    return df

def precifica_ccr(df
                ,df_cdi
                ,spread_ccr):
    df_cdi = df_cdi.copy()

    df['SPREAD_CCR'] = spread_ccr

    renamed_column_cdi = 'CDI_LIQUIDACAO_CCR'
    df_cdi.rename(columns={'TAXA INTERPOLADA':f'{renamed_column_cdi}'}, inplace=True)


    df['DC_ATE_LIQUIDACAO_CCR'] = 270
    df = df.merge(df_cdi[['DC', f'{renamed_column_cdi}']] 
                    ,how='left'
                    ,left_on='DC_ATE_LIQUIDACAO_CCR'
                    ,right_on='DC')
    df['CDI_CORRECAO_CCR'] = df[f'{renamed_column_cdi}']*0.8



    df['%SD'] = df.apply(
                    lambda x: 0 if (100 - x['PC_FC_PAGO'] + x['PC_TX_ADM'] - x['PC_TX_PAGO'] + x['PC_FUNDO_RESERVA'] - x['PC_FR_PAGO']) / 100 < 0 
                    else (100 - x['PC_FC_PAGO'] + x['PC_TX_ADM'] - x['PC_TX_PAGO'] + x['PC_FUNDO_RESERVA'] - x['PC_FR_PAGO']) / 100,
                    axis=1
                    )

    df['Lance_Total_CCR'] = df['VL_BEM_ATUAL'] * df['%SD']
    df['Lance_Embutido']  = df.apply(lambda x: 0.3*x['VL_BEM_ATUAL'] if 0.3*x['VL_BEM_ATUAL']< x['Lance_Total_CCR'] else x['Lance_Total_CCR'], axis=1) # 30%bem atual, se nao for maior que o lance total
    df['Valor_Em_Especie'] = df['Lance_Total_CCR'] - df['Lance_Embutido']
    #df['Credito_Ajustado'] = df['VL_BEM_ATUAL']*(1+df['CDI_CORRECAO'])**(1/2)
    df['Credito_Ajustado'] = (df['VL_BEM_ATUAL']- df['Lance_Embutido'])*(1+df['CDI_CORRECAO_CCR'])**(6/12)
    df['VP_CCR_DESAGIO'] = (df['Credito_Ajustado'] / ((1+df[f'{renamed_column_cdi}'])*(1+df['SPREAD_CCR'] ))**(df['DC_ATE_LIQUIDACAO_CCR']/365))
    df['OFERTA_BRBB_CCR'] =  (df['VP_CCR_DESAGIO'] - df['TAXA_SANTANDER'])/1.03

    df['TAXA_INLIRA_CCR'] = df['OFERTA_BRBB_CCR']*0.03
    df['DESEMBOLSO_TOTAL_CCR'] = df['OFERTA_BRBB_CCR']  + df['TAXA_SANTANDER'] + df['TAXA_INLIRA_CCR'] + df['Valor_Em_Especie']
    df['TIR_CCR_BRBB_DESEMBOLSO'] = df.apply(lambda x: float((x['Credito_Ajustado']/x['DESEMBOLSO_TOTAL_CCR'])**(12/7))-1, axis=1)

    return df

def importador_santander(df
                           ,df_inflacao_filtered
                           ,data_referencia
                           ,spread_carrego
                           ,spread_ccr
                           ,df_cdi):
    
    # Ajustes das taxas, ponto importante a verificar, pois pode ser alterado o padrão no arquivo que santander manda
    df['PC_TX_PAGO'] = df['PC_TX_PAGO']/10000

    df['DATA_REFERENCIA'] = data_referencia

    
    

    df = df[df['NM_SITU_ENTREGA_BEM'] != 'Bem Pendente de Entrega'] # Fazer modelo para precificar
    df = df[~((df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada') & (df['VL_DEVOLVER'] == 0))]
    
    
    df['VALOR_DEVOLVER_BRBB'] = np.where(df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada', df['VL_DEVOLVER'], (df['VL_BEM_ATUAL'] * (df['PC_FC_PAGO']/100)) *0.8)

    df = precifica_carrego(df
                    ,df_cdi
                    ,df_inflacao_filtered
                    ,spread_carrego)

    df = precifica_ccr(df
                ,df_cdi
                ,spread_ccr)
    

    return df


def precificacao_santander(df
                           ,df_inflacao_filtered
                           ,data_referencia
                           ,cdi_desagio
                           ,spread_carrego
                           ,spread_ccr):
    
    # Ajustes das taxas, ponto importante a verificar, pois pode ser alterado o padrão no arquivo que santander manda
    df['PC_TX_PAGO'] = df['PC_TX_PAGO']/10000
    #df['PC_FC_PAGO'] = df['PC_FC_PAGO']/100

    df['DATA_REFERENCIA'] = data_referencia

    df['CDI_DESAGIO'] = cdi_desagio
    df['SPREAD_CARREGO'] = spread_carrego
    df['SPREAD_CCR'] = spread_ccr
    df['CDI_CORRECAO'] = df['CDI_DESAGIO'] * 0.8

    df = df[df['NM_SITU_ENTREGA_BEM'] != 'Bem Pendente de Entrega'] # Fazer modelo para precificar
    df = df[~((df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada') & (df['VL_DEVOLVER'] == 0))]
    
    
    df['VALOR_DEVOLVER_BRBB'] = np.where(df['NM_SITU_ENTREGA_BEM'] == 'Cota Cancelada', df['VL_DEVOLVER'], (df['VL_BEM_ATUAL'] * (df['PC_FC_PAGO']/100)) *0.8)

    df['DATA_ULTIMA_AGO'] = df['PZ_RESTANTE_GRUPO'].apply(lambda x : data_referencia + relativedelta(months=x))
    df['DATA_PRIMEIRA_AGO'] = df.apply(lambda x : x['DATA_ULTIMA_AGO'] - relativedelta(months=x['PZ_COMERCIALIZACAO']) ,axis=1)
    df['ULTIMO_AJUSTE'] = df.apply(lambda x: x['DATA_PRIMEIRA_AGO'] + relativedelta(months=x['PZ_COMERCIALIZACAO']-1), axis=1)
    df['MESES_DATABASE_ATE_ULTIMO_AJUSTE'] = df.apply(lambda x: relativedelta(x['ULTIMO_AJUSTE'], x['DATA_REFERENCIA']).months + (relativedelta(x['ULTIMO_AJUSTE'], x['DATA_REFERENCIA']).years*12) if x['DATA_REFERENCIA'] < x['ULTIMO_AJUSTE']  else 0, axis=1)
    df['ANIVERSARIOS'] = df['MESES_DATABASE_ATE_ULTIMO_AJUSTE'].apply(lambda x: 0 if x == 0 else (x // 12 - 1 if x % 12 == 0 else x // 12))
    df['DATA_LIQUIDAÇÃO'] = df['DATA_ULTIMA_AGO'].apply(lambda x: x + relativedelta(days=120))

    #Buscando inflação baseado nos aniversários restantes
    maior_ano = df_inflacao_filtered['Ano'].max()
    df['ANO_MERGE'] = df['ANIVERSARIOS'].apply(lambda x: x if x <= maior_ano else maior_ano)
    df = df.merge(df_inflacao_filtered[['Ano', 'Inflação acumulada']], how='left', left_on='ANO_MERGE', right_on='Ano')
    df.drop(columns=['ANO_MERGE', 'Ano'], inplace=True)
    df.rename(columns={'Inflação acumulada' : 'INFLACAO_ATE_ULTIMA_AGO'}, inplace=True)

    # Melhorar estrategia para pegar cdi
    #df['CDI_AA'] = 0.1125

    # Construção das Taxas 
    df['TAXA_SANTANDER'] = df.apply(
        lambda x: 1000 if x['VL_DEVOLVER'] > 7000 and x['VL_BEM_ATUAL'] > 70000 else (
            700 if x['VL_DEVOLVER'] > 7000 or x['VL_DEVOLVER'] > 5000 else (
                500 if x['VL_DEVOLVER'] > 3000 else (
                    300 if x['VL_DEVOLVER'] > 1500 else 150
                )
            )
        ),
        axis=1)


    #Calculo dos Direitos Creditórios na Data da Última Assembleia e na Data da Liquidação
    #Data Liquidação = Ultima AGO + 120dias
    df['DIR_CRED_ULTIMA_AGO'] = df.apply(lambda x: 0 if x['VALOR_DEVOLVER_BRBB']== 0 else x['VALOR_DEVOLVER_BRBB'] * (1 + x['INFLACAO_ATE_ULTIMA_AGO']), axis=1)
    df['DIR_CRED_DATA_LIQUIDACAO'] = df['DIR_CRED_ULTIMA_AGO'] * (1+ df['CDI_CORRECAO'])**(120/252)

    df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'] = (df['DATA_LIQUIDAÇÃO'] - df['DATA_REFERENCIA']).dt.days

    #Oferta BRBB
    df['VP_CARREGO_DESAGIO'] = df['DIR_CRED_DATA_LIQUIDACAO'] / ((1+df['CDI_DESAGIO'])*(1+df['SPREAD_CARREGO']))**(df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']/365)
    df['OFERTA_BRBB_CARREGO'] = (df['VP_CARREGO_DESAGIO'] - df['TAXA_SANTANDER'])/1.03
    df['TAXA_INLIRA_CARREGO'] = df['OFERTA_BRBB_CARREGO'] * 0.03
    df['DESEMBOLSO_TOTAL_CARREGO'] = df['OFERTA_BRBB_CARREGO'] + df['TAXA_SANTANDER'] + df['TAXA_INLIRA_CARREGO']
    df['TIR_CARREGO_BRBB_DESEMBOLSO'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['DESEMBOLSO_TOTAL_CARREGO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1, axis=1)
    #df['TIR_CARREGO_BRBB_OFERTA'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['OFERTA_BRBB_CARREGO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1, axis=1)

    df['%SD'] = df.apply(
    lambda x: 0 if (100 - x['PC_FC_PAGO'] + x['PC_TX_ADM'] - x['PC_TX_PAGO'] + x['PC_FUNDO_RESERVA'] - x['PC_FR_PAGO']) / 100 < 0 
    else (100 - x['PC_FC_PAGO'] + x['PC_TX_ADM'] - x['PC_TX_PAGO'] + x['PC_FUNDO_RESERVA'] - x['PC_FR_PAGO']) / 100,
    axis=1
    )

    df['Lance_Total_CCR'] = df['VL_BEM_ATUAL'] * df['%SD']
    df['Lance_Embutido']  = df.apply(lambda x: 0.3*x['VL_BEM_ATUAL'] if 0.3*x['VL_BEM_ATUAL']< x['Lance_Total_CCR'] else x['Lance_Total_CCR'], axis=1) # 30%bem atual, se nao for maior que o lance total
    df['Valor_Em_Especie'] = df['Lance_Total_CCR'] - df['Lance_Embutido']
    #df['Credito_Ajustado'] = df['VL_BEM_ATUAL']*(1+df['CDI_CORRECAO'])**(1/2)
    df['Credito_Ajustado'] = (df['VL_BEM_ATUAL']- df['Lance_Embutido'])*(1+df['CDI_CORRECAO'])**(6/12)

    #df['VP_CCR_DESAGIO'] = (df['Credito_Ajustado'] / ((1+df['CDI_DESAGIO'])*(1+df['SPREAD_CCR'] ))**(1/2))
    df['VP_CCR_DESAGIO'] = (df['Credito_Ajustado'] / ((1+df['CDI_DESAGIO'])*(1+df['SPREAD_CCR'] ))**(7/12))
    df['OFERTA_BRBB_CCR'] =  (df['VP_CCR_DESAGIO'] - df['TAXA_SANTANDER'])/1.03
    df['TAXA_INLIRA_CCR'] = df['OFERTA_BRBB_CCR']*0.03
    df['DESEMBOLSO_TOTAL_CCR'] = df['OFERTA_BRBB_CCR']  + df['TAXA_SANTANDER'] + df['TAXA_INLIRA_CCR'] + df['Valor_Em_Especie']
    df['TIR_CCR_BRBB_DESEMBOLSO'] = df.apply(lambda x: float((x['Credito_Ajustado']/x['DESEMBOLSO_TOTAL_CCR'])**(12/7))-1, axis=1)

    return df

def oferta_parceiro(df
                    ,df_parceiro
                    ,qtde_ajuste):
    df = df.merge(df_parceiro[['NR_CONTRATO','MAIOR_OFERTA', 'OFERTANTE', 'MODELO']]
              ,on='NR_CONTRATO'
              ,how='left')
    
    df['TAXA_INLIRA_OFERTA_MERCADO'] = df['MAIOR_OFERTA']*0.03 # taxa originacao oferta mercado

    df = df[df['MAIOR_OFERTA'].notna()]

    df['INVESTIMENTO_CARREGO_MERCADO'] = df.apply(lambda x: x['MAIOR_OFERTA'] + x['TAXA_INLIRA_OFERTA_MERCADO'] + x['TAXA_SANTANDER'] if x['MODELO'] == 'CARREGO' else 0, axis=1)
    #df['TIR_CARREGO_MERCADO'] = (df['DIR_CRED_DATA_LIQUIDACAO'] /df['INVESTIMENTO_CARREGO_MERCADO'])**(365/df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'])-1
    df['TIR_CARREGO_MERCADO_DESEMBOLSO'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['INVESTIMENTO_CARREGO_MERCADO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1 if x['MODELO'] == 'CARREGO' else 0, axis=1)
    df['TIR_CARREGO_MERCADO_OFERTA'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['MAIOR_OFERTA'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1 if x['MODELO'] == 'CARREGO' else 0, axis=1)

    df['INVESTIMENTO_TOTAL_CCR_MERCADO'] = df.apply(lambda x: x['MAIOR_OFERTA'] + x['Valor_Em_Especie'] + x['TAXA_SANTANDER'] + x['TAXA_INLIRA_OFERTA_MERCADO']  if x['MODELO'] == 'CCR' else 0, axis=1)
    df['TIR_CCR_MERCADO_DESEMBOLSO'] = df.apply(lambda x: float((x['Credito_Ajustado']/x['INVESTIMENTO_TOTAL_CCR_MERCADO'])**(365/210))-1 if x['MODELO'] == 'CCR' else 0, axis=1)
    df['TIR_CCR_MERCADO_OFERTA'] = df.apply(lambda x: float((x['Credito_Ajustado']/x['MAIOR_OFERTA'])**(365/210))-1 if x['MODELO'] == 'CCR' else 0, axis=1)

    #df['MELHOR_OFERTA_AJUST_TIR'] = df.apply(lambda x: x['MAIOR_OFERTA']*((1+x['TIR_CCR_MERCADO_DESEMBOLSO'])**(1/12))  if x['MODELO'] == 'CCR' else (x['MAIOR_OFERTA']*((1+x['TIR_CARREGO_MERCADO_DESEMBOLSO'])**(1/12)) if x['MODELO'] == 'CARREGO' else 0),axis=1)
    df['MELHOR_OFERTA_AJUST_TIR'] = df.apply(lambda x: x['MAIOR_OFERTA']*((1+x['TIR_CCR_MERCADO_DESEMBOLSO'])**(qtde_ajuste/12))  if x['MODELO'] == 'CCR' else (x['MAIOR_OFERTA']*((1+x['TIR_CARREGO_MERCADO_OFERTA'])**(qtde_ajuste/12)) if x['MODELO'] == 'CARREGO' else 0),axis=1)
    df['MELHOR_TIR_MERCADO_OFERTA'] = df.apply(lambda x: x['TIR_CCR_MERCADO_DESEMBOLSO'] if x['MODELO'] == 'CCR' else (x['TIR_CARREGO_MERCADO_OFERTA'] if x['MODELO'] == 'CARREGO' else 0),axis=1)

    return df



def oferta_parceiro_ajustado(df
                    ,df_parceiro
                    ,qtde_ajuste
                    ,coluna_oferta
                    ,coluna_modelo):
    df = df.merge(df_parceiro[['NR_CONTRATO',f'{coluna_oferta}', f'{coluna_modelo}']]
              ,on='NR_CONTRATO'
              ,how='left')
    
    df['TAXA_INLIRA_OFERTA_MERCADO'] = df[f'{coluna_oferta}']*0.03 # taxa originacao oferta mercado

    df = df[df[f'{coluna_oferta}'].notna()]

    df['INVESTIMENTO_CARREGO_MERCADO'] = df.apply(lambda x: x[f'{coluna_oferta}'] + x['TAXA_INLIRA_OFERTA_MERCADO'] + x['TAXA_SANTANDER'] if x[f'{coluna_modelo}'] == 'CARREGO' else 0, axis=1)
    #df['TIR_CARREGO_MERCADO'] = (df['DIR_CRED_DATA_LIQUIDACAO'] /df['INVESTIMENTO_CARREGO_MERCADO'])**(365/df['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'])-1
    df['TIR_CARREGO_MERCADO_DESEMBOLSO'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x['INVESTIMENTO_CARREGO_MERCADO'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1 if x[f'{coluna_modelo}'] == 'CARREGO' else 0, axis=1)
    df['TIR_CARREGO_MERCADO_OFERTA'] = df.apply(lambda x: float((x['DIR_CRED_DATA_LIQUIDACAO']/x[f'{coluna_oferta}'])**(365/x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']))-1 if x[f'{coluna_modelo}'] == 'CARREGO' else 0, axis=1)

    df['INVESTIMENTO_TOTAL_CCR_MERCADO'] = df.apply(lambda x: x[f'{coluna_oferta}'] + x['Valor_Em_Especie'] + x['TAXA_SANTANDER'] + x['TAXA_INLIRA_OFERTA_MERCADO']  if x[f'{coluna_modelo}'] == 'CCR' else 0, axis=1)
    df['TIR_CCR_MERCADO_DESEMBOLSO'] = df.apply(lambda x: float((x['Credito_Ajustado']/x['INVESTIMENTO_TOTAL_CCR_MERCADO'])**(365/210))-1 if x[f'{coluna_modelo}'] == 'CCR' else 0, axis=1)
    df['TIR_CCR_MERCADO_OFERTA'] = df.apply(lambda x: float((x['Credito_Ajustado']/x[f'{coluna_oferta}'])**(365/210))-1 if x[f'{coluna_modelo}'] == 'CCR' else 0, axis=1)

    #df['MELHOR_OFERTA_AJUST_TIR'] = df.apply(lambda x: x['MAIOR_OFERTA']*((1+x['TIR_CCR_MERCADO_DESEMBOLSO'])**(1/12))  if x['MODELO'] == 'CCR' else (x['MAIOR_OFERTA']*((1+x['TIR_CARREGO_MERCADO_DESEMBOLSO'])**(1/12)) if x['MODELO'] == 'CARREGO' else 0),axis=1)
    df['MELHOR_OFERTA_AJUST_TIR'] = df.apply(lambda x: x[f'{coluna_oferta}']*((1+x['TIR_CCR_MERCADO_DESEMBOLSO'])**(qtde_ajuste/12))  if x[f'{coluna_modelo}'] == 'CCR' else (x[f'{coluna_oferta}']*((1+x['TIR_CARREGO_MERCADO_OFERTA'])**(qtde_ajuste/12)) if x[f'{coluna_modelo}'] == 'CARREGO' else 0),axis=1)
    df['TIR_AJUSTADA_BRBB'] = df.apply(lambda x: x['TIR_CCR_MERCADO_DESEMBOLSO'] if x[f'{coluna_modelo}'] == 'CCR' else (x['TIR_CARREGO_MERCADO_OFERTA'] if x[f'{coluna_modelo}'] == 'CARREGO' else 0),axis=1)
    df['INVESTIMENTO_TOTAL_MERCADO_BRBB'] = df.apply(lambda x: x['INVESTIMENTO_TOTAL_CCR_MERCADO'] if x[f'{coluna_modelo}'] == 'CCR' else (x['INVESTIMENTO_CARREGO_MERCADO'] if x[f'{coluna_modelo}'] == 'CARREGO' else 0),axis=1)
    df['TAXA_SANTANDER_MERCADO'] = df['TAXA_SANTANDER']
 
    df_return = df[['NR_CONTRATO','TAXA_INLIRA_OFERTA_MERCADO', 'TAXA_SANTANDER_MERCADO','INVESTIMENTO_TOTAL_MERCADO_BRBB','TIR_AJUSTADA_BRBB']]

    return df_return



def teste():

    """MONTA BASE DE OUTUBRO"""
    print('Precificando Outubro...')
    df_inflacao = pd.read_excel(r"C:\Users\peter\Projetos\Projeção de Curva de Inflação (Premissa para modelagens) - Outubro.xlsx"
                         ,sheet_name='Curva de inflação'
                         ,skiprows=3
                         ,usecols='B:G')

    df_inflacao_filtered = inflacao_ajustada(df_inflacao)


    data_referencia = datetime.strptime('2024-10-09', '%Y-%m-%d')
    spread_carrego = 0.1
    spread_ccr = 0.1

    df =  pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\PUBLICO_ALVO_CONSOLIDADO_2024-10-09.xlsx")
    df_cdi1 = pd.read_excel(r"C:\Users\peter\Projetos\curva_cdi.xlsx", sheet_name='Novembro')

    df = importador_santander(df
                           ,df_inflacao_filtered
                           ,data_referencia
                           ,spread_carrego
                           ,spread_ccr
                           ,df_cdi1)

    df_parceiro = pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\Resultado 2024-10-16.parceiros.xlsx"
                                ,header=None)
    df_parceiro.columns = pd.MultiIndex.from_frame(df_parceiro.iloc[:2].T)
    df_parceiro = df_parceiro[2:] 
    # Excluir Cotas que não obtiveram lances
    df_parceiro = df_parceiro[df_parceiro[(float('nan'), 'Propostas disponíveis')] > 0] 
    df_parceiro =  monta_parceiro(df_parceiro)
    df1  = oferta_parceiro(df
                        ,df_parceiro
                        ,1)
    df1_resumido = df1[['NR_CONTRATO','MODELO','OFERTANTE','MAIOR_OFERTA','MELHOR_OFERTA_AJUST_TIR','MELHOR_TIR_MERCADO_OFERTA']]

    """MONTA BASE DE NOVEMBRO"""
    print('\nPrecificando Novembro...')
    df_inflacao = pd.read_excel(r"C:\Users\peter\Projetos\Projeção de Curva de Inflação (Premissa para modelagens) - Novembro.xlsx"
                         ,sheet_name='Curva de inflação'
                         ,skiprows=3
                         ,usecols='B:G')

    df_inflacao_filtered = inflacao_ajustada(df_inflacao)
    data_referencia = datetime.strptime('2024-11-07', '%Y-%m-%d')
    spread_carrego = 0.1
    spread_ccr = 0.1
    df2 =  pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\PUBLICO_ALVO_CONSOLIDADO_2024-11-07.xlsx")
    df_cdi2 = pd.read_excel(r"C:\Users\peter\Projetos\curva_cdi.xlsx", sheet_name='Dezembro')
    df2 = importador_santander(df2
                           ,df_inflacao_filtered
                           ,data_referencia
                           ,spread_carrego
                           ,spread_ccr
                           ,df_cdi2)
    # print('\nParceiros Novembro...')
    # df_parceiro2 = pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\Leilões antigos\Resultado\Resultado Leilão 2024-11-18.parceiros.xlsx"
    #                         ,header=None)

    # Set the first two rows as multi-level columns for the sample data
    # df_parceiro2.columns = pd.MultiIndex.from_frame(df_parceiro2.iloc[:2].T)
    # df_parceiro2 = df_parceiro2[2:]  # Exclude the header rows from the data
    # df_parceiro2 = df_parceiro2[df_parceiro2[(float('nan'), 'Propostas disponíveis')] > 0]


    # df_parceiro2 =  monta_parceiro(df_parceiro2)
    # df2  = oferta_parceiro(df2
    #                 ,df_parceiro2
    #                 ,1)
    
    """MONTA DATAFRAME MERGE"""
    print('\nFazendo Merge...')
    mes = "OUTUBRO"
    df1_resumido.rename(columns={'MODELO':f'MODELO_{mes}'
                           ,'OFERTANTE':f'OFERTANTE_{mes}'
                           ,'MAIOR_OFERTA':f'MAIOR_OFERTA_{mes}'
                           ,'MELHOR_OFERTA_AJUST_TIR':f'MELHOR_OFERTA_AJUST_TIR_{mes}'
                           ,'MELHOR_TIR_MERCADO_OFERTA':f'MELHOR_TIR_MERCADO_OFERTA_{mes}'}
                           ,inplace=True)
    df_merge =df2.merge(df1_resumido, on='NR_CONTRATO', how='left')
    #print('\nExportando para excel...')
    # df_split = df_merge.head()
    # df_split.to_excel(r'C:\Users\peter\Projetos\Leilão Santander\split.xlsx', index=False)
    # df_merge.to_excel('outubro_testee.xlsx', index=False)
    # print(f'\nArquivo outubro_testee.xlsx exportado!')

    """TRAZENDO TIR BRBB"""
    print("Montando TIR Ajustada BRBB\n")
    df_tir_brbb = oferta_parceiro_ajustado(df2
                    ,df1_resumido
                    ,1
                    ,'MELHOR_OFERTA_AJUST_TIR_OUTUBRO'
                    ,'MODELO_OUTUBRO')
    # print('Exportando teste...')
    # df_teste = pl.from_pandas(df_teste)
    # df_teste.write_excel(column_totals=True
    #     ,autofit=True
    #     ,workbook = r"C:\Users\peter\Projetos\Leilão Santander\teste_polars.xlsx")  
    # print('exportado')
    #df_merge =df_merge.merge(df_tir_brbb[['NR_CONTRATO', 'TIR_AJUSTADA_BRBB']], on='NR_CONTRATO', how='left')
    df_merge =df_merge.merge(df_tir_brbb, on='NR_CONTRATO', how='left')

    df_merge["DIAS_LIQ_AJ"] = df_merge.apply(lambda x: x['DC_ATE_LIQUIDACAO_CCR'] if x[f'MODELO_{mes}'] == "CCR" else (x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'] if x[f'MODELO_{mes}'] == "CARREGO" else 0), axis=1)
    maior_dc = df_cdi2['DC'].max()
    taxa_maior_dc = df_cdi2[df_cdi2['DC'] == df_cdi2['DC'].max()]['TAXA INTERPOLADA']
    df_merge = df_merge.merge(df_cdi2[['DC', 'TAXA INTERPOLADA']], right_on='DC', left_on='DIAS_LIQ_AJ', how='left')
    df_merge['TAXA INTERPOLADA'] = df_merge.apply(lambda x: taxa_maior_dc if x['DIAS_LIQ_AJ'] > maior_dc else x['TAXA INTERPOLADA'],axis=1)
    df_merge['TAXA INTERPOLADA'] = df_merge['TAXA INTERPOLADA'].fillna(0)
    #df_merge['CDI_PERIODO'] = df_merge.apply(lambda x: ((1+x['TAXA INTERPOLADA'])**(210/365))-1 if x[f'MODELO_{mes}'] == "CCR" else (((1+x['TAXA INTERPOLADA'])**(x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']/365))-1 if x[f'MODELO_{mes}'] == "CARREGO" else 0), axis=1 )
    df_merge['SPREAD_CDI'] = df_merge.apply(lambda x: ((1+x['TIR_AJUSTADA_BRBB'])/(1+x['TAXA INTERPOLADA']))-1 if x[f'MODELO_{mes}'] in ["CCR", "CARREGO"] else 0, axis=1 )
    # df_merge.to_excel(r'C:\Users\peter\OneDrive\Área de Trabalho\precificacao-leilao\123.xlsx', index=False)
    df_teste = pl.from_pandas(df_merge)
    df_teste.write_excel(column_totals=True
        ,autofit=True
        ,workbook = r'C:\Users\peter\OneDrive\Área de Trabalho\precificacao-leilao\123.xlsx')  


def main():

    """MONTA BASE DE OUTUBRO"""
    print('Precificando Outubro...')
    df_inflacao = pd.read_excel(r"C:\Users\peter\Projetos\Projeção de Curva de Inflação (Premissa para modelagens) - Outubro.xlsx"
                         ,sheet_name='Curva de inflação'
                         ,skiprows=3
                         ,usecols='B:G')

    df_inflacao_filtered = inflacao_ajustada(df_inflacao)


    data_referencia = datetime.strptime('2024-10-09', '%Y-%m-%d')
    cdi_desagio = 0.1125
    spread_carrego = 0.1
    spread_ccr = 0.1

    df =  pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\PUBLICO_ALVO_CONSOLIDADO_2024-10-09.xlsx")
    df = precificacao_santander(df
                            ,df_inflacao_filtered
                            ,data_referencia
                            ,cdi_desagio
                            ,spread_carrego
                            ,spread_ccr)

    df_parceiro = pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\Resultado 2024-10-16.parceiros.xlsx"
                                ,header=None)
    df_parceiro.columns = pd.MultiIndex.from_frame(df_parceiro.iloc[:2].T)
    df_parceiro = df_parceiro[2:] 
    # Excluir Cotas que não obtiveram lances
    df_parceiro = df_parceiro[df_parceiro[(float('nan'), 'Propostas disponíveis')] > 0] 
    df_parceiro =  monta_parceiro(df_parceiro)
    df1  = oferta_parceiro(df
                        ,df_parceiro
                        ,1)
    df1_resumido = df1[['NR_CONTRATO','MODELO','OFERTANTE','MAIOR_OFERTA','MELHOR_OFERTA_AJUST_TIR','MELHOR_TIR_MERCADO_OFERTA']]


    """MONTA BASE DE NOVEMBRO"""
    print('\nPrecificando Novembro...')
    df_inflacao = pd.read_excel(r"C:\Users\peter\Projetos\Projeção de Curva de Inflação (Premissa para modelagens) - Novembro.xlsx"
                         ,sheet_name='Curva de inflação'
                         ,skiprows=3
                         ,usecols='B:G')

    df_inflacao_filtered = inflacao_ajustada(df_inflacao)
    data_referencia = datetime.strptime('2024-11-07', '%Y-%m-%d')
    cdi_desagio = 0.1125
    spread_carrego = 0.1
    spread_ccr = 0.1
    df2 =  pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\PUBLICO_ALVO_CONSOLIDADO_2024-11-07.xlsx")
    df2 = precificacao_santander(df2
                            ,df_inflacao_filtered
                            ,data_referencia
                            ,cdi_desagio
                            ,spread_carrego
                            ,spread_ccr)
    print('\nParceiros Novembro...')
    df_parceiro2 = pd.read_excel(r"C:\Users\peter\BRBB\Drive - Asset Management\FIDC I (Cotas canceladas)\Leilão Santander\Leilões antigos\Resultado\Resultado Leilão 2024-11-18.parceiros.xlsx"
                            ,header=None)

    # Set the first two rows as multi-level columns for the sample data
    df_parceiro2.columns = pd.MultiIndex.from_frame(df_parceiro2.iloc[:2].T)
    df_parceiro2 = df_parceiro2[2:]  # Exclude the header rows from the data
    df_parceiro2 = df_parceiro2[df_parceiro2[(float('nan'), 'Propostas disponíveis')] > 0]


    df_parceiro2 =  monta_parceiro(df_parceiro2)
    df2  = oferta_parceiro(df2
                    ,df_parceiro2
                    ,1)
    

    """MONTA DATAFRAME MERGE"""
    print('\nFazendo Merge...')
    mes = "OUTUBRO"
    df1_resumido.rename(columns={'MODELO':f'MODELO_{mes}'
                           ,'OFERTANTE':f'OFERTANTE_{mes}'
                           ,'MAIOR_OFERTA':f'MAIOR_OFERTA_{mes}'
                           ,'MELHOR_OFERTA_AJUST_TIR':f'MELHOR_OFERTA_AJUST_TIR_{mes}'
                           ,'MELHOR_TIR_MERCADO_OFERTA':f'MELHOR_TIR_MERCADO_OFERTA_{mes}'}
                           ,inplace=True)
    df_merge =df2.merge(df1_resumido, on='NR_CONTRATO', how='left')
    print('\nExportando para excel...')
    # df_split = df_merge.head()
    # df_split.to_excel(r'C:\Users\peter\Projetos\Leilão Santander\split.xlsx', index=False)
    # df_merge.to_excel('outubro_testee.xlsx', index=False)
    # print(f'\nArquivo outubro_testee.xlsx exportado!')

    """TRAZENDO TIR BRBB"""
    print("Montando TIR Ajustada BRBB\n")
    df_tir_brbb = oferta_parceiro_ajustado(df2
                    ,df1_resumido
                    ,1
                    ,'MELHOR_OFERTA_AJUST_TIR_OUTUBRO'
                    ,'MODELO_OUTUBRO')
    # print('Exportando teste...')
    # df_teste = pl.from_pandas(df_teste)
    # df_teste.write_excel(column_totals=True
    #     ,autofit=True
    #     ,workbook = r"C:\Users\peter\Projetos\Leilão Santander\teste_polars.xlsx")  
    # print('exportado')
    #df_merge =df_merge.merge(df_tir_brbb[['NR_CONTRATO', 'TIR_AJUSTADA_BRBB']], on='NR_CONTRATO', how='left')
    df_merge =df_merge.merge(df_tir_brbb, on='NR_CONTRATO', how='left')

    # Monta CDI SPREAD
    df_cdi = pd.read_excel(r"C:\Users\peter\Projetos\curva_cdi.xlsx", sheet_name='Novembro')

    df_merge["DIAS_LIQ_AJ"] = df_merge.apply(lambda x: 210 if x[f'MODELO_{mes}'] == "CCR" else (x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO'] if x[f'MODELO_{mes}'] == "CARREGO" else 0), axis=1)
    maior_dc = df_cdi['DC'].max()
    taxa_maior_dc = df_cdi[df_cdi['DC'] == df_cdi['DC'].max()]['TAXA INTERPOLADA']
    df_merge = df_merge.merge(df_cdi[['DC', 'TAXA INTERPOLADA']], right_on='DC', left_on='DIAS_LIQ_AJ', how='left')
    df_merge['TAXA INTERPOLADA'] = df_merge.apply(lambda x: taxa_maior_dc if x['DIAS_LIQ_AJ'] > maior_dc else x['TAXA INTERPOLADA'],axis=1)
    df_merge['TAXA INTERPOLADA'] = df_merge['TAXA INTERPOLADA'].fillna(0)
    #df_merge['CDI_PERIODO'] = df_merge.apply(lambda x: ((1+x['TAXA INTERPOLADA'])**(210/365))-1 if x[f'MODELO_{mes}'] == "CCR" else (((1+x['TAXA INTERPOLADA'])**(x['DIAS_CORRIDOS_DATABASE_LIQUIDACAO']/365))-1 if x[f'MODELO_{mes}'] == "CARREGO" else 0), axis=1 )
    df_merge['SPREAD_CDI'] = df_merge.apply(lambda x: ((1+x['TIR_AJUSTADA_BRBB'])/(1+x['TAXA INTERPOLADA']))-1 if x[f'MODELO_{mes}'] in ["CCR", "CARREGO"] else 0, axis=1 )
    df_merge.to_excel(r'03122024.xlsx', index=False)
    # df_teste = pl.from_pandas(df_merge)
    # df_teste.write_excel(column_totals=True
    #     ,autofit=True
    #     ,workbook = r"aplicacao_novembro.xlsx") 
    




    


if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):
        teste()
        #main()