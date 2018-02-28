def getRamanIndex(dataframe,column_index=1):
    if 'Raman' in dataframe.columns:
        return dataframe['Raman'].values
    else:
        return dataframe[dataframe.columns[column_index]].values


