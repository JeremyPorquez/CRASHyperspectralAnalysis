def getRamanIndex(dataframe,column_index=1):
    if 'Raman' in dataframe.columns:
        return dataframe['Raman'].values
    elif 'X' in  dataframe.columns:
        return dataframe['X'].values
    else:
        return dataframe[dataframe.columns[column_index]].values


