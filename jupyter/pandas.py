from IPython.display import display
def previewDf(df, head=5):
    display(df.head(head))
    display(df.shape)
