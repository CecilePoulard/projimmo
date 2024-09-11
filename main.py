#main.py










#def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
 #   """
 #   - Query the raw dataset from Le Wagon's BigQuery dataset
 #   - Cache query result as a local CSV if it doesn't exist locally
#    - Process query data
#    - Store processed data on your personal BQ (truncate existing table if it exists)
 #   - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
#    """










#def train(
 #       min_date:str = '2009-01-01',
 #       max_date:str = '2015-01-01',
 #       split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
 #       learning_rate=0.0005,
 #       batch_size = 256,
 #       patience = 2
 #   ) -> float:
#
 #   """
 #   - Download processed data from your BQ table (or from cache if it exists)
 #   - Train on the preprocessed dataset (which should be ordered by date)
 #   - Store training results and model weights
#
 #   Return val_mae as a float
 #   """







#def evaluate(
#        min_date:str = '2014-01-01',
 #       max_date:str = '2015-01-01',
#        stage: str = "Production"
#    ) -> float:
#    """
 #   Evaluate the performance of the latest production model on processed data
#    Return MAE as a float
 #   """




#def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#"""
#    Make a prediction using the latest trained model
#    """



# if __name__=='__main__':
#   preprocess()
#   train()
#   evaluate()
#   pred()
