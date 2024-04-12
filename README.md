I have tried to simulate bitcoin datapoints into the PCA + Logistic Regression algorithm that has been explained in the below paper.

Its the implementation of a research paper named "Futures Quantitative Trading Strategies Based on Market Capital Flows by QiaoXu Qin, GengJian Zhou, WeiZhou Lin"

(https://doi.org/10.11114/aef.v5i2.3008)

#Analysis:

The given dataset has 935 rows x 20 columns.
![Uploading image.png…]()


#Code:
  1. **Important Libraries**
      import pandas as pd
      import numpy as np
      import seaborn as sns
      import matplotlib.pyplot as plt
      from sklearn.metrics import classification_report
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      from sklearn.linear_model import LogisticRegression

   2. **Loading Data & View Dataframe**
      data = pd.read_csv('data.csv')
      df = pd.DataFrame(data)
      df
            
      df.shape
      
      df.info()
      
      len(df[df.duplicated()])
      
      print(df.isnull().sum())
      
      sns.heatmap(df.isnull(), cbar=False)
      
      df.columns
      
      df.describe(include='all')

      for i in df.columns.tolist():
      print("No. of unique values in ",i,"is",df[i].nunique(),".")
      
      df1 = df.drop(['signal'], axis=1)
      df1 = df1.iloc[:,1:13]
      print(df1.isnull().sum())
      df1
   4. **Empirical Analysis of PCA**
      scaler = StandardScaler()
      bitcoin_std = scaler.fit_transform(df1)
      bitcoin_std = pd.DataFrame(bitcoin_std, columns=df1.columns)
      
      print("Standardized Data:")
      print(bitcoin_std)
      correlation_matrix = bitcoin_std.corr()
      print("Sum of Diagonals = ",np.trace(correlation_matrix))
      eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
      print("Total sum of eigenvalues = ", sum(eigenvalues))
      
      eigenvalues = pd.DataFrame(eigenvalues, index=correlation_matrix.index, columns=['Eigenvalues'])
      eigenvectors = pd.DataFrame(eigenvectors, columns=df1.columns)
      print("\n")
      total_variance = eigenvalues['Eigenvalues'].sum()
      eigenvalues['Percentage of Variance'] = (eigenvalues['Eigenvalues'] / total_variance) * 100
      print(eigenvalues)
      plt.figure(figsize=(10, 6))
      plt.bar(eigenvalues.index, eigenvalues['Percentage of Variance'], color='skyblue')
      plt.xlabel('Principal Components')
      plt.ylabel('Percentage of Explained Variance')
      plt.title('Percentage of Variance Explained by Variables')
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
      
      print("\nEigenvectors:")
      print(eigenvectors)
      eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
      sorted_indices = np.argsort(eigenvalues)[::-4]
      sorted_eigenvalues = eigenvalues[sorted_indices]
      principal_components = eigenvectors[:, sorted_indices]
      main_component_scores = np.dot(bitcoin_std, principal_components)
      print("Principal Component Scores of dimension =",main_component_scores.shape)
      main_component_scores
      
   6. **Empirical Analysis of Logistic Model**
      y = data['signal'].map({'buy': 0, 'sell': 1, 'none': 2}).astype(int)  # Encoding signals  #dependent variable
      X_train, X_test, y_train, y_test = train_test_split(main_component_scores, y, test_size=0.2, random_state=42)
      y_logit = np.log(y / (1 - y))
      logistic_model = LogisticRegression()
      logistic_model.fit(X_train, y_train)
      y_pred_proba = logistic_model.predict_proba(X_test)
      print("Predicted Probabilities:")
      print(y_pred_proba)
      
   8. **Final Plotting results**
      df['datetime'] = pd.to_datetime(df['datetime'])
      
      plt.figure(figsize=(14, 7))
      plt.plot(df['datetime'][:150], df['close'][:150], label='BTC Close Price', color='skyblue', linewidth=2)
      buy_signals = df.iloc[:150][y_pred_proba[:150, 0] < 0.5]  # Assuming 0 represents 'buy'
      plt.scatter(buy_signals['datetime'], buy_signals['close'], label='Buy Signal', marker='^', color='green', alpha=1, s=100)
      sell_signals = df.iloc[:150][y_pred_proba[:150, 1] > 0.5]  # Assuming 1 represents 'sell'
      plt.scatter(sell_signals['datetime'], sell_signals['close'], label='Sell Signal', marker='v', color='red', alpha=1, s=100)
      plt.title('BTC Price with Predicted Buy/Sell Signals (First 150 points)')
      plt.xlabel('Date')
      plt.ylabel('BTC Close Price')
      plt.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()

      
#Findings:
