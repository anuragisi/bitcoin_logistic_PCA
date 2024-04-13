I have tried to simulate bitcoin datapoints into the PCA + Logistic Regression algorithm that has been explained in the below paper.

Its the implementation of a research paper named "Futures Quantitative Trading Strategies Based on Market Capital Flows by QiaoXu Qin, GengJian Zhou, WeiZhou Lin"

(https://doi.org/10.11114/aef.v5i2.3008)

#Analysis:
<br>
The given dataset has 935 rows x 20 columns.
<br>
<img width="438" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/89494b1c-8fd1-4abd-8a41-5a44592e5e40">
<br>


<img width="435" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/d09032ba-1e51-4089-afe7-95857f45c774">
<br>

<img width="540" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/7f37e0b3-8c59-4c6e-ae4c-4df4bdb6d0d6">
<br>

<img width="1236" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/0afb719b-6733-4308-ad92-d73db42a8fa2">
<br>

<img width="570" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/52a4b456-5fcf-402f-816f-956f8fdd46f7">
<br>

<img width="328" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/7aafd228-ef02-4278-9aa7-99c1a0a02668">
<br>
<img width="383" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/ca1bab02-a60d-4127-bfb2-9d13166f5989">
<br>

<img width="539" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/764b53b7-4fec-491d-98cd-623f46d2d46e">
<br>

<img width="1063" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/ef9e4192-3678-4e33-b463-01425ddab713">
<br>

<img width="411" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/04b638b4-048a-4266-b2df-9deeb834cd22">
<br>

<img width="811" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/8a894617-5a0c-4d02-a508-2c91e713b647">
<br>

<img width="546" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/9cabf11f-565a-4400-a014-06f49f27caf1">
<br>

<img width="704" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/663be280-b086-4efa-bd31-f1a0f72b872f">
<br>

<img width="554" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/f33ed687-7222-4677-980c-72ae05936630">
<br>

#Code:
  1. **Important Libraries**<br>
      import pandas as pd
      import numpy as np
      import seaborn as sns
      import matplotlib.pyplot as plt
      from sklearn.metrics import classification_report
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      from sklearn.linear_model import LogisticRegression

   2. **Loading Data & View Dataframe**<br>
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
   4. **Empirical Analysis of PCA**<br>
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
      
   6. **Empirical Analysis of Logistic Model**<br>
      y = data['signal'].map({'buy': 0, 'sell': 1, 'none': 2}).astype(int)  # Encoding signals  #dependent variable
      X_train, X_test, y_train, y_test = train_test_split(main_component_scores, y, test_size=0.2, random_state=42)
      y_logit = np.log(y / (1 - y))
      logistic_model = LogisticRegression()
      logistic_model.fit(X_train, y_train)
      y_pred_proba = logistic_model.predict_proba(X_test)
      print("Predicted Probabilities:")
      print(y_pred_proba)
      
   8. **Final Plotting results**<br>
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

      
#Findings:<br>
<img width="1107" alt="image" src="https://github.com/anuragprasad95/bitcoin_logistic_PCA/assets/3609255/de47b3e2-a84e-43f5-8b90-7336237d83c5">
