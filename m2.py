import numpy as np
import pandas as pd
import Tkinter
import matplotlib.pyplot as pyt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df=pd.read_csv("Ecommerce Customers")

X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=df[['Yearly Amount Spent']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
ln=LinearRegression()
ln.fit(X_train,y_train)
predicted=ln.predict(X_test)
print metrics.mean_absolute_error(y_test,predicted)

sns.set_style("whitegrid")
sns.pairplot(data=df)
# z.savefig("text.pdf")
pyt.show()
