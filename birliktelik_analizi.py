import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv("/Users/batuhanozdogan/Downloads/Online Retail.csv")

veri=data.copy()
veri=veri.dropna()
veri=veri[~veri["InvoiceNo"].str.contains("C")]

veri2=veri["Country"].value_counts()
ulke=veri[veri["Country"]=="United Kingdom"]
sepet=ulke.iloc[:,[0,2,3]]

sepet=sepet.groupby(["InvoiceNo","Description"])["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")

def num(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

sepetson = sepet.apply(lambda x: x.map(num))

df1=apriori(sepetson.astype("bool"),min_support=0.02,use_colnames=True)
df2=association_rules(df1,metric="lift",min_threshold=1)
print(df2)
