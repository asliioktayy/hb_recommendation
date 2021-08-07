import json

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import pandas as pd

from get_data import events, meta


event_data = events.drop(["brand","category","subcategory","name", "eventtime", "event"], axis=1)

session_cnt_by_prdct = pd.DataFrame({'session_cnt_by_prdct': event_data.groupby(['productid'])['sessionid'].count()})

session_cnt_by_prdct.sort_values("session_cnt_by_prdct").quantile(0.80)
session_cnt_by_prdct=pd.DataFrame(session_cnt_by_prdct.loc[session_cnt_by_prdct.session_cnt_by_prdct > 30]).reset_index()
event_data = event_data.merge(session_cnt_by_prdct, on='productid')
event_data["price"] = event_data["price"].astype(float)

basket = (event_data.groupby(['sessionid', 'productid'])['price']
          .sum().unstack().reset_index().fillna(0)
          .set_index('sessionid'))

print("******************")

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets = basket_sets.fillna(0)

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

print(rules.sort_values(by="lift", ascending=False))
