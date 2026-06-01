import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRecommender:
    def fit(self, df):
        self.catalog = df.groupby("product_id").first().reset_index()
        self.products = self.catalog["product_id"].tolist()

        cats = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(self.catalog[["category", "subcategory", "brand"]])
        nums = MinMaxScaler().fit_transform(self.catalog[["final_price", "rating", "discount"]])

        features = list(cats.T) + list(nums.T)
        features = pd.DataFrame(features).T.values

        self.similarity = cosine_similarity(features)

        self.bought = df.groupby("user_id")["product_id"].apply(list).to_dict()

        return self
    
    def recommend(self, user_id, n=5):
        owned = self.bought.get(user_id, [])
        if not owned:
             top = self.catalog.sort_values("rating", ascending=False)
             return top.head(n)[["product_id", "category", "brand", "rating"]]
        
        idx = [self.products.index(p) for p in owned]
        scores = self.similarity[idx].sum(axis=0)

        ranked = scores.argsort()[::-1]
        picks = [i for i in ranked if self.products[i] not in owned][:n]
        return self.catalog.iloc[picks][
            ["product_id", "category", "subcategory", "brand", "final_price", "rating"]
        ]
    
if __name__ == "__main__":
    df = pd.read_csv("./data.csv")
    rec = SimpleRecommender().fit(df)

    print(rec.recommend("U000000").to_string(index=False))

    features = pd.concat([pd.DataFrame(cats), pd.DataFrame(nums)])
