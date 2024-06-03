import pandas as pd


class ServiceMakeFeatures:
    """前処理と特徴量作成する."""

    def run(
        self, df_customer_info: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        前処理と特徴量作成を実行する.

        Args:
            df_customer_info (pd.DataFrame): 乗員情報のdf.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 乗員IDのdf, 特徴量のdf, 正解のdf
        """
        df_X, df_id = self._make_X(df_customer_info)
        df_y = self._make_y(df_id=df_id, df_customer_info=df_customer_info)
        return df_id, df_X, df_y

    def _make_X(
        self, df_customer_info: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """特徴量のdfとidを作成する."""
        # 欠損値処理する
        df_filled = self._handle_missing_values(df_customer_info=df_customer_info)

        # 制約違反の行を捨てる
        df_obeyed = self._handle_violations(df_filled=df_filled)

        # 特徴量作成
        df_X_and_id = self._make_features(df_obeyed=df_obeyed).reset_index(drop=True)

        df_X = df_X_and_id.drop("PassengerId", axis=1)
        df_id = df_X_and_id[["PassengerId"]]

        return df_X, df_id

    def _make_y(
        self, df_id: pd.DataFrame, df_customer_info: pd.DataFrame
    ) -> pd.DataFrame:
        """正解のdfを作成する."""
        df_y = pd.merge(df_id, df_customer_info, on="PassengerId", how="inner")

        if "Survived" not in df_y.columns:
            return pd.DataFrame()

        df_y = df_y[["Survived"]]
        return df_y.reset_index(drop=True)

    def _handle_missing_values(self, df_customer_info) -> pd.DataFrame:
        """欠損値処理する."""
        df_customer_info["Sex"] = df_customer_info["Sex"].fillna("male")
        df_customer_info["Age"] = df_customer_info["Age"].fillna(20)
        df_customer_info["Embarked"] = df_customer_info["Embarked"].fillna("S")
        df_customer_info["Pclass"] = df_customer_info["Pclass"].fillna(2)
        df_customer_info = df_customer_info.dropna(subset=[
            "Sex",
            "Embarked",
            "Pclass",
            "Age",
            "Fare"
        ])
        return df_customer_info

    def _handle_violations(self, df_filled) -> pd.DataFrame:
        """制約違反を処理する."""
        df_filled = df_filled[df_filled["Pclass"].isin([1, 2, 3])]
        df_filled = df_filled[df_filled["Sex"].isin(["male", "female"])]
        df_filled = df_filled[
            (df_filled["Age"] >= 0) &
            (df_filled["Age"] <= 130) &
            (df_filled["Age"].apply(float.is_integer))
        ]
        df_filled = df_filled[
            (df_filled["Fare"] >= 0) &
            (df_filled["Fare"] <= 1000) &
            (df_filled["Fare"].apply(lambda x: isinstance(x, float)))
        ]
        df_filled = df_filled[df_filled["Embarked"].isin(["C", "Q", "S"])]
        return df_filled

    def _make_features(self, df_obeyed: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作る."""
        df_obeyed = df_obeyed[
            ["Sex", "Embarked", "Pclass", "Age", "Fare"]
        ]
        df_obeyed.loc[:, "Sex"] = (
            df_obeyed["Sex"].replace({"male": 0, "female": 1}).astype("int64")
        )
        df_obeyed = pd.get_dummies(df_obeyed, columns=["Embarked"], dtype=float)
        return df_obeyed
