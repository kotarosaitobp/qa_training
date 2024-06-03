import pandas as pd


class ServiceMakeFeatures:
    """前処理と特徴量作成する."""

    # 定数の定義
    DEFAULT_SEX = "male"
    DEFAULT_AGE = 20
    DEFAULT_EMBARKED = "S"
    DEFAULT_PCLASS = 2
    AGE_LOWER_BOUND = 0
    AGE_UPPER_BOUND = 130
    FARE_LOWER_BOUND = 0
    FARE_UPPER_BOUND = 1000
    VALID_PCLASS = [1, 2, 3]
    VALID_SEX = ["male", "female"]
    VALID_EMBARKED = ["C", "Q", "S"]
    FEATURE_COLUMNS = ["Sex", "Embarked", "Pclass", "Age", "Fare"]
    CATEGORICAL_COLUMNS = ["Embarked"]
    AGE_BINS = [0, 10, 18, 40, 64, float('inf')]
    AGE_LABELS = ['0-10', '11-18', '19-40', '41-64', '65+']

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
        df_customer_info["Sex"] = df_customer_info["Sex"].fillna(self.DEFAULT_SEX)
        df_customer_info["Age"] = df_customer_info["Age"].fillna(self.DEFAULT_AGE)
        df_customer_info["Embarked"] = df_customer_info["Embarked"].fillna(
            self.DEFAULT_EMBARKED
            )
        df_customer_info["Pclass"] = df_customer_info["Pclass"].fillna(
            self.DEFAULT_PCLASS
            )
        df_customer_info = df_customer_info.dropna(subset=self.FEATURE_COLUMNS)
        return df_customer_info

    def _handle_violations(self, df_filled) -> pd.DataFrame:
        """制約違反を処理する."""
        df_filled = df_filled[df_filled["Pclass"].isin(self.VALID_PCLASS)]
        df_filled = df_filled[df_filled["Sex"].isin(self.VALID_SEX)]
        df_filled = df_filled[
            (df_filled["Age"] >= self.AGE_LOWER_BOUND) &
            (df_filled["Age"] <= self.AGE_UPPER_BOUND) &
            (df_filled["Age"].apply(float.is_integer))
        ]
        df_filled = df_filled[
            (df_filled["Fare"] >= self.FARE_LOWER_BOUND) &
            (df_filled["Fare"] <= self.FARE_UPPER_BOUND) &
            (df_filled["Fare"].apply(lambda x: isinstance(x, float)))
        ]
        df_filled = df_filled[df_filled["Embarked"].isin(self.VALID_EMBARKED)]
        return df_filled

    def _make_features(self, df_obeyed: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作る."""
        df_obeyed = df_obeyed[
            self.FEATURE_COLUMNS
        ]
        df_obeyed.loc[:, "Sex"] = (
            df_obeyed["Sex"].replace({"male": 0, "female": 1}).astype("int64")
        )
        df_obeyed = pd.get_dummies(
            df_obeyed,
            columns=self.CATEGORICAL_COLUMNS,
            dtype=float
            )

        df_obeyed['AgeGroup'] = pd.cut(
            df_obeyed['Age'], bins=self.AGE_BINS, labels=self.AGE_LABELS, right=False)
        df_obeyed = pd.get_dummies(
            df_obeyed, columns=['AgeGroup'], prefix='Age', dtype=int)
        df_obeyed.drop(columns=['Age'], inplace=True)
        print(df_obeyed)
        return df_obeyed
