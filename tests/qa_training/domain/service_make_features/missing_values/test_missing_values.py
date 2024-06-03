import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert


@pytest.fixture
def fixture_run() -> tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]:
    service = ServiceMakeFeatures()
    df_customer_info = pd.read_csv(
        'tests/qa_training/domain/service_make_features/missing_values/df_customer_info.csv')
    df_filled_expected = pd.read_csv(
        'tests/qa_training/domain/service_make_features/missing_values/df_filled_expected.csv')

    return service, df_customer_info, df_filled_expected


def test_run(fixture_run: tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]):
    # Arrange
    service, df_customer_info, df_filled_expected = fixture_run

    # Act
    df_filled = service._handle_missing_values(df_customer_info=df_customer_info)

    # Assert
    MyAssert().assert_df(df_filled, df_filled_expected)
