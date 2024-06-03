import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert


@pytest.fixture
def fixture_run() -> tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]:
    service = ServiceMakeFeatures()
    df_handled = pd.read_csv(
        'tests/qa_training/domain/service_make_features/make_features/df_handled.csv'
    )
    df_obeyed_expected = pd.read_csv(
        'tests/qa_training/domain/service_make_features/make_features/df_obeyed_expected.csv'
    )

    return service, df_handled, df_obeyed_expected


def test_run(fixture_run: tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]):
    # Arrange
    service, df_handled, df_obeyed_expected = fixture_run

    # Act
    df_obeyed = service._make_features(df_handled)

    # Assert
    MyAssert().assert_df(df_obeyed, df_obeyed_expected)
