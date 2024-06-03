import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert


@pytest.fixture
def fixture_run() -> tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]:
    service = ServiceMakeFeatures()
    df_filled = pd.read_csv(
        'tests/qa_training/domain/service_make_features/handle_violation/df_filled.csv')

    df_handled_expected = pd.read_csv(
        'tests/qa_training/domain/service_make_features/handle_violation/df_handled_expected.csv'
    )

    return service, df_filled, df_handled_expected


def test_run(fixture_run: tuple[
    ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
]):
    # Arrange
    service, df_filled, df_obeyed_expected = fixture_run

    # Act
    df_obeyed = service._handle_violations(df_filled=df_filled)

    # Assert
    MyAssert().assert_df(df_obeyed, df_obeyed_expected)
