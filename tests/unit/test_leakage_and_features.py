import pandas as pd
import pytest

from omnisupport.data.features import build_features
from omnisupport.data.leakage import FORBIDDEN_RAW_FIELDS as FORBIDDEN, LeakageError, assert_raw_columns_allowed


def test_forbidden_columns_are_rejected():
    for col in list(FORBIDDEN)[:5]:
        with pytest.raises(LeakageError):
            assert_raw_columns_allowed(["timestamp", col])


def test_build_features_is_stateless_and_deterministic():
    df = pd.DataFrame([{"timestamp": "2025-03-01T10:00:00", "support_channel": "email", "customer_region": "London", "customer_tenure_months": 10, "product_category": "home",
                        "order_value": 50.0, "delivery_delay_days": 3, "previous_ticket_count": 1, "issue_title": "Late", "issue_description": "Parcel 3 days late!!", "attachment_available": 0}])
    a, b = build_features(df), build_features(df)
    pd.testing.assert_frame_equal(a, b)
    assert not {"escalated", "resolution_time_hours", "csat_score"} & set(a.columns)
