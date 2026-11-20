"""
Dari data si aril
"""

SEED = 42
ID_COL = "id"
TARGET = "is_canceled"
CATEGORICAL = ["meal_package", "room_category", "booking_channel"]
NUMERIC = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

# Deterministic, target-independent features used by the data pipeline.
ENGINEERED = ["total_nights", "total_guests", "is_free"]

SKEWED = [
    "days_before_arrival",
    "nightly_rate_idr",
    "prior_cancellations",
    "prior_completed_bookings",
]

# Asumsi cost Bonus
COST_FP = 2.0
COST_FN = 1.0

