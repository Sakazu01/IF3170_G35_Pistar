"""Konstanta bersama. Ubah di sini kalau nama kolom atau seed berubah."""

SEED = 42  # dipakai di semua tempat yang ada angka acak

ID_COL = "id"
TARGET = "is_canceled"  # 1 = batal (kelas positif)

CATEGORICAL = [
    "meal_package",
    "room_category",
    "booking_channel",
]

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

# Fitur buatan, dihitung di Preprocessor._engineer
ENGINEERED = [
    "total_nights",
    "total_guests",
    "is_free",
]

# Kolom yang miring ke kanan, di-log1p di matrix view
SKEWED = [
    "days_before_arrival",
    "nightly_rate_idr",
    "prior_cancellations",
    "prior_completed_bookings",
]

# Biaya salah prediksi: FP dihitung 2x lebih mahal dari FN
COST_FP = 2.0
COST_FN = 1.0
