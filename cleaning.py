"""
STEP 1 - DATA CLEANING
Run:  python 01_cleaning.py [path_to_raw_csv]
Output: cleaned_ecommerce_data.csv  (used by 02_visualization.py)
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "Kaggle_Ecommerce_Data.csv"
OUT = "ecom_output"
os.makedirs(OUT, exist_ok=True)


raw = pd.read_csv("C:\\Users\\Vishwaranjal\\OneDrive\\Desktop\\code verse\\Kaggle_Ecommerce Data.csv")
df = raw.copy()
print("=" * 70, "\n1. DATA STRUCTURE\n" + "=" * 70)
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print(f"Unique orders: {df.order_id.nunique():,} | customers: {df.customer_id.nunique():,} | "
      f"products: {df.product_id.nunique():,}")
print("Grain: one row = one order line (order_id is unique).")


print("\n" + "=" * 70, "\n2. CLEANING & VALIDATION\n" + "=" * 70)

# 2a. Standardise text columns (trim, consistent case, fix known variants)
text_cols = ["category", "payment_method", "region", "returned", "return_reason", "customer_gender"]
for c in text_cols:
    df[c] = df[c].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
for c in ["category", "region", "customer_gender"]:
    df[c] = df[c].str.title()
df["returned"] = df["returned"].str.title().map({"Yes": "Yes", "Y": "Yes", "No": "No", "N": "No"})
df["payment_method"] = df["payment_method"].replace({"Cod": "COD", "Cash On Delivery": "COD",
                                                     "Credit card": "Credit Card", "Debit card": "Debit Card",
                                                     "Upi": "UPI", "Paypal": "PayPal"})
df["customer_id"] = df["customer_id"].astype(str).str.strip().str.upper()
df["order_id"] = df["order_id"].astype(str).str.strip().str.upper()
df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()

# 2b. Dates (file uses day/month/year)
for c in ["order_date", "delivered_date", "request_date"]:
    df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
print("Unparseable dates:", {c: int(df[c].isna().sum()) for c in ["order_date", "delivered_date"]},
      "(request_date blanks are expected = not returned)")

# 2c. Numeric validation
num_cols = ["price", "discount", "quantity", "total_amount", "shipping_cost", "profit_margin", "customer_age"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

issues = {
    "delivered before ordered": (df.delivered_date < df.order_date).sum(),
    "return requested before delivery": (df.request_date < df.delivered_date).sum(),
    "price <= 0": (df.price <= 0).sum(),
    "quantity <= 0": (df.quantity <= 0).sum(),
    "discount outside 0-1": ((df.discount < 0) | (df.discount > 1)).sum(),
    "age outside 15-100": ((df.customer_age < 15) | (df.customer_age > 100)).sum(),
    "negative shipping": (df.shipping_cost < 0).sum(),
}
for k, v in issues.items():
    print(f"  {k:<36}: {int(v)}")


df["calc_total"] = df.price * df.quantity * (1 - df.discount)
mismatch = (df.total_amount - df.calc_total).abs() > 0.02
print(f"  total_amount != price*qty*(1-disc)  : {int(mismatch.sum())}  -> total_amount EXCLUDES shipping")


print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
# Returns fields blank for non-returned orders is structural (not an error).
inconsistent = ((df.returned == "Yes") & df.request_date.isna()).sum() + ((df.returned == "No") & df.request_date.notna()).sum()
print(f"Return-flag vs request_date inconsistencies: {int(inconsistent)}")
df["return_reason"] = df["return_reason"].astype(object)
df.loc[(df.returned == "Yes") & df.return_reason.isna(), "return_reason"] = "Unknown"
df.loc[df.returned == "No", "return_reason"] = "Not Returned"

df = df.dropna(subset=["order_id", "customer_id", "order_date", "price", "quantity", "total_amount"])
df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())

print(f"\nExact duplicate rows : {df.duplicated().sum()}")
print(f"Duplicate order_ids  : {df.order_id.duplicated().sum()}")
print(f"Same cust+product+date repeats: {df.duplicated(['customer_id', 'product_id', 'order_date']).sum()}")
df = df.drop_duplicates().drop_duplicates("order_id", keep="first")

df["is_returned"] = (df.returned == "Yes").astype(int)
df["returned_amount"] = df.total_amount * df.is_returned
df["net_sales"] = df.total_amount - df.returned_amount
df["gross_before_discount"] = df.price * df.quantity
df["discount_amount"] = df.gross_before_discount - df.total_amount
df["delivery_days"] = (df.delivered_date - df.order_date).dt.days
df["return_lag_days"] = (df.request_date - df.delivered_date).dt.days
df["year_month"] = df.order_date.dt.to_period("M").dt.to_timestamp()
df["year"] = df.order_date.dt.year
df["quarter"] = df.order_date.dt.to_period("Q").astype(str)
df["weekday"] = df.order_date.dt.day_name()
df["month_num"] = df.order_date.dt.month
df["age_group"] = pd.cut(df.customer_age, [17, 25, 35, 45, 55, 70], labels=["18-25", "26-35", "36-45", "46-55", "56-69"])
df["price_band"] = pd.cut(df.price, [0, 20, 50, 100, 250, 500, np.inf],
                          labels=["<$20", "$20-50", "$50-100", "$100-250", "$250-500", "$500+"])
df["discount_band"] = pd.cut(df.discount, [-.01, 0, .05, .10, .20, .30],
                             labels=["0%", "1-5%", "6-10%", "11-20%", "21-30%"])
df["delivery_band"] = pd.cut(df.delivery_days, [-1, 3, 5, 7, 10, 100], labels=["0-3d", "4-5d", "6-7d", "8-10d", "10d+"])

df = df.rename(columns={"profit_margin": "profit"})
df["profit_pct_of_sales"] = df.profit / df.total_amount
df["net_profit"] = np.where(df.is_returned == 1, 0, df.profit)  

first_date, last_date = df.order_date.min(), df.order_date.max()
month_start = df.year_month
month_end = month_start + pd.offsets.MonthEnd(0)
df["is_partial_month"] = ((month_start < first_date) | (month_end > last_date)).astype(int)
print("\nPartial-month orders flagged:", int(df.is_partial_month.sum()), "| months:",
      sorted(df.loc[df.is_partial_month == 1, "year_month"].dt.strftime("%b %Y").unique()))

df.to_csv(f"{OUT}/cleaned_ecommerce_data.csv", index=False)
print(f"\nClean dataset: {len(df):,} rows saved -> {OUT}/cleaned_ecommerce_data.csv")

print("\nCleaning done. Now run 02_visualization.py")