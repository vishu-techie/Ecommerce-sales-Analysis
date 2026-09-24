"""
STEP 2 - VISUALIZATION & ANALYSIS
Run:  python 02_visualization.py     (after 01_cleaning.py)
Input : ecom_output/cleaned_ecommerce_data.csv
Output: PNG dashboards in ./ecom_output/ + key findings in console
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")
OUT = "ecom_output"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "axes.titlesize": 12,
    "axes.titleweight": "bold", "font.size": 9,
})
C1, C2, C3 = "#2E5EAA", "#E4572E", "#76B041"   
money = mtick.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K" if abs(x) >= 1e3 else f"${x:.0f}")
pct = mtick.PercentFormatter(1.0, decimals=1)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.png")


df = pd.read_csv(f"{OUT}/cleaned_ecommerce_data.csv",
                 parse_dates=["order_date", "delivered_date", "request_date", "year_month"])
df["age_group"] = pd.Categorical(df.age_group, ["18-25", "26-35", "36-45", "46-55", "56-69"], ordered=True)
df["price_band"] = pd.Categorical(df.price_band, ["<$20", "$20-50", "$50-100", "$100-250", "$250-500", "$500+"], ordered=True)
df["discount_band"] = pd.Categorical(df.discount_band, ["0%", "1-5%", "6-10%", "11-20%", "21-30%"], ordered=True)
df["delivery_band"] = pd.Categorical(df.delivery_band, ["0-3d", "4-5d", "6-7d", "8-10d", "10d+"], ordered=True)

# Complete months only -> used for ALL time-trend charts (KPIs & other charts use full df)
df_full = df[df.is_partial_month == 0]
partial_months = df.loc[df.is_partial_month == 1, "year_month"].unique()


gross = df.total_amount.sum()
returned_val = df.returned_amount.sum()
kpi = {
    "Gross Sales": gross, "Returned Value": returned_val, "Net Sales": df.net_sales.sum(),
    "Orders": len(df), "Customers": df.customer_id.nunique(),
    "AOV": df.total_amount.mean(), "Return Rate (orders)": df.is_returned.mean(),
    "Return Rate (value)": returned_val / gross, "Total Profit": df.profit.sum(),
    "Avg Delivery Days": df.delivery_days.mean(),
}
print("\n" + "=" * 70, "\n3. KPIs\n" + "=" * 70)
for k, v in kpi.items():
    print(f"  {k:<22}: {v:,.2f}" if v > 1 else f"  {k:<22}: {v:.2%}")

# ---------- Reusable aggregations ----------
def summarise(by):
    g = df.groupby(by, observed=True).agg(
        orders=("order_id", "count"), sales=("total_amount", "sum"), net_sales=("net_sales", "sum"),
        returned_value=("returned_amount", "sum"), profit=("profit", "sum"),
        return_rate=("is_returned", "mean"), aov=("total_amount", "mean"), units=("quantity", "sum"),
        avg_discount=("discount", "mean"))
    g["return_value_rate"] = g.returned_value / g.sales
    g["profit_pct"] = g.profit / g.sales
    return g

cat, reg, pay = summarise("category"), summarise("region"), summarise("payment_method")
monthly = summarise("year_month")
# partial months (Sep 2023 & Sep 2025) are excluded from trend charts
monthly = monthly[~monthly.index.isin(partial_months)]
overall_rr = df.is_returned.mean()

# =====================================================================
# 4. DASHBOARD 1 - EXECUTIVE OVERVIEW
# =====================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle("E-commerce Executive Dashboard", fontsize=18, fontweight="bold", y=.99)
gs = fig.add_gridspec(4, 5, hspace=.75, wspace=.7, height_ratios=[.7, 1.5, 1.5, 1.5])

tiles = [("Gross Sales", money(gross, 0), C1), ("Net Sales", money(kpi["Net Sales"], 0), C3),
         ("Orders", f"{len(df):,}", C1), ("Return Rate", f"{overall_rr:.1%}", C2),
         ("Total Profit", money(kpi["Total Profit"], 0), C3)]
for i, (t, v, col) in enumerate(tiles):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=col, alpha=.12, transform=ax.transAxes))
    ax.text(.5, .62, v, ha="center", va="center", fontsize=20, fontweight="bold", color=col)
    ax.text(.5, .18, t, ha="center", va="center", fontsize=10, color="#444")

ax = fig.add_subplot(gs[1, :3])
ax.plot(monthly.index, monthly.sales, marker="o", color=C1, label="Gross sales")
ax.plot(monthly.index, monthly.net_sales, marker="o", color=C3, label="Net sales (after returns)")
ax.fill_between(monthly.index, monthly.net_sales, monthly.sales, color=C2, alpha=.2, label="Returned value")
ax.yaxis.set_major_formatter(money); ax.set_title("Monthly Sales Trend (full months only)"); ax.legend(frameon=False)
ax.tick_params(axis="x", rotation=30)

ax = fig.add_subplot(gs[1, 3:])
s = cat.sort_values("sales")
ax.barh(s.index, s.sales, color=C1); ax.xaxis.set_major_formatter(money); ax.set_title("Sales by Category")

ax = fig.add_subplot(gs[2, :2])
ax.bar(reg.index, reg.sales, color=C1); ax.yaxis.set_major_formatter(money); ax.set_title("Sales by Region")
ax = fig.add_subplot(gs[2, 2:4])
s = pay.sort_values("orders", ascending=False)
ax.bar(s.index, s.orders, color=C1); ax.set_title("Orders by Payment Method"); ax.tick_params(axis="x", rotation=25)
ax = fig.add_subplot(gs[2, 4])
g = df.customer_gender.value_counts()
ax.pie(g, labels=g.index, autopct="%1.0f%%", colors=[C1, C2, "#999"], startangle=90); ax.set_title("Gender"); ax.grid(False)

ax = fig.add_subplot(gs[3, :2])
s = cat.sort_values("return_rate", ascending=False)
ax.bar(s.index, s.return_rate, color=[C2 if v > overall_rr else "#999" for v in s.return_rate])
ax.axhline(overall_rr, color="k", ls="--", lw=1, label=f"Avg {overall_rr:.1%}")
ax.yaxis.set_major_formatter(pct); ax.set_title("Return Rate by Category"); ax.legend(frameon=False, loc="upper center"); ax.tick_params(axis="x", rotation=25)
ax = fig.add_subplot(gs[3, 2:4])
rr = df[df.is_returned == 1].return_reason.value_counts()
ax.barh(rr.index[::-1], rr.values[::-1], color=C2); ax.set_title("Return Reasons")
ax = fig.add_subplot(gs[3, 4])
ax.hist(df.delivery_days, bins=range(0, int(df.delivery_days.max()) + 2), color=C1, rwidth=.85)
ax.set_title("Delivery Days")
save(fig, "01_executive_dashboard")


fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Sales Performance", fontsize=16, fontweight="bold")

q = df_full.groupby("quarter").agg(sales=("total_amount", "sum"), orders=("order_id", "count"),
                                   months=("year_month", "nunique"))
q = q[q.months == 3]          # keep only quarters with all 3 full months
axs[0, 0].bar(q.index, q.sales, color=C1); axs[0, 0].yaxis.set_major_formatter(money)
axs[0, 0].set_title("Quarterly Sales (complete quarters)"); axs[0, 0].tick_params(axis="x", rotation=30)

ms = df_full.groupby(["month_num", "year"]).total_amount.sum().unstack()
for y in ms.columns:
    axs[0, 1].plot(ms.index, ms[y], marker="o", label=str(y))
axs[0, 1].set_xticks(range(1, 13)); axs[0, 1].yaxis.set_major_formatter(money)
axs[0, 1].set_title("Seasonality: Sales by Month & Year (full months)"); axs[0, 1].legend(frameon=False, title="Year")

wd = df.groupby("weekday").total_amount.sum().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
axs[0, 2].bar(wd.index.str[:3], wd.values, color=C1); axs[0, 2].yaxis.set_major_formatter(money); axs[0, 2].set_title("Sales by Weekday")

mom = monthly.sales.pct_change()
axs[1, 0].bar(mom.index, mom.values, width=20, color=[C3 if v >= 0 else C2 for v in mom.fillna(0)])
axs[1, 0].yaxis.set_major_formatter(pct); axs[1, 0].set_title("Month-over-Month Sales Growth"); axs[1, 0].tick_params(axis="x", rotation=30)

cm = df_full.pivot_table(index="category", columns="year_month", values="total_amount", aggfunc="sum")
cm.T.plot(ax=axs[1, 1], lw=1.5); axs[1, 1].yaxis.set_major_formatter(money)
axs[1, 1].set_title("Monthly Sales by Category"); axs[1, 1].legend(fontsize=7, ncol=2, frameon=False); axs[1, 1].set_xlabel("")

pm = cat.sort_values("profit_pct")
axs[1, 2].barh(pm.index, pm.profit_pct, color=[C2 if v < pm.profit_pct.mean() else C3 for v in pm.profit_pct])
axs[1, 2].xaxis.set_major_formatter(pct); axs[1, 2].set_title("Profit as % of Sales, by Category")
plt.tight_layout(); save(fig, "02_sales_performance")


fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Product & Category Performance", fontsize=16, fontweight="bold")

s = cat.sort_values("sales", ascending=False)
x = np.arange(len(s)); w = .4
axs[0, 0].bar(x - w/2, s.sales, w, label="Gross sales", color=C1)
axs[0, 0].bar(x + w/2, s.profit, w, label="Profit", color=C3)
axs[0, 0].set_xticks(x); axs[0, 0].set_xticklabels(s.index, rotation=25); axs[0, 0].yaxis.set_major_formatter(money)
axs[0, 0].legend(frameon=False); axs[0, 0].set_title("Sales vs Profit by Category")

axs[0, 1].bar(cat.index, cat.aov, color=C1); axs[0, 1].set_title("Average Order Value by Category")
axs[0, 1].tick_params(axis="x", rotation=25); axs[0, 1].yaxis.set_major_formatter(money)

sc = axs[0, 2].scatter(cat.sales, cat.return_rate, s=cat.orders / 12, c=cat.profit_pct, cmap="RdYlGn", edgecolor="k")
for n, r in cat.iterrows():
    axs[0, 2].annotate(n, (r.sales, r.return_rate), xytext=(6, 6), textcoords="offset points", fontsize=8)
axs[0, 2].xaxis.set_major_formatter(money); axs[0, 2].yaxis.set_major_formatter(pct)
axs[0, 2].set_xlabel("Sales"); axs[0, 2].set_ylabel("Return rate")
axs[0, 2].set_title("Category Map: size=orders, colour=profit %"); plt.colorbar(sc, ax=axs[0, 2], label="Profit %")

prod = df.groupby("product_id").agg(sales=("total_amount", "sum"), orders=("order_id", "count"),
                                    returns=("is_returned", "sum"), category=("category", "first"))
top = prod.sort_values("sales", ascending=False).head(10)
axs[1, 0].barh(top.index[::-1], top.sales[::-1], color=C1); axs[1, 0].xaxis.set_major_formatter(money)
axs[1, 0].set_title("Top 10 Products by Sales")

topr = prod[prod.orders >= 3].assign(rr=lambda d: d.returns / d.orders).sort_values(["rr", "orders"], ascending=False).head(10)
axs[1, 1].barh(topr.index[::-1], topr.rr[::-1], color=C2); axs[1, 1].xaxis.set_major_formatter(pct)
axs[1, 1].set_title("Highest Return-Rate Products (>=3 orders)")

pb = summarise("price_band")
ax2 = axs[1, 2].twinx()
axs[1, 2].bar(pb.index, pb.sales, color=C1, alpha=.8); axs[1, 2].yaxis.set_major_formatter(money)
ax2.plot(pb.index, pb.return_rate, color=C2, marker="o"); ax2.yaxis.set_major_formatter(pct); ax2.grid(False)
axs[1, 2].set_title("Sales (bars) & Return Rate (line) by Price Band"); axs[1, 2].tick_params(axis="x", rotation=25)
plt.tight_layout(); save(fig, "03_product_category")

cust = df.groupby("customer_id").agg(orders=("order_id", "count"), spend=("total_amount", "sum"),
                                     returns=("is_returned", "sum"), first=("order_date", "min"), last=("order_date", "max"))
cust["return_rate"] = cust.returns / cust.orders
snap = df.order_date.max() + pd.Timedelta(days=1)
cust["recency"] = (snap - cust["last"]).dt.days
cust["segment"] = pd.cut(cust.orders, [0, 1, 3, 6, 100], labels=["1 order", "2-3 orders", "4-6 orders", "7+ orders"])

fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Customer Purchasing Behaviour", fontsize=16, fontweight="bold")

oc = cust.orders.value_counts().sort_index()
axs[0, 0].bar(oc.index, oc.values, color=C1); axs[0, 0].set_title("Orders per Customer"); axs[0, 0].set_xlabel("# orders")
one = (cust.orders == 1).mean()
axs[0, 0].text(.95, .9, f"{1 - one:.0%} are repeat customers", transform=axs[0, 0].transAxes, ha="right")

ag = summarise("age_group")
axs[0, 1].bar(ag.index, ag.sales, color=C1); axs[0, 1].yaxis.set_major_formatter(money)
a2 = axs[0, 1].twinx(); a2.plot(ag.index, ag.aov, color=C2, marker="o"); a2.grid(False); a2.yaxis.set_major_formatter(money)
axs[0, 1].set_title("Sales (bars) & AOV (line) by Age Group")

gd = summarise("customer_gender")
x = np.arange(len(gd))
axs[0, 2].bar(x - .2, gd.aov, .4, color=C1, label="AOV"); axs[0, 2].set_xticks(x); axs[0, 2].set_xticklabels(gd.index)
a3 = axs[0, 2].twinx(); a3.bar(x + .2, gd.return_rate, .4, color=C2, label="Return rate"); a3.yaxis.set_major_formatter(pct); a3.grid(False)
axs[0, 2].set_title("AOV (blue) & Return Rate (red) by Gender")

sp = cust.spend.sort_values(ascending=False).reset_index(drop=True)
cum = sp.cumsum() / sp.sum()
axs[1, 0].plot(np.arange(1, len(cum) + 1) / len(cum), cum.values, color=C1)
axs[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
top20 = cum.iloc[int(len(cum) * .2)]
axs[1, 0].axvline(.2, color=C2, ls=":"); axs[1, 0].text(.22, .3, f"Top 20% of customers\n= {top20:.0%} of sales", color=C2)
axs[1, 0].xaxis.set_major_formatter(pct); axs[1, 0].yaxis.set_major_formatter(pct)
axs[1, 0].set_title("Customer Concentration (Pareto)")

ap = df.pivot_table(index="age_group", columns="category", values="total_amount", aggfunc="sum")
ap = ap.div(ap.sum(axis=1), axis=0)
im = axs[1, 1].imshow(ap.values, cmap="Blues", aspect="auto")
axs[1, 1].set_xticks(range(ap.shape[1])); axs[1, 1].set_xticklabels(ap.columns, rotation=30)
axs[1, 1].set_yticks(range(ap.shape[0])); axs[1, 1].set_yticklabels(ap.index); axs[1, 1].grid(False)
for i in range(ap.shape[0]):
    for j in range(ap.shape[1]):
        axs[1, 1].text(j, i, f"{ap.values[i, j]:.0%}", ha="center", va="center", fontsize=8)
axs[1, 1].set_title("Category Mix by Age Group")

seg = cust.groupby("segment", observed=True).agg(customers=("spend", "size"), spend=("spend", "sum"), rr=("return_rate", "mean"))
axs[1, 2].bar(seg.index, seg.spend, color=C1); axs[1, 2].yaxis.set_major_formatter(money)
a4 = axs[1, 2].twinx(); a4.plot(seg.index, seg.rr, color=C2, marker="o"); a4.yaxis.set_major_formatter(pct); a4.grid(False)
axs[1, 2].set_title("Spend (bars) & Avg Return Rate (line) by Frequency Segment")
plt.tight_layout(); save(fig, "04_customer_behaviour")

ret = df[df.is_returned == 1]
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Return Behaviour", fontsize=16, fontweight="bold")

axs[0, 0].plot(monthly.index, monthly.return_rate, marker="o", color=C2)
axs[0, 0].axhline(overall_rr, color="k", ls="--", lw=1); axs[0, 0].yaxis.set_major_formatter(pct)
axs[0, 0].set_title("Monthly Return Rate"); axs[0, 0].tick_params(axis="x", rotation=30)

rc = ret.pivot_table(index="category", columns="return_reason", values="order_id", aggfunc="count", fill_value=0)
rc = rc.div(rc.sum(axis=1), axis=0)
rc.plot(kind="barh", stacked=True, ax=axs[0, 1], colormap="Set2"); axs[0, 1].xaxis.set_major_formatter(pct)
axs[0, 1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1)); axs[0, 1].set_title("Return Reason Mix by Category"); axs[0, 1].set_ylabel("")

axs[0, 2].hist(ret.return_lag_days, bins=30, color=C2, rwidth=.9)
axs[0, 2].axvline(ret.return_lag_days.median(), color="k", ls="--")
axs[0, 2].set_title(f"Days from Delivery to Return Request (median {ret.return_lag_days.median():.0f})")

reg_cat = df.pivot_table(index="category", columns="region", values="is_returned", aggfunc="mean")
im = axs[1, 0].imshow(reg_cat.values, cmap="Reds", aspect="auto")
axs[1, 0].set_xticks(range(reg_cat.shape[1])); axs[1, 0].set_xticklabels(reg_cat.columns)
axs[1, 0].set_yticks(range(reg_cat.shape[0])); axs[1, 0].set_yticklabels(reg_cat.index); axs[1, 0].grid(False)
for i in range(reg_cat.shape[0]):
    for j in range(reg_cat.shape[1]):
        axs[1, 0].text(j, i, f"{reg_cat.values[i, j]:.1%}", ha="center", va="center", fontsize=8)
axs[1, 0].set_title("Return Rate: Category x Region (drill-down)")

axs[1, 1].bar(pay.index, pay.return_rate, color=C2); axs[1, 1].axhline(overall_rr, color="k", ls="--", lw=1)
axs[1, 1].yaxis.set_major_formatter(pct); axs[1, 1].set_title("Return Rate by Payment Method"); axs[1, 1].tick_params(axis="x", rotation=25)

rv = ret.groupby("return_reason").total_amount.sum().sort_values()
axs[1, 2].barh(rv.index, rv.values, color=C2); axs[1, 2].xaxis.set_major_formatter(money)
axs[1, 2].set_title("Value Lost to Returns by Reason")
plt.tight_layout(); save(fig, "05_return_behaviour")

fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Relationships Between Sales Drivers and Returns", fontsize=16, fontweight="bold")

def rr_plot(ax, col, title, rot=0):
    t = df.groupby(col, observed=True).is_returned.agg(["mean", "count"])
    ax.bar(t.index.astype(str), t["mean"], color=C2)
    ax.axhline(overall_rr, color="k", ls="--", lw=1)
    ax.yaxis.set_major_formatter(pct); ax.set_title(title); ax.tick_params(axis="x", rotation=rot)
    for i, (m, n) in enumerate(zip(t["mean"], t["count"])):
        ax.text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=7)

rr_plot(axs[0, 0], "discount_band", "Return Rate by Discount Level")
rr_plot(axs[0, 1], "quantity", "Return Rate by Quantity per Order")
rr_plot(axs[0, 2], "delivery_band", "Return Rate by Delivery Time")
rr_plot(axs[1, 0], "price_band", "Return Rate by Price Band", 25)
rr_plot(axs[1, 1], "age_group", "Return Rate by Age Group")

num = df[["price", "discount", "quantity", "total_amount", "shipping_cost", "profit",
          "customer_age", "delivery_days", "is_returned"]].corr()
im = axs[1, 2].imshow(num.values, cmap="coolwarm", vmin=-1, vmax=1); axs[1, 2].grid(False)
axs[1, 2].set_xticks(range(len(num))); axs[1, 2].set_xticklabels(num.columns, rotation=60, ha="right", fontsize=7)
axs[1, 2].set_yticks(range(len(num))); axs[1, 2].set_yticklabels(num.columns, fontsize=7)
for i in range(len(num)):
    for j in range(len(num)):
        axs[1, 2].text(j, i, f"{num.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
axs[1, 2].set_title("Correlation Matrix"); plt.colorbar(im, ax=axs[1, 2], fraction=.046)
plt.tight_layout(); save(fig, "06_sales_vs_returns")


fig, axs = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Unusual Observations & Outliers", fontsize=16, fontweight="bold")

q1, q3 = df.total_amount.quantile([.25, .75]); upper = q3 + 3 * (q3 - q1)
big = df[df.total_amount > upper]
axs[0, 0].boxplot([df[df.category == c].total_amount for c in cat.index], labels=cat.index, showfliers=True,
                  flierprops=dict(marker=".", markersize=2, alpha=.4))
axs[0, 0].set_yscale("log"); axs[0, 0].tick_params(axis="x", rotation=30)
axs[0, 0].set_title("Order Value Distribution by Category (log)")

axs[0, 1].hist(np.log10(df.total_amount), bins=60, color=C1)
axs[0, 1].axvline(np.log10(upper), color=C2, ls="--"); axs[0, 1].set_title(f"Order Value (log10); {len(big)} extreme orders > ${upper:,.0f}")

neg = df[df.profit < 0]
nc = neg.groupby("category").size() / df.groupby("category").size()
axs[0, 2].bar(nc.index, nc.values, color=C2); axs[0, 2].yaxis.set_major_formatter(pct)
axs[0, 2].set_title(f"Loss-Making Orders ({len(neg) / len(df):.1%}) by Category"); axs[0, 2].tick_params(axis="x", rotation=30)

axs[1, 0].scatter(df.discount, df.profit_pct_of_sales.clip(-1, 1), s=3, alpha=.15, color=C1)
axs[1, 0].set_xlabel("Discount"); axs[1, 0].set_ylabel("Profit / sales"); axs[1, 0].set_title("Discount vs Profit %")

bigc = cust.sort_values("spend", ascending=False).head(10)
axs[1, 1].barh(bigc.index[::-1], bigc.spend[::-1], color=C1); axs[1, 1].xaxis.set_major_formatter(money)
axs[1, 1].set_title("Top 10 Customers by Spend")

serial = cust[(cust.orders >= 4)].sort_values("return_rate", ascending=False).head(10)
axs[1, 2].barh(serial.index[::-1], serial.return_rate[::-1], color=C2); axs[1, 2].xaxis.set_major_formatter(pct)
axs[1, 2].set_title("Highest Return-Rate Customers (>=4 orders)")
plt.tight_layout(); save(fig, "07_unusual_observations")


print("\n" + "=" * 70, "\n4. KEY FINDINGS\n" + "=" * 70)
def top(s, n=1, asc=False): return s.sort_values(ascending=asc).head(n)

print(f"* Gross sales ${gross:,.0f}; net of returns ${kpi['Net Sales']:,.0f}; returns cost {kpi['Return Rate (value)']:.1%} of sales value "
      f"(order return rate {overall_rr:.1%}) -> returned orders are larger than average "
      f"(${ret.total_amount.mean():,.0f} vs ${df[df.is_returned == 0].total_amount.mean():,.0f}).")
print(f"* Top category by sales: {top(cat.sales).index[0]} (${top(cat.sales).iloc[0]:,.0f}); "
      f"highest return rate: {top(cat.return_rate).index[0]} ({top(cat.return_rate).iloc[0]:.1%}); "
      f"lowest: {top(cat.return_rate, asc=True).index[0]} ({top(cat.return_rate, asc=True).iloc[0]:.1%}).")
print(f"* Best profit % category: {top(cat.profit_pct).index[0]} ({top(cat.profit_pct).iloc[0]:.1%}); "
      f"worst: {top(cat.profit_pct, asc=True).index[0]} ({top(cat.profit_pct, asc=True).iloc[0]:.1%}).")
print(f"* Regions: top sales = {top(reg.sales).index[0]}; return rate ranges {reg.return_rate.min():.1%}-{reg.return_rate.max():.1%} (small spread).")
print(f"* Top 20% of customers generate {top20:.0%} of sales; {1 - one:.0%} of customers are repeat buyers.")
print(f"* {len(neg) / len(df):.1%} of orders are loss-making; {len(big)} extreme orders (> ${upper:,.0f}) account for "
      f"{big.total_amount.sum() / gross:.1%} of sales.")
top_reason = ret.return_reason.value_counts(normalize=True)
print(f"* Top return reason: {top_reason.index[0]} ({top_reason.iloc[0]:.0%}); 'Slow delivery' is just {top_reason.get('Slow delivery', 0):.1%}.")
print(f"* Median time from delivery to return request: {ret.return_lag_days.median():.0f} days.")

try:
    from scipy import stats
    ct = pd.crosstab(df.category, df.is_returned)
    chi2, p, *_ = stats.chi2_contingency(ct)
    print(f"* Category vs return: chi2 p-value = {p:.2e} (significant association)" if p < .05 else f"* Category vs return not significant (p={p:.3f})")
    for col in ["region", "payment_method", "customer_gender", "discount_band"]:
        _, p, *_ = stats.chi2_contingency(pd.crosstab(df[col], df.is_returned))
        print(f"  {col:<16} vs return: p = {p:.3f} -> {'significant' if p < .05 else 'no significant link'}")
    r, p = stats.pointbiserialr(df.is_returned, df.total_amount)
    print(f"  order value vs return: r = {r:.3f}, p = {p:.2e}")
except ImportError:
    print("(install scipy for the statistical significance tests)")

print(f"\nDone. All charts are in ./{OUT}/")

