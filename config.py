"""
Central configuration for the alpha engine.

Every tunable lives here so that the screener, the live intraday engine and the
backtest all read the SAME numbers. If a threshold appears in two modules with
two different values, the backtest stops describing the live system — which is
the single most common way a backtest ends up lying to you.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CACHE = BASE / ".cache"
LOGS = BASE / "logs"
STATE = BASE / ".state"
for _p in (DATA, CACHE, LOGS, STATE):
    _p.mkdir(parents=True, exist_ok=True)

WATCHLIST_FILE = DATA / "watchlist.json"

# ---------------------------------------------------------------- account & risk
# FALLBACK ONLY. intraday.py fetches the broker's real account EQUITY (not
# buying_power — that includes margin leverage, e.g. this Alpaca paper account
# shows ~$396k buying_power against ~$99k actual equity, and sizing risk off
# leveraged buying power would over-risk every position ~4x) via
# broker().get_equity() on every run, live. This value is used only if that
# call fails (network issue, broker down) — set close to the real account so
# a fallback run still sizes sanely, not to $200 (that produced fractional,
# broker-rejected share counts on anything over ~$50 — see git history/
# session notes if this regresses). The backtest overrides this with
# BACKTEST_CAPITAL regardless, so it is never affected by this value.
ACCOUNT_SIZE = 100_000.0

RISK_PCT = 0.010          # fraction of equity risked on a single trade
MAX_POSITION_PCT = 0.30   # hard cap: one name may not exceed 30% of equity
# Raised from 2 for swing. Holding multi-day means slots stay occupied for days
# rather than hours, so a cap of 2 would block most signals and let arrival
# order, not quality, pick the book. 4 active position slots (locked config).
# Owned by execution_engine.py/intraday.py (the older, currently-dormant
# engine) ONLY — hybrid_engine.py has its OWN HYBRID_MAX_CONCURRENT below,
# deliberately not sharing this constant, per the isolation rule the two
# engines have kept since hybrid_engine.py was first built.
MAX_CONCURRENT = 4
MIN_POSITION_USD = 5.00   # below this the trade is not worth the friction
ALLOW_FRACTIONAL = True   # Robinhood supports fractional shares

# hybrid_engine.py's own concurrency cap — the Widened Spread Regime Filter
# deployment (see hybrid_indicators.py's REGIME_* constants).
HYBRID_MAX_CONCURRENT = 6

# ---------------------------------------------------------------- mode
# SWING: positions are held across sessions until the stop or the target is hit.
# There is no end-of-day flatten. This changes the risk model, not just the
# holding period — an overnight gap can jump straight through a stop, so a fill
# is never guaranteed at the stop price.
SWING_MODE = True
EOD_FLAT = False          # day-trading legacy; True restores the 15:55 flatten
MAX_HOLD_DAYS = None      # None = hold until stop/target, as specified

# ---------------------------------------------------------------- entry
ENTRY_MODE = "oversold"   # "oversold" | "pullback" | "breakout"
PULLBACK_SUPPORT = "vwap"          # "vwap" | "ema20" | "sma20d"
PULLBACK_MAX_EXTENSION = 0.004     # how far above support still counts as a dip
PULLBACK_DIP_LOOKBACK = 6          # bars: the dip must be recent (6 x 5min = 30min)
PULLBACK_DIP_TOLERANCE = 0.0015    # "touched" support within this fraction
REQUIRE_STRONG_TREND = True        # demand close > SMA50 as well as SMA200

# ---------------------------------------------------------------- exits
# Exits drive most of the P&L, so treat these as a deliberate choice to be
# reviewed, not as received wisdom.
STOP_METHOD = "pct"        # "structure" | "pct" | "atr"
STOP_PCT = 0.025           # 2.5% fixed stop (locked config) — used when STOP_METHOD == "pct"
STOP_ATR_MULT = 1.5        # used when STOP_METHOD == "atr"
STOP_BUFFER = 0.0015       # structural stop sits this far under the dip low
MAX_STOP_PCT = 0.06        # never risk more than this per share
TARGET_R = 2.5             # profit target as a multiple of stop distance (2.5R runner)
FLAT_HHMM = (15, 55)       # only applied when EOD_FLAT is True

# ---------------------------------------------------------------- execution range
# A breakout entry decays fast as price extends past the trigger. The "ideal
# execution range" is the band just above the trigger where the setup is still
# an entry rather than a chase.
MAX_EXTENSION_PCT = 0.0075   # top of the ideal band, above the trigger
APPROACH_PCT = 0.0050        # below the trigger: "approaching", worth watching

# ---------------------------------------------------------------- cadence
SCAN_INTERVAL_MIN = 15       # engine runs every 15 minutes
WINDOW_START = (10, 0)       # first scan, ET
WINDOW_END = (14, 0)         # last scan, ET

# ---------------------------------------------------------------- screener gates
WATCHLIST_SIZE = 30
MAX_PER_SECTOR = 5           # sector isolation
MAX_PER_INDUSTRY = 1         # "no two companies with identical business models"

MIN_MARKET_CAP = 2.0e9
MIN_AVG_DOLLAR_VOL = 50e6    # intraday tradability
MIN_PRICE = 10.0
MAX_PRICE = 1200.0

MAX_DEBT_TO_EQUITY = 2.50    # expressed as a ratio (yfinance reports percent)
MIN_CURRENT_RATIO = 1.00
MAX_FORWARD_PE = 60.0
MIN_FORWARD_PE = 0.0
REQUIRE_POSITIVE_FCF = True
MIN_REVENUE_GROWTH = 0.00
MIN_PROFIT_MARGIN = 0.00

EARNINGS_BLACKOUT_DAYS = 5   # skip names reporting within N calendar days

# ---------------------------------------------------------------- DCF
DCF_WACC = 0.090
DCF_TERMINAL_GROWTH = 0.025
DCF_HORIZON_YEARS = 5
DCF_FADE = 0.80              # growth decays 20%/yr toward terminal
DCF_MAX_INITIAL_GROWTH = 0.25

# ---------------------------------------------------------------- macro regime
# NOTE: `^VIX` returns no data from this host (verified — not assumed; it fails
# via both yf.download and Ticker.history). `^VIX9D` resolves but serves a
# single row, so it cannot support a trend. The regime therefore gates on SPY
# REALIZED volatility, which is always computable from bars we already hold.
# VIX is read opportunistically and used only if it resolves.
MACRO_SYMBOLS = {"spy": "SPY", "qqq": "QQQ", "tnx": "^TNX"}
VIX_PROXIES = ["^VIX9D", "VIXY"]      # best-effort, may all fail
REALIZED_VOL_WINDOW = 20              # trading days
REALIZED_VOL_RISK_OFF = 0.22          # 22% annualised SPY vol = hostile tape
VIX_RISK_OFF = 28.0                   # only applied when a real VIX level resolves
MACRO_SIZE_HAIRCUT = 0.50             # multiplier applied to size in a risk-off regime

# ---------------------------------------------------------------- costs
SLIPPAGE_BPS = 5.0           # per side, basis points
COMMISSION_PER_TRADE = 0.0   # Robinhood

# ---------------------------------------------------------------- backtest
BACKTEST_DAYS = 59
BACKTEST_CAPITAL = 10_000.0
BACKTEST_REQUIRE_EXEC_RANGE = True   # honour the ideal-execution-range gate

# ---------------------------------------------------------------- notifications
DISCORD_KEYCHAIN_SERVICE = "discord-alpha-webhook"
DISCORD_USERNAME = "Alpha Engine"
DISCORD_COLOR_EXECUTE = 0x2ECC71
DISCORD_COLOR_APPROACH = 0xF1C40F
DISCORD_COLOR_EXTENDED = 0xE67E22
DISCORD_COLOR_INFO = 0x3498DB

TZ = "America/New_York"

# ---------------------------------------------------------------- oversold entry
# Deep-oversold dip: buy weakness inside an intact long-term uptrend.
RSI_PERIOD = 14
RSI_OVERSOLD = 35.0
BB_PERIOD = 20
BB_K = 2.0
OVERSOLD_REQUIRE_ABOVE_SMA200 = True   # live price above the 200 SMA, not just prev close
OVERSOLD_STOP_LOOKBACK = 12            # bars used for the structural stop (1 hour)

# ---------------------------------------------------------------- partial scaling
# Sell a fraction at SCALE_R, then move the stop on the remainder to breakeven
# and let it run to TARGET_R.
#
# READ THIS BEFORE BELIEVING THE WIN RATE. Scaling raises win rate almost
# mechanically: any trade that reaches the scale level banks a profit, so even
# when the runner is stopped at breakeven the trade closes positive and counts
# as a win. It also caps every winner. Win rate and P&L therefore move in
# OPPOSITE directions here, and win rate stops being an informative metric —
# judge this by expectancy, profit factor and net P&L.
SCALE_ENABLED = True
SCALE_R = 1.0            # sell at this R multiple (locked config: +1.0R)
SCALE_FRACTION = 0.50     # how much to sell
BREAKEVEN_AFTER_SCALE = True

# ---------------------------------------------------------------- earnings
# Multi-day holds can straddle an earnings print, which is the fattest gap risk
# in the book. Dates are public well in advance, so using them is not lookahead.
EARNINGS_BLOCK_DAYS = 5   # do not ENTER if a report lands within N days
EARNINGS_EXIT = True      # close out before a report rather than hold through it

# ---------------------------------------------------------------- runner exit
# What the remaining 50% does after the first half is sold at SCALE_R.
#   "fixed"  — take profit at TARGET_R (capped upside)
#   "ema20h" — trail the last completed 1-hour 20 EMA (uncapped)
#   "atr"    — chandelier trail: high-water minus TRAIL_ATR_MULT x ATR (uncapped)
#   "pct"    — fixed percentage trail: high-water x (1 - TRAIL_PCT) (uncapped)
RUNNER_EXIT = "fixed"
TRAIL_ATR_MULT = 1.5
TRAIL_EMA_SPAN = 20
TRAIL_PCT = 0.04         # used when RUNNER_EXIT == "pct"
