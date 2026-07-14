# Raspberry Pi Trading Agent

This project runs a small Lumibot strategy against an Alpaca account. It checks
once per trading day whether **Asset B** has fallen by a configured percentage
from its recent high. If that dip occurs and the account owns **Asset A**, the
agent sells Asset A and uses the available cash to buy as many whole shares of
Asset B as possible.

Start with Alpaca paper trading. Paper trading uses simulated money and is the
appropriate place to learn how the agent behaves. Do not enable live trading
until you have watched it operate successfully for an extended period and fully
understand the risks.

> **Important:** This software does not guarantee a profit and is not financial
> advice. A price dip can continue after a purchase. Market orders can fill at a
> worse price than expected. Software, internet, power, market-data, and broker
> failures can all affect trading.

## What the strategy does

The default configuration uses:

- Asset A: `SPY`
- Asset B: `QQQ`
- Dip threshold: `5%`
- Recent-high window: `20` daily bars

For example, suppose QQQ's highest daily high during the selected window was
`$500`. A 5% dip level is `$475`:

```text
Dip percentage = (recent high - current price) / recent high × 100
Dip percentage = ($500 - $475) / $500 × 100 = 5%
```

If QQQ is at or below that level and the account owns SPY, the agent performs
this sequence:

1. Submit a market order to sell the entire long SPY position.
2. Wait for Alpaca to confirm that the sale filled.
3. Read the available cash and current QQQ price.
4. Submit a market order for the maximum whole number of QQQ shares that
   roughly 99% of the available cash can purchase. About 1% is held back so a
   small upward price move between the quote and the fill cannot cause the
   order to be rejected or overspend the account.
5. Leave the safety buffer and any cash insufficient for another whole share
   unused.

The rotation is tracked in a small state file, so a reboot, crash, or rejected
order between the sale and the purchase does not strand the cash: the agent
reconciles its state against actual positions and open orders on the next
evaluation and finishes the rotation.

The agent does **not** automatically create the initial Asset A position. It
also does not rotate from Asset B back into Asset A. After a completed A-to-B
rotation, it will take no further rotation action unless Asset A is held again.

## Project files

```text
pi-trading-agent/
├── README.md           This guide
├── config.example.json Placeholder template copied to config.json
├── config.json         Credentials, symbols, and strategy settings
├── requirements.txt    Python package requirements
├── main.py             Configuration validation and application startup
├── adaptive_news_model.py  Persistent learning from news and later returns
├── trade_memory.py      SQLite journal and learning from past rotation signals
├── news_context.py     Recent-news retrieval and transparent risk scoring
├── llm_news.py         Optional LLM daily news assessment (Gemini/Claude)
├── strategy.py         Daily dip and rotation logic
└── setup_service.sh    Virtual environment and systemd installer
```

The installer later creates `.venv/`, which contains an isolated Python
environment. Do not edit that directory. When email reporting is enabled, the
agent also creates `.last_email_report`. That small file contains only the date
of the most recently sent report and prevents duplicates after a restart. The
adaptive model creates `.news_learning_state.json` to preserve its observations.
Decision memory creates `.trade_memory.sqlite3`, a local database of market
snapshots, decisions, and fills (never credentials or balances).
`.rotation_state.json` remembers a rotation that is partway through (sold
Asset A, not yet bought Asset B) so restarts cannot strand the cash.

## What you need

Before beginning, you need:

- A Raspberry Pi 5 running a current 64-bit Raspberry Pi OS or Debian-based OS.
- A reliable internet connection.
- Correct system time and timezone synchronization.
- An Alpaca account with paper-trading API credentials.
- A user account that can run `sudo` commands.

The Pi should use reliable storage and power. An uninterruptible power supply
is worth considering for a machine that may place real orders.

## Step 1: Create Alpaca paper credentials

1. Sign in to the Alpaca dashboard.
2. Select the **paper trading** account, not the live account.
3. Create or regenerate its API key and secret.
4. Copy both values immediately. The secret may only be displayed once.
5. Never post these values in chat, email, screenshots, or source control.

Paper and live accounts use different credentials. A paper key cannot trade the
live account.

## Step 2: Prepare the Raspberry Pi

Update the package lists and install Python's virtual-environment support:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Confirm that Python is available:

```bash
python3 --version
```

## Step 3: Configure the agent

Move into the project directory:

```bash
cd /mnt/dietpi_userdata/staging/pi-trading/pi-trading-agent
```

If `config.json` does not exist yet (for example after cloning the repository),
create it from the template first:

```bash
cp config.example.json config.json
```

Open the configuration file with a terminal editor:

```bash
nano config.json
```

The initial file is:

```json
{
  "ALPACA_API_KEY": "REPLACE_WITH_YOUR_ALPACA_API_KEY",
  "ALPACA_SECRET_KEY": "REPLACE_WITH_YOUR_ALPACA_SECRET_KEY",
  "IS_PAPER_TRADING": true,
  "ASSET_A": "SPY",
  "ASSET_B": "QQQ",
  "DIP_THRESHOLD_PERCENT": 5.0,
  "RECENT_HIGH_LOOKBACK_DAYS": 20,
  "EMAIL_REPORT_ENABLED": false,
  "EMAIL_SMTP_HOST": "smtp.gmail.com",
  "EMAIL_SMTP_PORT": 587,
  "EMAIL_SMTP_USERNAME": "REPLACE_WITH_YOUR_EMAIL_ADDRESS",
  "EMAIL_SMTP_PASSWORD": "REPLACE_WITH_YOUR_EMAIL_APP_PASSWORD",
  "EMAIL_FROM_ADDRESS": "REPLACE_WITH_YOUR_EMAIL_ADDRESS",
  "EMAIL_TO_ADDRESS": "REPLACE_WITH_REPORT_RECIPIENT_EMAIL",
  "EMAIL_USE_TLS": true,
  "NEWS_CONTEXT_ENABLED": true,
  "NEWS_LOOKBACK_HOURS": 24,
  "NEWS_MAX_ARTICLES": 50,
  "NEWS_BLOCK_ON_HIGH_RISK": true,
  "NEWS_HIGH_RISK_SCORE": -6,
  "NEWS_LEARNING_ENABLED": true,
  "NEWS_LEARNING_BLOCK_ENABLED": true,
  "NEWS_LEARNING_MIN_OBSERVATIONS": 20,
  "NEWS_LEARNING_MAX_OBSERVATIONS": 120,
  "NEWS_LEARNING_MIN_CORRELATION": 0.15,
  "NEWS_PREDICTED_RETURN_BLOCK_PERCENT": -1.0,
  "DECISION_MEMORY_ENABLED": true,
  "DECISION_MEMORY_BLOCK_ENABLED": false,
  "DECISION_MEMORY_MIN_OBSERVATIONS": 40,
  "DECISION_MEMORY_MAX_OBSERVATIONS": 180,
  "DECISION_MEMORY_MIN_CORRELATION": 0.25,
  "DECISION_MEMORY_EDGE_BLOCK_PERCENT": -0.75,
  "DECISION_MEMORY_BACKFILL_DAYS": 1000,
  "LLM_NEWS_ENABLED": false,
  "LLM_NEWS_PROVIDER": "gemini",
  "LLM_NEWS_API_KEY": "REPLACE_WITH_YOUR_LLM_API_KEY",
  "LLM_NEWS_MODEL": "gemini-2.5-flash",
  "LLM_NEWS_BASE_URL": "",
  "LLM_NEWS_BLOCK_ON_HIGH_RISK": false,
  "LLM_NEWS_BLOCK_SCORE": -6
}
```

Replace only the two Alpaca credential strings. Keep the quotation marks. For example,
the key line should have the same shape as this fictitious value:

```json
"ALPACA_API_KEY": "PKEXAMPLE123456"
```

Do not copy that fictitious key. Use the paper key shown in your Alpaca
dashboard.

In Nano, save with `Ctrl+O`, press Enter, and exit with `Ctrl+X`.

Protect the file so only its owner can read and write it:

```bash
chmod 600 config.json
```

### Configuration reference

| Setting | Meaning | Valid example |
|---|---|---|
| `ALPACA_API_KEY` | Alpaca API key identifier | Your paper API key |
| `ALPACA_SECRET_KEY` | Alpaca API secret | Your paper API secret |
| `IS_PAPER_TRADING` | Selects simulated or real trading | `true` |
| `ASSET_A` | Entire position to sell when the signal occurs | `"SPY"` |
| `ASSET_B` | Asset whose dip is measured and purchased | `"QQQ"` |
| `DIP_THRESHOLD_PERCENT` | Required fall from the recent high | `5.0` |
| `RECENT_HIGH_LOOKBACK_DAYS` | Number of daily bars used for the high | `20` |
| `EMAIL_REPORT_ENABLED` | Turns the daily summary on or off | `false` |
| `EMAIL_SMTP_HOST` | Outgoing mail server | `"smtp.gmail.com"` |
| `EMAIL_SMTP_PORT` | Outgoing mail server port | `587` |
| `EMAIL_SMTP_USERNAME` | Login name, usually the sending email | `"name@example.com"` |
| `EMAIL_SMTP_PASSWORD` | Provider-issued app password | Your app password |
| `EMAIL_FROM_ADDRESS` | Address shown as the sender | `"name@example.com"` |
| `EMAIL_TO_ADDRESS` | Address that receives reports | `"recipient@example.com"` |
| `EMAIL_USE_TLS` | Enables STARTTLS encryption | `true` |
| `NEWS_CONTEXT_ENABLED` | Fetches and scores recent Alpaca news | `true` |
| `NEWS_LOOKBACK_HOURS` | Age window for news articles | `24` |
| `NEWS_MAX_ARTICLES` | Maximum articles checked daily | `50` |
| `NEWS_BLOCK_ON_HIGH_RISK` | Allows severe news context to block a rotation | `true` |
| `NEWS_HIGH_RISK_SCORE` | Score at or below which a trade is blocked | `-6` |
| `NEWS_LEARNING_ENABLED` | Learns news-score/next-return relationships | `true` |
| `NEWS_LEARNING_BLOCK_ENABLED` | Allows a mature learned forecast to block rotation | `true` |
| `NEWS_LEARNING_MIN_OBSERVATIONS` | Samples required before forecasts affect decisions | `20` |
| `NEWS_LEARNING_MAX_OBSERVATIONS` | Rolling history retained by the model | `120` |
| `NEWS_LEARNING_MIN_CORRELATION` | Minimum relationship strength for a learned veto | `0.15` |
| `NEWS_PREDICTED_RETURN_BLOCK_PERCENT` | Forecast at or below which rotation is blocked | `-1.0` |
| `DECISION_MEMORY_ENABLED` | Records dip decisions and their subsequent relative result | `true` |
| `DECISION_MEMORY_BLOCK_ENABLED` | Allows mature decision memory to veto a rotation | `false` |
| `DECISION_MEMORY_MIN_OBSERVATIONS` | Comparable dip signals needed before a forecast | `40` |
| `DECISION_MEMORY_MAX_OBSERVATIONS` | Rolling comparable-signal history retained | `180` |
| `DECISION_MEMORY_MIN_CORRELATION` | Fit strength required for a decision-memory veto | `0.25` |
| `DECISION_MEMORY_EDGE_BLOCK_PERCENT` | B-minus-A forecast at or below which rotation is blocked | `-0.75` |
| `DECISION_MEMORY_BACKFILL_DAYS` | Daily bars imported at startup to shorten decision-memory warm-up (`0` disables) | `1000` |
| `LLM_NEWS_ENABLED` | Sends the day's headlines to an LLM for one risk assessment | `false` |
| `LLM_NEWS_PROVIDER` | `gemini`, `openai_compatible`, or `anthropic` | `"gemini"` |
| `LLM_NEWS_API_KEY` | API key for the chosen provider | Your API key |
| `LLM_NEWS_MODEL` | Model used for the assessment | `"gemini-2.5-flash"` |
| `LLM_NEWS_BASE_URL` | Endpoint for `openai_compatible` providers only | `""` |
| `LLM_NEWS_BLOCK_ON_HIGH_RISK` | Allows the LLM assessment to block a rotation | `false` |
| `LLM_NEWS_BLOCK_SCORE` | LLM score at or below which a trade is blocked | `-6` |

JSON is strict:

- Boolean values are lowercase `true` and `false` and have no quotation marks.
- Text values require double quotation marks.
- Each line except the final setting needs a comma.
- Asset A and Asset B must be different symbols.
- The dip threshold must be greater than 0 and less than 100.
- The lookback must be an integer of at least 2.
- The SMTP port must be an integer from 1 through 65535.
- News lookback must be from 1 through 168 hours.
- News article count must be from 1 through 50.
- The high-risk score must be a negative integer.
- Learning minimum must be from 10 through 500 observations.
- Learning maximum must be at least the minimum and no more than 1,000.
- Minimum learned correlation must be from `0.0` through `1.0`.
- The learned-return blocking threshold must be negative and at least `-25`.
- The LLM block score must be an integer from `-10` through `-1`.
- The LLM provider must be `gemini`, `openai_compatible`, or `anthropic`.
- When the provider is `openai_compatible`, `LLM_NEWS_BASE_URL` is required.
- When `LLM_NEWS_ENABLED` is `true`, a real `LLM_NEWS_API_KEY` is required.

The lookback is a number of market-data bars, not necessarily calendar days.
Weekends and exchange holidays do not produce normal stock-market daily bars.

## Optional daily email report

Email reporting is initially disabled. Leaving it disabled does not affect
trading. When enabled, the agent attempts to send one summary after its daily
market evaluation. The report includes:

- Evaluation date and time.
- Asset A and Asset B symbols.
- Current prices and quantities, when available.
- Asset B's recent high and calculated dip, when available.
- The configured dip threshold.
- The action taken or the reason no action was taken.
- Any caught evaluation error.

For stock and ETF strategies, “once a day” means once on a day when Lumibot runs
the scheduled market evaluation. A report is not normally expected on weekends
or exchange holidays. Email delivery timing depends on the market schedule,
internet connection, and mail provider.

### Email setup using Gmail

Do not place your normal Google account password in `config.json`. Google
typically requires two-step verification and a separately generated app
password for SMTP clients. Account menus and eligibility can vary, so consult
Google's current account help if the App Passwords option is unavailable.

With a Gmail app password, settings usually have this form:

```json
"EMAIL_REPORT_ENABLED": true,
"EMAIL_SMTP_HOST": "smtp.gmail.com",
"EMAIL_SMTP_PORT": 587,
"EMAIL_SMTP_USERNAME": "your.address@gmail.com",
"EMAIL_SMTP_PASSWORD": "YOUR_GOOGLE_APP_PASSWORD",
"EMAIL_FROM_ADDRESS": "your.address@gmail.com",
"EMAIL_TO_ADDRESS": "where.to.send@example.com",
"EMAIL_USE_TLS": true
```

An app password is still a secret. Enter the actual provider-generated value,
not the example above, and keep `config.json` protected with mode `600`.

### Using another email provider

Find the provider's authenticated SMTP settings. This implementation supports
ordinary SMTP upgraded with STARTTLS, most commonly on port 587. Enter its SMTP
hostname, port, username, password or app password, and sender address. Do not
assume Gmail's settings work for a different provider.

If the provider requires implicit SSL from the beginning of the connection,
commonly associated with port 465, the current implementation is not configured
for that mode. Use the provider's STARTTLS endpoint instead.

### Enable email safely

Stop the service, edit the settings, and restart:

```bash
sudo systemctl stop trading-agent.service
nano config.json
chmod 600 config.json
sudo systemctl start trading-agent.service
sudo journalctl -u trading-agent.service -f
```

Look for `Daily email report sent`. The agent does not send a test message at
startup; it sends after the next scheduled daily evaluation. Check the spam or
junk folder if the log reports success but the message is not in the inbox.

The `.last_email_report` file is updated only after SMTP reports successful
delivery. If sending fails, the error is logged and trading continues. The agent
can try again on another iteration. It will not intentionally send another
summary for a date already stored in that file.

To disable reports, set this value and restart the service:

```json
"EMAIL_REPORT_ENABLED": false
```

## World-event and news awareness

The agent has a basic news-awareness layer. Before evaluating a rotation, it
asks Alpaca for recent general market news. It examines each headline and
summary for explicit risk and opportunity phrases, calculates an aggregate
score, and records the most influential headlines in the system log and daily
email.

This is **not machine learning**, artificial general intelligence, or an
understanding of geopolitics. It is a small, auditable rule system. It does not
learn facts permanently, predict the truth of a story, read articles behind
links, or understand irony and subtle context. Its useful property is that a
novice can inspect exactly which words affected a decision.

### How scoring works

The rules are in `news_context.py`:

- Severe risk phrases such as `invasion`, `bank failure`, and `terrorist attack`
  contribute `-3` each.
- Ordinary risk phrases such as `recession`, `sanctions`, `rate hike`, and
  `supply disruption` contribute `-1` each.
- Constructive phrases such as `ceasefire`, `rate cut`, `stimulus`, and `trade
  agreement` contribute `+1` each.

Each phrase is counted no more than once per article. Scores from all retrieved
articles are added together. With the default `NEWS_HIGH_RISK_SCORE` of `-6`, a
score of `-6`, `-7`, or lower is classified as high risk.

The daily log can look like:

```text
World-event context: risk=high, score=-7, articles=50.
News evidence: [-3] Example headline text (matched: invasion)
Dip signal met, but rotation was blocked by the configured world-event risk guard.
```

Headlines in that example are illustrative. Actual output comes from Alpaca.

### How news changes a trade decision

News does not create a buy signal by itself. The normal price-dip test still has
to pass, and the account must still own Asset A.

When all normal trading conditions pass:

1. If the news score is above the high-risk threshold, rotation proceeds.
2. If the score is at or below the threshold and
   `NEWS_BLOCK_ON_HIGH_RISK` is `true`, the agent does not sell Asset A that day.
3. The blocked decision, score, and matching headlines are logged and included
   in the email report.
4. The agent evaluates conditions again on its next scheduled iteration.

If Alpaca news or the internet is unavailable, news status becomes
`unavailable`. The error is logged and the original price strategy continues.
This is called **fail-open** behavior. It prevents a news-provider outage from
silently shutting down the price strategy, but it also means news protection is
absent during that outage.

### Advisory-only mode

To see news context without allowing it to block trades, use:

```json
"NEWS_CONTEXT_ENABLED": true,
"NEWS_BLOCK_ON_HIGH_RISK": false
```

This is a sensible first paper-trading mode. Headlines, scores, and risk levels
still appear in logs and emails, but the score cannot veto a rotation.

To disable news fetching entirely:

```json
"NEWS_CONTEXT_ENABLED": false
```

Restart the service after changing either setting.

### Important news limitations

- Alpaca news is a financial-news feed, not a complete record of every event in
  every country.
- Breaking events may appear late, be absent, or change after publication.
- Multiple articles about the same event can make its score larger.
- Keyword matching can misread context or classify an irrelevant story.
- A constructive keyword does not mean an asset will rise.
- A negative score does not mean an asset will fall.
- The configured threshold has not been proven profitable merely because it is
  included in the software.

Use paper mode to compare the logged scores with actual headlines and market
behavior before relying on the guard.

## Optional LLM news assessment

In addition to the fixed keyword rules, the agent can send the same daily
Alpaca headlines to a language model for one risk assessment per trading day.
Unlike the keyword scorer, the model reads the articles with genuine language
understanding: it can recognize that ten articles describe one event, that a
headline is speculation rather than fact, or that a negative-sounding story
is irrelevant to broad US equity markets.

Like every other news feature, this layer can only **veto** a rotation. It
never creates a buy signal, and if the API is unreachable the price strategy
continues without it (the same fail-open behavior as the rest of the news
stack).

### Getting a free API key (Gemini, the default)

The default provider is Google's Gemini API, which has a genuinely free,
rate-limited tier that comfortably covers this agent's one request per
trading day. Chat subscriptions such as Claude Pro or ChatGPT Plus **cannot**
be used here; those products do not include API access.

1. Sign in at `aistudio.google.com` with a Google account.
2. Create an API key (no credit card is required for the free tier).
3. Paste it into `LLM_NEWS_API_KEY` in `config.json`.

One caveat of the free tier: Google may use free-tier prompts to improve its
products. This agent only ever sends **public news headlines and summaries**
to the model — never your credentials, positions, balances, or account data —
so there is nothing sensitive in those prompts.

Treat the API key like every other secret in `config.json`: never commit it,
keep the file at mode `600`, and rotate the key if exposure is suspected.

### Enabling it

Stop the service, edit the settings, and restart:

```json
"LLM_NEWS_ENABLED": true,
"LLM_NEWS_PROVIDER": "gemini",
"LLM_NEWS_API_KEY": "YOUR_GEMINI_API_KEY",
"LLM_NEWS_MODEL": "gemini-2.5-flash",
"LLM_NEWS_BASE_URL": "",
"LLM_NEWS_BLOCK_ON_HIGH_RISK": false,
"LLM_NEWS_BLOCK_SCORE": -6
```

With `LLM_NEWS_BLOCK_ON_HIGH_RISK` set to `false` (the default), the
assessment is **advisory only**: the score, risk level, and reasoning appear
in the logs and the daily email, but cannot block a trade. This is the
recommended starting mode. Review several weeks of paper-trading logs and
compare the model's assessments with what actually happened before setting it
to `true`.

### Other providers

| Provider setting | What it is | Key settings |
|---|---|---|
| `"gemini"` | Google Gemini API (free tier available) | Model such as `"gemini-2.5-flash"`; leave `LLM_NEWS_BASE_URL` empty |
| `"anthropic"` | Claude API (paid, a few cents per month here) | Key from `platform.claude.com`; model such as `"claude-opus-4-8"` or the cheaper `"claude-haiku-4-5"` |
| `"openai_compatible"` | Any OpenAI-compatible endpoint (Groq, OpenRouter, a local server, ...) | Set `LLM_NEWS_BASE_URL`, e.g. `"https://api.groq.com/openai/v1"`, plus that provider's key and model name |

Free tiers are rate-limited and their terms can change; if a provider starts
failing, the agent logs the error and continues on price logic and the
keyword guard alone.

### How it works

1. The existing news layer fetches up to `NEWS_MAX_ARTICLES` recent articles
   from Alpaca (so `NEWS_CONTEXT_ENABLED` must remain `true`).
2. The headlines and summaries are sent to the configured model with
   instructions to score aggregate near-term market risk from `-10` (severe,
   market-wide danger) to `+10` (strongly constructive), scoring
   conservatively and treating duplicate coverage as one event.
3. The model must reply in a fixed JSON format containing the score, a risk
   level, and two or three sentences of reasoning that cite the headlines.
4. The score, level, and reasoning are logged and included in the email
   report. When blocking is enabled and all normal trading conditions pass, a
   score at or below `LLM_NEWS_BLOCK_SCORE` vetoes that day's rotation.

The log line looks like:

```text
LLM news assessment: risk=elevated, score=-3. Several articles describe ...
```

### Limitations

- The model sees only headlines and short summaries, not full articles.
- A language model can misjudge significance in either direction; its
  reasoning is plausible-sounding even when wrong.
- The assessment depends on an external API: an outage means no LLM
  protection that day (trading continues on price logic and the keyword
  guard).
- Scores are not comparable to the keyword score; the two guards use separate
  thresholds and either can veto independently once enabled.

## Adaptive news learning

In addition to fixed headline rules, the agent now learns a simple relationship
from its own daily observations. This is genuine statistical adaptation, but it
is intentionally much smaller and more explainable than an AI language model.

Each daily evaluation records:

1. The date.
2. That day's aggregate news score.
3. Asset B's current price.

At the next evaluation, it calculates Asset B's percentage return since the
previous observation and pairs that return with the previous news score. After
the configured minimum number of completed observations, it fits a stabilized
linear regression:

```text
predicted next return = baseline return + learned sensitivity × current news score
```

The model reports its observation count, predicted next-session return,
news-score sensitivity, and historical correlation. These values appear in the
system log and email summary.

### Warm-up period

The default minimum is 20 completed observations. Until then, logs say something
like:

```text
Learning safely: 7/20 required completed observations collected.
```

During warm-up, the adaptive forecast cannot block a trade. At roughly one
observation per trading day, 20 observations usually require about four market
weeks. Restarts do not erase progress because observations are stored in
`.news_learning_state.json`.

### Learned risk veto

Once mature, the adaptive model can block a rotation when its forecast is at or
below `NEWS_PREDICTED_RETURN_BLOCK_PERCENT`. With the default `-1.0`, a predicted
next-session Asset B return of `-1.00%` or lower blocks that day's rotation. The
absolute historical correlation must also meet `NEWS_LEARNING_MIN_CORRELATION`;
the default requires at least `0.15`. If news scores do not vary enough, the
model remains non-authoritative even after collecting the minimum sample count.

This veto is separate from the fixed high-risk keyword veto. Either can block a
trade. To collect learning data without allowing learned forecasts to affect
orders, temporarily use:

```json
"NEWS_LEARNING_ENABLED": true,
"NEWS_LEARNING_BLOCK_ENABLED": false
```

To stop collection as well, set `NEWS_LEARNING_ENABLED` to `false`. Existing
state is preserved and becomes available again if learning is re-enabled.

### Rolling and bounded behavior

- Only the newest configured number of completed observations is retained.
- The default rolling maximum is 120 observations.
- Single-session returns stored for training are capped to the range `-25%` to
  `+25%` to reduce the influence of corrupt prices or extreme outliers.
- Forecasts are capped to `-10%` through `+10%`.
- A small ridge stabilizer prevents huge coefficients when news scores barely
  change.
- The model uses only information available at each daily observation; it does
  not use future prices when producing that day's forecast.

### Learning limitations

- Twenty observations is only a safety minimum, not proof of accuracy.
- Correlation does not establish that headlines caused later returns.
- Market relationships change, so previously learned behavior can stop working.
- One daily Asset B price cannot measure intraday reactions or execution prices.
- The model has only one predictor: the aggregate news score.
- A statistically produced forecast can still be completely wrong.

Keep the agent in paper mode throughout warm-up and review many mature forecasts
before considering whether this feature is useful.

## Decision memory: learning from its own rotations

The news learner estimates Asset B's absolute return. Decision memory adds the
question this strategy needs to answer: after a comparable dip, would owning
Asset B have done better than continuing to own Asset A?

For each evaluable day, its local SQLite database records the two prices, dip,
available news score, signal state, and final decision. At the next market
evaluation it settles the earlier record using `Asset B return - Asset A
return`. Only prior dip signals train its conservative, ridge-stabilized model;
the inputs are dip size and news score. Broker-confirmed fills are recorded
separately for auditability.

It is advisory by default. After at least 40 comparable settled signals, you
may enable `DECISION_MEMORY_BLOCK_ENABLED` in paper trading after reviewing its
forecasts. A veto requires both a negative predicted edge and the configured
minimum fit correlation. It never creates a trade, increases order size, or
overrides the existing safeguards. Delete `.trade_memory.sqlite3` only if you
intend to reset this learning history.

### Startup catch-up

On its first valid evaluation after a start, the agent imports up to
`DECISION_MEMORY_BACKFILL_DAYS` of daily price bars (default: 1,000). It
calculates each historical dip using the configured lookback and stores only
next-session outcomes that were already complete. Existing dates are never
overwritten, so restarts are safe; a transient data failure is retried on the
next daily evaluation. Set the value to `0` to disable this request.

The adaptive news learner is not backfilled: it needs the actual news score
known on each historical date. Reusing today's score for earlier prices would
produce misleading training data. Its daily warm-up therefore remains intact.

### Resetting learned history

Normally, do not edit the learning-state file. To deliberately start learning
from zero, stop the service, preserve a backup, and restart:

```bash
sudo systemctl stop trading-agent.service
cp .news_learning_state.json .news_learning_state.json.backup
rm .news_learning_state.json
sudo systemctl start trading-agent.service
```

If the state file becomes invalid JSON, the model moves it to a `.corrupt` file
and starts clean rather than trusting damaged observations.

## Step 4: Install and start the service

Make sure the installer is executable, then run it with administrator rights:

```bash
chmod +x setup_service.sh
sudo ./setup_service.sh
```

The installer:

1. Creates `.venv` in the project directory.
2. Installs the packages from `requirements.txt` into that environment.
3. restricts `config.json` to owner-only access.
4. Creates `/etc/systemd/system/trading-agent.service`.
5. Enables the service at boot.
6. Starts the service immediately.

Package installation can take several minutes on a Raspberry Pi. Do not turn
off the Pi while it is running.

## Step 5: Verify operation

Check the service state:

```bash
sudo systemctl status trading-agent.service
```

Look for `active (running)`. Press `q` to leave the status screen.

Follow new log messages in real time:

```bash
sudo journalctl -u trading-agent.service -f
```

Press `Ctrl+C` to stop watching the logs. This only exits the log viewer; it
does not stop the trading agent.

Show the most recent 100 log lines without following them:

```bash
sudo journalctl -u trading-agent.service -n 100 --no-pager
```

Also sign in to the Alpaca paper dashboard and confirm that you are viewing the
paper account. Review its positions, orders, buying power, and activity.

## Understanding common log messages

`Starting paper trading for SPY/QQQ`

: The configuration loaded successfully and the broker lifecycle started.

`SPY=$... (... shares), QQQ=$... (... shares), ...-day high=$..., dip=...%`

: The daily evaluation succeeded. This message reports both prices and
positions, the recent high, and the calculated dip.

`Dip signal triggered. Submitted market sale ...`

: Asset B met the threshold and the agent submitted the sale of Asset A.

`Filled sell order ...`

: Alpaca confirmed the Asset A sale.

`Submitted market buy ...`

: The agent submitted the maximum whole-share Asset B purchase supported by
the available cash.

`Price data unavailable ... retrying next cycle`

: The broker or network did not provide usable data. No decision was made.

`Trading iteration failed safely ...`

: An unexpected broker, data, or network error was caught. The strategy process
remains alive and will evaluate again on its schedule.

`Daily email report sent for ...`

: The SMTP server accepted the daily summary and the date was recorded locally.

`Daily email report failed safely ...`

: Authentication, networking, TLS, or SMTP delivery failed. Trading continues;
inspect the rest of the message for the cause.

`World-event context: risk=..., score=..., articles=...`

: News retrieval succeeded. The message summarizes the deterministic score.

`News evidence: ...`

: A headline matched one or more configured phrases. The number in brackets is
that article's score and the matching terms are shown afterward.

`News context unavailable; price strategy will continue ...`

: The news request failed. No news veto is applied, but the price strategy and
daily email continue where possible.

## Service commands

Stop the agent before editing its configuration:

```bash
sudo systemctl stop trading-agent.service
```

Start it:

```bash
sudo systemctl start trading-agent.service
```

Restart it after editing code or configuration:

```bash
sudo systemctl restart trading-agent.service
```

Check whether it is running:

```bash
sudo systemctl is-active trading-agent.service
```

Check whether it will start after reboot:

```bash
sudo systemctl is-enabled trading-agent.service
```

The service waits 30 seconds before restarting after a crash. Repeated restarts
usually indicate invalid configuration, bad credentials, unavailable packages,
or a startup error. Inspect the journal rather than repeatedly rerunning the
installer.

## Changing strategy settings

Always stop the service before changing `config.json`:

```bash
sudo systemctl stop trading-agent.service
nano config.json
chmod 600 config.json
sudo systemctl start trading-agent.service
sudo journalctl -u trading-agent.service -n 50 --no-pager
```

Changing the threshold or lookback changes the signal materially. A smaller dip
threshold generally causes a signal sooner; a larger threshold requires a
deeper fall. A longer lookback may preserve an older, higher reference price.
These observations are mechanical descriptions, not claims about profitability.

Verify that every configured symbol is supported and tradable at Alpaca. This
strategy is designed for ordinary whole-share stock or ETF symbols. It is not
configured for options, short positions, leveraged borrowing logic, or crypto
pairs.

## Paper-trading test checklist

Before even considering live trading, verify all of the following:

- The logs explicitly say `Starting paper trading`.
- The Alpaca dashboard is visibly in paper mode.
- Asset A and Asset B are the intended symbols.
- The paper account has the intended Asset A position.
- Daily prices and holdings appear correctly in logs.
- A signal creates one Asset A sell and one Asset B buy, not duplicates.
- Broker order quantities match the log messages.
- The service remains active across several market days.
- The service starts normally after a controlled Pi reboot.
- Disconnecting and reconnecting the network does not produce unintended
  duplicate orders.
- You understand how to stop the service immediately.

Paper execution does not reproduce every live-market condition. It may differ
from live trading in fills, slippage, liquidity, and delays.

## Live-trading warning

Setting `IS_PAPER_TRADING` to `false` can submit real-money orders when used with
live credentials. Do not make that change merely to test whether the service
works. Use paper mode for testing.

If you eventually choose live mode:

1. Stop the systemd service.
2. Confirm there are no open or pending orders in either Alpaca account.
3. Back up the existing configuration securely.
4. Enter the live account credentials.
5. Change `IS_PAPER_TRADING` to `false`.
6. Recheck the symbols, positions, threshold, and account balance.
7. Start the service while actively watching both the logs and broker dashboard.

The software cannot determine whether a symbol is suitable for you or whether
the configured trade size matches your risk tolerance.

## Troubleshooting

### The service repeatedly restarts

Read its recent logs:

```bash
sudo journalctl -u trading-agent.service -n 100 --no-pager
```

Common causes include malformed JSON, credential placeholders that were never
replaced, incorrect keys, or no network connection.

### `Replace the Alpaca credential placeholders`

Open `config.json` and replace both `REPLACE_WITH_...` values with actual Alpaca
paper credentials.

### JSON parsing error

Check quotation marks and commas. You can validate the file without printing
its secret contents:

```bash
python3 -m json.tool config.json >/dev/null && echo "JSON syntax is valid"
```

### Authentication or unauthorized error

Confirm that the API key and secret belong to the same Alpaca account and that
paper credentials are paired with `IS_PAPER_TRADING: true`. If a secret was
copied incorrectly, regenerate the key pair in Alpaca and update both values.

### No trades occur

This can be correct. Check all of these conditions:

- The market is in an appropriate trading session.
- Asset B's calculated dip meets or exceeds the configured threshold.
- The account owns a positive long quantity of Asset A.
- There is no unresolved order already pending at the broker.
- The symbols have current and historical data.

The agent does not buy Asset A for you.

### Asset A sold but Asset B was not bought

Review the logs and Alpaca order activity first. A connection failure or missing
price immediately after the sale can delay the purchase until the next strategy
cycle. Insufficient cash for one whole Asset B share also prevents a purchase.
The pending rotation is stored in `.rotation_state.json` and retried
automatically on the next evaluation, including after a rejected order or a
reboot. Do not manually create a duplicate order until you have checked
Alpaca's open, filled, rejected, and canceled orders.

### Dependency installation fails

Confirm internet access and free disk space:

```bash
df -h
ping -c 3 pypi.org
```

Then rerun the installer. It safely reuses the project virtual environment:

```bash
sudo ./setup_service.sh
```

### Daily email does not arrive

First inspect the journal:

```bash
sudo journalctl -u trading-agent.service -n 100 --no-pager
```

Then check:

- `EMAIL_REPORT_ENABLED` is the unquoted Boolean `true`.
- All placeholder email values were replaced.
- The SMTP hostname and port match the mail provider's documentation.
- `EMAIL_USE_TLS` matches a STARTTLS endpoint.
- The username is usually the complete email address.
- An app password is used when the provider requires one.
- The sending account has not blocked or challenged the login.
- The recipient's spam or junk folder does not contain the message.
- The Pi can reach the internet and its clock is correct.

If `.last_email_report` already contains today's date, the report was accepted
previously and duplicates are being suppressed. Display the date with:

```bash
cat .last_email_report
```

Do not delete or alter that file merely to force repeated production emails.

### News is always unavailable

Inspect the full error in the service journal. Confirm the Pi has internet
access and the installed `alpaca-py` package is current within the project's
virtual environment:

```bash
sudo journalctl -u trading-agent.service -n 100 --no-pager
.venv/bin/python -m pip show alpaca-py
```

The account's market-data entitlement or the news service may also affect
availability. Trading continues using price logic while news is unavailable.

### News blocks too many paper trades

Do not tune settings based on one surprising headline. Review several weeks of
paper logs first. For an immediate non-trading-impact test, switch to advisory
mode by setting `NEWS_BLOCK_ON_HIGH_RISK` to `false` and restart the service.
Avoid changing the term lists unless you understand Python and can test the
consequences.

### The Pi rebooted or lost power

After network connectivity returns, systemd should start the service
automatically. Verify the service and inspect logs from the current boot:

```bash
sudo systemctl status trading-agent.service
sudo journalctl -u trading-agent.service -b --no-pager
```

Always inspect the Alpaca dashboard for orders or partially completed activity
after an unexpected interruption.

## Security guidance

- Keep `config.json` at permission mode `600`.
- Do not commit credentials to Git.
- Do not paste journal output publicly without checking it for private data.
- Treat the SMTP password as carefully as the Alpaca API secret.
- Use unique Alpaca credentials and rotate them if exposure is suspected.
- Keep Raspberry Pi OS and Python security updates current.
- Restrict SSH access, use SSH keys, and disable password login when practical.
- Do not expose the Pi directly to the public internet.
- Do not run the Python process itself as root. The installer configures the
  service to use the non-root user who invoked `sudo`.

Check the credential file permissions with:

```bash
ls -l config.json
```

The beginning should resemble `-rw-------`.

## Updating Python dependencies

Dependency updates can change behavior. Stop the service and preserve the
working environment before updating intentionally:

```bash
sudo systemctl stop trading-agent.service
sudo ./setup_service.sh
sudo systemctl status trading-agent.service
```

Review the logs and repeat paper-mode validation after every update.

## Removing the service

This stops and removes the systemd unit but leaves the project and configuration
files in place:

```bash
sudo systemctl disable --now trading-agent.service
sudo rm /etc/systemd/system/trading-agent.service
sudo systemctl daemon-reload
sudo systemctl reset-failed
```

The project directory remains available for inspection or later reinstallation.
Delete it only after securely handling the API credentials in `config.json`.

## Operational routine

Even an automated service needs supervision. A sensible routine is to:

- Check service health and Alpaca activity each trading day.
- Review all filled, rejected, and canceled orders.
- Compare news scores with the underlying headlines and note false matches.
- Check disk usage and system updates regularly.
- Confirm system time remains synchronized.
- Test reboot recovery periodically while still in paper mode.
- Keep a written record of configuration changes.
- Revoke Alpaca credentials immediately if the Pi is lost or compromised.

Automation makes execution repeatable; it does not remove market or operational
risk.
