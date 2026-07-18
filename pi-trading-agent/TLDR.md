# What this thing actually does (plain-English version)

This is the short version, with no jargon, for anyone who doesn't want to
read the full [README](README.md) or the technical
[decision pipeline doc](../.claude/docs/decision-pipeline.md). If you just
want to understand what the robot is doing and why, start here.

## The one-sentence version

It's a very cautious, very patient robot shopper: it watches a short list of
stocks/ETFs, buys one only when it looks "on sale" and history says that
usually pays off, sells for a small profit or to cut a loss, and checks the
news first so it doesn't buy into a crisis.

## How often does it actually do anything?

Twice a trading day: once when the market opens, and once again a few hours
later. The rest of the time it's just watching — it does not trade every
few minutes, and it does not trade every day if nothing looks good.

## What it looks for, step by step

1. **Check the news first.** Before doing anything, it checks recent
   headlines and (optionally) asks a small AI model whether things look
   calm or scary out there. If the news looks bad enough, it will refuse to
   open any *new* position that round — better to sit still than buy into
   a falling market.
2. **Look for a "sale."** For every stock/ETF on its watchlist, it checks:
   has this dropped a noticeable amount from its recent high? That's the
   "dip."
3. **Check its own history.** It doesn't just react to any dip — it looks
   back at what happened after similar dips in this same stock before. Did
   it usually bounce back? By how much? If the answer isn't good enough
   after accounting for trading costs, it passes.
4. **Rank the candidates.** If more than one stock looks like a good buy, it
   ranks them and picks the strongest one(s), given how much cash is
   available and how many "slots" it's allowed to hold at once.
5. **Buy.** If a slot is open (or a current holding looks clearly weaker
   than a new candidate), it buys.
6. **Watch what it's holding.** Every round, it re-checks everything it
   already owns: if a holding has gone up enough, it sells for the profit.
   If it's dropped too much, it sells to cut the loss. If it's just sitting
   there held too long without doing much of anything, it eventually sells
   that too, so it doesn't tie up money in a stock going nowhere forever.
7. **Remember what happened.** It keeps a private notebook of every dip it
   saw, what it decided, and what actually happened next — so its "does this
   dip usually pay off?" judgment keeps getting better informed over time,
   not just repeating a fixed rule forever.

## The "bonus trade" (Opportunistic Opportunity)

Alongside the normal watchlist logic, it also keeps an eye on one specific
pair of assets (for example, a "safer" one and a "spicier" one). If the
spicier one looks like it's about to outperform the safer one, and the robot
is confident enough based on past results, it will swap from the safer one
into the spicier one — but only once per day, at most, and only when the
odds clearly favor it.

## It can optionally go looking for new stocks on its own

By default, it only ever considers the specific list of stocks you gave it.
There's an optional "discovery" mode you can turn on where it also samples
from the broader market, checks whether a newly-found stock is priced
reasonably and trades enough volume to be worth bothering with, and only
adds it to its permanent watchlist if it also clears the same "does the dip
usually pay off" bar as everything else. It never trades a stock just
because it exists — discovery only ever expands the list of *candidates*,
not the buying rules themselves.

## The guardrails, in plain terms

- **It never panics into a crash.** If the news looks bad, or its own
  models say a stock has stopped being trustworthy, it just won't open a
  new position — it doesn't try to be clever about a falling market.
- **It never bets everything on nothing.** It only trades when there's
  enough of a track record behind the decision. No track record, no trade.
- **It never forgets a position.** Anything it buys stays on its radar
  every single round until it's sold — it can't "lose track" of a holding.
- **If something breaks — the internet, the news source, a data
  feed — it just skips that check and carries on as safely as it can,
  rather than crashing or freezing.** A crash mid-trading-day is worse than
  a cautious skip.
- **It writes home once a day.** If you've turned on email reports, you get
  a plain-English summary of what it looked at and what (if anything) it did.

## The important disclaimer, restated simply

This is not a magic money machine. It is a rule-following program that
makes decisions based on patterns in past prices and news — those patterns
can and do fail. Start with Alpaca's **paper trading** (fake money) and
watch it for a good while before ever considering real money. Nothing here
is financial advice, and there is no guarantee it makes money — it can lose
money, including all of it.
