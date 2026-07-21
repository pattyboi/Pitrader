# Audit findings

## CRYPTO-001: No purchases when dip signals have negative net expectancy

- **Observed:** 2026-07-19 11:17 EDT
- **Classification:** Expected safety behavior; no runtime or broker fault
- **Status:** Open as an operational finding
- **Last verified against code:** 2026-07-21

The crypto service was running, connected to Alpaca paper trading, and completing
its scheduled 15-minute evaluations. It did not submit purchase orders because
none of the evaluated symbols passed the strategy's profitability floor.

The active configuration required a 2.5% dip and at least +0.30% expected net
profit. Expected profit deducts the greater of the configured 0.50% round-trip
cost and the observed live spread. In the 11:17 EDT signal snapshot:

| Symbol | Dip | Qualified dip | Expected net edge |
| --- | ---: | :---: | ---: |
| BTC | 1.57% | No | -0.57% |
| ETH | 3.69% | Yes | -0.61% |
| BONK | 45.64% | Yes | -1.19% |
| CRV | 7.51% | Yes | -0.66% |
| DOGE | 8.61% | Yes | -0.79% |
| DOT | 9.15% | Yes | -0.77% |

The eligibility pipeline checks the raw expected-profit floor before applying
risk-posture ranking. Consequently, `CRYPTO_RISK_POSTURE="risky"` cannot promote
a negative-expectancy signal into the candidate list. The buy method is never
called when that list is empty, so there are no broker rejection messages.

`CRYPTO_FILL_QUALIFIED_SLOTS=true` does not change that conclusion. It fills
more fundable slots only *after* candidates pass the dip, observation,
expected-profit, out-of-sample, learned-edge, and news guards; it cannot turn
one of the negative-expectancy rows above into an eligible purchase.

### Assessment

This behavior is consistent with the strategy's documented safety controls.
Lowering the expected-profit floor, reducing the round-trip-cost assumption, or
allowing risk posture to override the floor would be a trading-policy change,
not a defect fix. Such a change should be supported by paper-fill spread data
and out-of-sample performance evidence before implementation.
