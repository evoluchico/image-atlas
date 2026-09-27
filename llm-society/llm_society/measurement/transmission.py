"""Linguistic transmission from the exposure record.

For each agent i and round t >= 2, a word x is a *candidate* if it already
exists in the run (someone used it before t) but i has never used it. The
candidate is *exposed* if x appeared in any message that was placed in i's
context when i generated its round-t message. Adoption = i uses x at t.

    P(adopt | exposed)  vs  P(adopt | unexposed)

The unexposed rate is the baseline for spontaneous use (e.g. shared model
priors); `adoption_lift` is their ratio. Adoption events record the first
use together with the prior exposures (when, how many, from whom).

The unit is currently a word; the same logic applies to phrases, frames or
concepts by supplying a different `units_of` extractor.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd


def transmission_analysis(
    messages: pd.DataFrame,
    exposures: pd.DataFrame,
    units_of: Callable[[str, str], set[str]],
) -> tuple[list[dict], list[dict]]:
    """Returns (per-round metric rows as dicts, adoption event rows).
    `units_of(message_id, text)` extracts the cultural units of a message."""
    msgs = messages.sort_values("sequence")
    units = {mid: units_of(mid, text) for mid, text in zip(msgs["message_id"], msgs["text"])}
    sender = dict(zip(msgs["message_id"], msgs["sender_id"]))
    by_round_agent = {(r, a): mid for mid, r, a in zip(msgs["message_id"], msgs["round"], msgs["sender_id"])}
    gen_exp = exposures[exposures["phase"] == "generation"]
    exposed_ids = gen_exp.groupby(["round", "agent_id"])["message_id"].apply(list).to_dict()

    rounds = sorted(msgs["round"].unique())
    agents = sorted(msgs["sender_id"].unique())
    used: dict[str, set[str]] = defaultdict(set)
    run_vocab: set[str] = set()
    # History of exposures: (agent, unit) -> list of (round, sender)
    exposure_log: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    metric_rows, events = [], []

    for t in rounds:
        n_exp = n_exp_adopt = n_unexp = n_unexp_adopt = 0
        for a in agents:
            seen = exposed_ids.get((t, a), [])
            exposed_units: set[str] = set()
            for mid in seen:
                if sender.get(mid) == a:
                    continue
                for u in units.get(mid, ()):
                    exposed_units.add(u)
                    exposure_log[(a, u)].append((t, sender[mid]))
            mid = by_round_agent.get((t, a))
            if mid is None:
                continue
            produced = units[mid]
            if t > rounds[0]:
                candidates = run_vocab - used[a]
                exp_c = candidates & exposed_units
                unexp_c = candidates - exposed_units
                adopted = produced & candidates
                n_exp += len(exp_c)
                n_exp_adopt += len(adopted & exp_c)
                n_unexp += len(unexp_c)
                n_unexp_adopt += len(adopted & unexp_c)
                for u in sorted(adopted):
                    log = exposure_log.get((a, u), [])
                    events.append(
                        {
                            "round": int(t),
                            "agent_id": a,
                            "unit": u,
                            "exposed_before_adoption": bool(log),
                            "first_exposure_round": int(log[0][0]) if log else None,
                            "n_exposures": len(log),
                            "exposing_agents": ",".join(sorted({s for _, s in log})),
                        }
                    )
        for a in agents:
            mid = by_round_agent.get((t, a))
            if mid is not None:
                used[a] |= units[mid]
                run_vocab |= units[mid]
        if t > rounds[0]:
            pe = n_exp_adopt / n_exp if n_exp else float("nan")
            pu = n_unexp_adopt / n_unexp if n_unexp else float("nan")
            metric_rows.append(
                {
                    "round": int(t),
                    "adoption_rate_exposed": pe,
                    "adoption_rate_unexposed": pu,
                    "adoption_lift": pe / pu if pu and not np.isnan(pu) and pu > 0 else float("nan"),
                    "n_exposed_candidates": float(n_exp),
                }
            )
    return metric_rows, events
