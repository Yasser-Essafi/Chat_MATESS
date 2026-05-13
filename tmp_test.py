"""Comprehensive end-to-end test suite for the STATOUR chatbot.

Categories tested:
- Analytics (months, years, breakdowns, comparisons, charts)
- Prediction (scenarios, months, edge cases)
- Researcher (factor questions, news, web)
- Normal (greetings, multilingual)
- Follow-up context preservation
- Edge cases (typos, ambiguity, language mixing)
"""
import sys
import time
import json
import requests
import statistics

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:5000"


def new_conv():
    r = requests.post(f"{BASE}/api/conversations", timeout=30)
    if r.status_code != 200:
        return None
    j = r.json()
    return j.get("id") or j.get("conversation_id")


def chat(message, conv_id=None):
    body = {"message": message}
    if conv_id:
        body["conversation_id"] = conv_id
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/api/chat", json=body, timeout=300)
        elapsed = (time.time() - t0) * 1000
    except Exception as e:
        return {"error": f"REQUEST_FAILED: {e}", "wall_elapsed_ms": (time.time() - t0) * 1000}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:500], "wall_elapsed_ms": elapsed}
    data = r.json()
    data["wall_elapsed_ms"] = round(elapsed, 0)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases — (label, message, expected_agent, must_have, must_not, options)
# options.fresh_conv: True = new conversation (default)
# options.in_conv:    str  = continue this named conversation key
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ─── Analytics — month-specific ────────────────────────────────────────
    ("A1 month_feb_2026", "donne-moi les arrivées de février 2026", "analytics",
        ["1,372,858"], ["jailbreak", "Impossible", "année 2026", "total annuel"], {}),
    ("A2 month_jan_2025", "combien de touristes en janvier 2025", "analytics",
        ["janv", "2025"], ["jailbreak", "Impossible"], {}),
    ("A3 month_juillet_2024", "arrivées en juillet 2024", "analytics",
        ["juil", "2024"], ["jailbreak", "Impossible"], {}),
    ("A4 month_only_no_year", "arrivées en mars", "analytics",
        ["mars"], ["jailbreak", "Impossible"], {}),

    # ─── Analytics — annual / fast-path ─────────────────────────────────────
    ("A5 year_2024_total", "combien d'arrivées en 2024", "analytics",
        ["2024"], ["Impossible"], {}),
    ("A6 year_2025_mre", "combien de MRE en 2025", "analytics",
        ["2025", "MRE"], ["Impossible"], {}),
    ("A7 year_2025_tes", "combien de TES en 2025", "analytics",
        ["2025", "TES"], ["Impossible"], {}),
    ("A8 grand_total", "quel est le total des arrivées depuis 2019", "analytics",
        ["arriv"], ["Impossible"], {}),
    ("A9 last_period", "quelles sont les données les plus récentes", "analytics",
        ["2026"], ["Impossible"], {}),

    # ─── Analytics — top / breakdown ────────────────────────────────────────
    ("A10 top5_pays", "top 5 pays de résidence en 2025", "analytics",
        ["pays", "2025"], ["Impossible", "Nationalité"], {}),
    ("A11 top10_voies", "répartition par voie en 2024", "analytics",
        ["voie", "2024"], ["Impossible"], {}),
    ("A12 evolution_mensuelle", "évolution mensuelle des arrivées en 2024", "analytics",
        ["2024"], ["Impossible"], {}),

    # ─── Analytics — comparisons ────────────────────────────────────────────
    ("A13 compare_years", "comparaison arrivées 2024 vs 2025", "analytics",
        ["2024", "2025"], ["Impossible"], {}),

    # ─── Analytics — Hébergement/Nuitées estimées ───────────────────────────
    # Ces tests vérifient que les questions hébergement n'utilisent PAS les données APF.
    # must_not: "postes frontières", "MRE", "TES" — termes exclusifs à APF.
    # must_not utilise "postes frontières" et "MRE" (exclusifs APF) au lieu de "TES"
    # car "tes" est sous-chaîne de "touristes", "hôtes", "gîtes" dans les réponses hébergement.
    ("H1 nuitees_2024", "combien de nuitées en 2024", "analytics",
        ["2024"], ["postes frontières", "MRE", "Impossible"], {}),
    ("H2 nuitees_type_hebergement", "nuitées par type d'hébergement en 2024", "analytics",
        ["2024", "nuit"], ["postes frontières", "MRE", "Impossible"], {}),
    ("H3 top5_regions_nuitees", "top 5 régions par nuitées en 2023", "analytics",
        ["2023"], ["postes frontières", "MRE", "Impossible"], {}),
    ("H4 arrivees_hotelières_2025", "arrivées hôtelières en 2025", "analytics",
        ["2025"], ["postes frontières", "MRE", "Impossible"], {}),

    # ─── Prediction — month-specific ────────────────────────────────────────
    ("P1 pred_fev_mai_2027", "estimation du flux pour février et mai 2027", "prediction",
        ["Fév", "Mai"], ["0.0%/an", "scénario **pessimiste**"], {}),
    ("P2 pred_juillet_2027", "estimation des arrivées en juillet 2027", "prediction",
        ["Jul"], ["0.0%/an"], {}),

    # ─── Prediction — year-only ─────────────────────────────────────────────
    ("P3 pred_2027_baseline", "estimation pour 2027", "prediction",
        ["2027"], ["0.0%/an", "scénario **pessimiste**"], {}),
    ("P4 pred_external_factors", "estimation 2027 basée sur les facteurs externes", "prediction",
        ["Contexte externe"], ["0.0%/an", "scénario **pessimiste**"], {}),
    ("P5 pred_optimistic", "scénario optimiste pour 2028", "prediction",
        ["2028"], ["0.0%/an"], {}),
    ("P6 pred_pessimistic", "scénario pessimiste pour 2027", "prediction",
        ["2027"], ["0.0%/an"], {}),

    # ─── Researcher — factor / context questions ─────────────────────────
    ("R1 why_decline", "pourquoi la baisse du tourisme international", "researcher",
        [], ["Impossible"], {}),
    ("R2 vision_2030", "qu'est-ce que la vision 2030 du tourisme marocain", "researcher",
        ["2030"], ["Impossible"], {}),
    ("R3 mondial_2030", "quel impact aura la coupe du monde 2030 sur le Maroc", "researcher",
        ["2030"], ["Impossible"], {}),

    # ─── Normal — greetings / platform ─────────────────────────────────────
    ("N1 bonjour", "bonjour", "normal", [], [], {}),
    ("N2 hello_en", "hello", "normal", [], [], {}),
    ("N3 salam_ar", "السلام عليكم", "normal", [], [], {}),
    ("N4 platform_q", "qu'est-ce que STATOUR", "normal", ["STATOUR"], [], {}),

    # ─── Edge cases — typos / mixed ──────────────────────────────────────
    ("E1 typo_arrivees_no_accent", "donne moi les arrivees de fevrier 2026", "analytics",
        ["2026"], ["Impossible"], {}),
    ("E2 mixed_lang", "show me arrivals in 2025", "analytics",
        ["2025"], ["Impossible"], {}),
    ("E3 short_query", "données 2024", "analytics",
        ["2024"], ["Impossible"], {}),

    # ─── Follow-up tests (within same conversation) ─────────────────────
    ("F1a setup_q", "top 5 pays de résidence en 2025", "analytics",
        ["2025"], ["Impossible"], {"in_conv": "follow1"}),
    ("F1b followup_donne_chart", "donne-moi un graphique", None,
        [], ["Impossible"], {"in_conv": "follow1"}),

    ("F2a setup_pred", "estimation 2027 baseline", "prediction",
        ["2027"], ["0.0%/an"], {"in_conv": "follow2"}),
    ("F2b followup_pessimistic", "et en pessimiste", None,
        ["2027"], [], {"in_conv": "follow2"}),

    # ─── Out-of-range years (should reroute to researcher) ──────────────
    ("O1 historical_2010", "tourisme au Maroc en 2010", None,
        [], ["Impossible"], {}),

    # ─── Tests analyse causale — doivent aller au researcher ─────────────────
    ("X1 causal_hausse", "pourquoi la hausse des arrivées en juillet 2025", "researcher",
        [], ["Impossible"], {}),

    ("X2 causal_pays", "pourquoi les touristes britanniques ont augmenté en 2024", "researcher",
        ["2024"], ["Impossible"], {}),

    ("X3 ambiguity_arrivees", "combien d'arrivées en 2024", "analytics",
        ["arrivées", "2024"], [], {}),

    # ─── Tests prévision — taux de croissance doit être > 5% ──────────────────
    ("P7 pred_growth_rate_sane", "estimation flux touristique 2027", "prediction",
        ["2027"], ["0.0%/an", "0.1%/an", "0.2%/an"], {}),

    ("P8 pred_with_factors", "prévision optimiste 2028", "prediction",
        ["2028", "scénario", "optimiste"], ["0.0%/an"], {}),
]


def run():
    print(f"\n{'#' * 72}")
    print(f"# COMPREHENSIVE TEST SUITE — {len(TESTS)} tests")
    print(f"{'#' * 72}\n")

    # Conv tracking for follow-up tests
    conv_map = {}

    results = []
    timings = []

    for label, msg, expected_agent, must_have, must_not, opts in TESTS:
        # Determine conversation
        in_conv_key = opts.get("in_conv")
        if in_conv_key:
            if in_conv_key not in conv_map:
                conv_map[in_conv_key] = new_conv()
            conv = conv_map[in_conv_key]
        else:
            conv = new_conv()

        if not conv:
            print(f"❌ {label}: failed to create conversation")
            results.append((label, "ERROR", ["no conv"]))
            continue

        d = chat(msg, conv)

        if "error" in d:
            wall = d.get("wall_elapsed_ms", 0)
            print(f"❌ {label}: {d['error']} (wall={wall}ms)")
            results.append((label, "ERROR", [d["error"]]))
            continue

        agent = d.get("agent", "?")
        cls_ms = d.get("classification_time_ms", 0)
        total_ms = d.get("total_time_ms", 0)
        wall_ms = d.get("wall_elapsed_ms", 0)
        rerouted = d.get("rerouted", False)
        chart = d.get("chart_url")
        resp = d.get("response", "")

        timings.append((label, cls_ms, total_ms, wall_ms, agent))

        issues = []
        if expected_agent and agent != expected_agent and not rerouted:
            issues.append(f"agent={agent} (expected {expected_agent})")
        for s in must_have:
            if s.lower() not in resp.lower():
                issues.append(f"missing '{s}'")
        for s in must_not:
            if s.lower() in resp.lower():
                issues.append(f"forbidden '{s}'")

        status = "PASS" if not issues else "FAIL"
        marker = "✅" if status == "PASS" else "❌"
        chart_mark = " 📊" if chart else ""
        re_mark = " ↪️" if rerouted else ""
        print(f"{marker} {label} | agent={agent}{re_mark}{chart_mark} | "
              f"cls={cls_ms:.0f}ms total={total_ms:.0f}ms wall={wall_ms:.0f}ms")
        if issues:
            for i in issues:
                print(f"     ⚠️  {i}")
            print(f"     RESP: {resp[:300]}{'...' if len(resp)>300 else ''}")

        results.append((label, status, issues))

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'#' * 72}\n# SUMMARY\n{'#' * 72}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    error = sum(1 for _, s, _ in results if s == "ERROR")
    print(f"Total: {len(results)} | ✅ Pass: {passed} | ❌ Fail: {failed} | 💥 Error: {error}")

    if failed or error:
        print(f"\n--- FAILURES ---")
        for label, status, issues in results:
            if status != "PASS":
                print(f"  {status} {label}: {issues}")

    # ─── Performance breakdown ──────────────────────────────────────────
    if timings:
        print(f"\n--- PERFORMANCE (per agent) ---")
        by_agent = {}
        for label, cls, total, wall, agent in timings:
            by_agent.setdefault(agent, []).append((cls, total, wall))
        for agent, samples in sorted(by_agent.items()):
            n = len(samples)
            avg_cls = sum(s[0] for s in samples) / n
            avg_total = sum(s[1] for s in samples) / n
            avg_wall = sum(s[2] for s in samples) / n
            med_total = statistics.median(s[1] for s in samples)
            max_total = max(s[1] for s in samples)
            print(f"  {agent:12s} n={n:2d}  cls avg={avg_cls:5.0f}ms  "
                  f"total avg={avg_total:5.0f}ms median={med_total:5.0f}ms max={max_total:5.0f}ms  "
                  f"wall avg={avg_wall:5.0f}ms")

        all_total = [t[2] for t in timings]
        print(f"\n  ALL AGENTS  n={len(timings)}  "
              f"total avg={sum(all_total)/len(all_total):.0f}ms  "
              f"median={statistics.median(all_total):.0f}ms  "
              f"max={max(all_total):.0f}ms")


if __name__ == "__main__":
    run()
