from src.news_review_filter import MAX_REVIEW_POOL_SIZE, select_review_pool


def candidate(cid, title, topic_group="OTHER_SYSTEMIC", summary="", source="BBC News", priority="P1",
             published_at="2026-09-07T01:00:00+00:00", event_summary=None):
    return {
        "candidate_id": cid,
        "title": title,
        "summary": summary,
        "source": source,
        "priority": priority,
        "published_at": published_at,
        "url": f"https://example.com/{cid}",
        "topic_group": topic_group,
        "event_category": "other",
        "event_summary": title if event_summary is None else event_summary,
    }


def review_ids(result):
    return {item["candidate_id"] for item in result["review_candidates"]}


def filtered_ids(result):
    return {item["candidate_id"] for item in result["filtered_candidates"]}


# --- Stage B borderline / reserve / near-miss always wins, even with no keyword signal ---

def test_borderline_flagged_candidate_is_kept_even_without_any_keyword_signal():
    c = candidate("a", "Some obscure niche story with no obvious signal words")
    result = select_review_pool([c], {"borderline_ids": ["a"]})
    assert review_ids(result) == {"a"}


def test_reserve_flagged_candidate_is_kept_even_without_any_keyword_signal():
    c = candidate("a", "Some obscure niche story with no obvious signal words")
    result = select_review_pool([c], {"reserve_ids": ["a"]})
    assert review_ids(result) == {"a"}


def test_topic_cap_dropped_candidate_is_kept_even_without_any_keyword_signal():
    c = candidate("a", "Some obscure niche story with no obvious signal words")
    result = select_review_pool([c], {"topic_cap_dropped_ids": ["a"]})
    assert review_ids(result) == {"a"}


def test_candidate_not_flagged_by_stage_b_and_without_signal_is_filtered():
    c = candidate("a", "Some obscure niche story with no obvious signal words")
    result = select_review_pool([c], {"borderline_ids": ["other-id"]})
    assert review_ids(result) == set()
    assert filtered_ids(result) == {"a"}


# --- US macro / Fed / Treasury / dollar ---

def test_fed_unselected_news_enters_review_pool():
    c = candidate("fed1", "Federal Reserve holds interest rates steady", topic_group="US_MARKET_MACRO")
    result = select_review_pool([c])
    assert review_ids(result) == {"fed1"}


def test_us_treasury_yield_enters_review_pool():
    c = candidate("t1", "US Treasury yield climbs after auction", topic_group="US_MARKET_MACRO")
    result = select_review_pool([c])
    assert review_ids(result) == {"t1"}


def test_dollar_index_dxy_enters_review_pool():
    c = candidate("dxy1", "US dollar index (DXY) hits fresh six-month high", topic_group="US_MARKET_MACRO")
    result = select_review_pool([c])
    assert review_ids(result) == {"dxy1"}


def test_bare_ambiguous_macro_words_do_not_auto_qualify():
    """Regression: a bare 'rates'/'jobs'-style word with no explicit US
    monetary-policy context must not qualify just for the word's presence."""
    c = candidate("x1", "Auction clearance rates jump in Sydney property market", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()
    assert filtered_ids(result) == {"x1"}


# --- Mega-cap tech / AI / semiconductor structural news ---

def test_nvidia_ai_chip_update_enters_review_pool():
    c = candidate("nv1", "Nvidia unveils new AI chip for data centers", topic_group="AI_CHIPS")
    result = select_review_pool([c])
    assert review_ids(result) == {"nv1"}


def test_spacex_starship_incident_enters_review_pool():
    c = candidate("sx1", "SpaceX Starship suffers explosion during test flight", topic_group="MEGA_CAP_TECH")
    result = select_review_pool([c])
    assert review_ids(result) == {"sx1"}


def test_mega_cap_acquisition_enters_review_pool_even_below_homepage_bar():
    c = candidate("amzn1", "Amazon announces acquisition of logistics startup", topic_group="MEGA_CAP_TECH")
    result = select_review_pool([c])
    assert review_ids(result) == {"amzn1"}


def test_ordinary_consumer_electronics_update_is_filtered():
    """Regression: mentioning a tracked company alone is not enough -- a trivial
    product update (new color, minor refresh) must still be filtered."""
    c = candidate("cons1", "Amazon launches new color options for Kindle e-reader", topic_group="MEGA_CAP_TECH")
    result = select_review_pool([c])
    assert review_ids(result) == set()
    assert filtered_ids(result) == {"cons1"}


# --- Geopolitics / energy with real market transmission ---

def test_major_oil_shipping_geopolitics_is_retained():
    c = candidate("oil1", "Oil prices surge after Iran threatens Strait of Hormuz shipping", topic_group="GEOPOLITICS")
    result = select_review_pool([c])
    assert review_ids(result) == {"oil1"}


def test_china_export_control_geopolitics_is_retained():
    c = candidate("cn1", "China tightens export controls on rare earth chip materials", topic_group="GEOPOLITICS")
    result = select_review_pool([c])
    assert review_ids(result) == {"cn1"}


# --- Explicit exclusion categories ---

def test_ordinary_foreign_gdp_is_filtered():
    c = candidate(
        "gdp1", "Turkey cuts 2027 GDP growth forecast as elections beckon",
        topic_group="OTHER_SYSTEMIC",
        summary="Ankara's revision comes as fallout from the Iran war reverberates across the region.",
    )
    result = select_review_pool([c])
    assert review_ids(result) == set()
    assert filtered_ids(result) == {"gdp1"}


def test_ordinary_local_politics_is_filtered():
    c = candidate("pol1", "German state election sees coalition talks continue", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()


def test_children_lifestyle_news_is_filtered():
    c = candidate("kid1", "Pediatricians recommend new screen time guidelines for toddlers", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()


def test_ordinary_agricultural_commodity_news_is_filtered():
    c = candidate("sugar1", "Sugar prices ease on ample Brazilian harvest outlook", topic_group="ENERGY_COMMODITIES")
    result = select_review_pool([c])
    assert review_ids(result) == set()


def test_ordinary_local_real_estate_news_is_filtered():
    c = candidate("re1", "Suburban home prices tick up in local housing market", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()


def test_generic_cybersecurity_news_is_filtered():
    c = candidate("cyber1", "Researchers warn of new phishing campaign targeting small businesses", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()


# --- Regression: real 2026-09-07 replay data ---

def test_company_name_glued_to_chinese_event_summary_still_matches():
    """Regression from the real 2026-09-07 replay: Stage A's Chinese
    event_summary often glues an English company name directly onto adjacent
    Chinese characters with no space (e.g. "SpaceX继续..."). Python's \\w (and
    therefore the shared word-boundary keyword matcher) treats CJK ideographs
    as word characters, so a keyword immediately followed by a Chinese
    character has no boundary on that side and silently fails to match unless
    this module normalizes the text first."""
    c = candidate(
        "sx2", "Starbase Infrastructure Advances Toward Flight 14", topic_group="MEGA_CAP_TECH",
        event_summary="SpaceX继续推进Starbase基础设施建设以支持第14次飞行。",
    )
    result = select_review_pool([c])
    assert review_ids(result) == {"sx2"}


# --- OTHER_SYSTEMIC default exclusion ---

def test_other_systemic_is_not_auto_kept_without_a_high_confidence_signal():
    c = candidate("other1", "Company announces routine quarterly update", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == set()
    assert filtered_ids(result) == {"other1"}


def test_other_systemic_is_kept_when_text_carries_a_high_confidence_signal():
    """topic_group is only a display bucket, never a gate -- an OTHER_SYSTEMIC
    item whose actual text is macro-relevant must still qualify."""
    c = candidate("other2", "Federal Reserve chair signals rate path amid inflation data", topic_group="OTHER_SYSTEMIC")
    result = select_review_pool([c])
    assert review_ids(result) == {"other2"}


# --- Pool-size cap, diversity, and no minimum ---

def test_no_minimum_pool_size_when_nothing_qualifies():
    candidates = [candidate(str(i), f"Ordinary story {i} with no signal") for i in range(5)]
    result = select_review_pool(candidates)
    assert result["review_candidate_count"] == 0
    assert result["review_candidates"] == []


def test_max_pool_cap_is_enforced_on_a_high_volume_day():
    candidates = [
        candidate(f"macro{i}", f"Federal Reserve official {i} discusses inflation outlook", topic_group="US_MARKET_MACRO")
        for i in range(40)
    ]
    result = select_review_pool(candidates)
    assert result["review_candidate_count"] == MAX_REVIEW_POOL_SIZE
    assert result["review_filtered_count"] == 40 - MAX_REVIEW_POOL_SIZE
    assert all(item["filter_reason"] == "capped_at_max_pool_size" for item in result["filtered_candidates"])


def test_truncation_preserves_topic_diversity_over_a_single_dominant_category():
    macro = [
        candidate(f"macro{i}", f"Federal Reserve official {i} discusses inflation outlook", topic_group="US_MARKET_MACRO")
        for i in range(40)
    ]
    geo = [candidate("geo1", "Oil prices surge after Iran threatens Strait of Hormuz shipping", topic_group="GEOPOLITICS")]
    result = select_review_pool(macro + geo)
    assert result["review_candidate_count"] == MAX_REVIEW_POOL_SIZE
    assert "geo1" in review_ids(result)


# --- Coverage/purity invariants ---

def test_every_unselected_candidate_is_accounted_for_as_kept_or_filtered():
    candidates = [
        candidate("fed1", "Federal Reserve holds interest rates steady", topic_group="US_MARKET_MACRO"),
        candidate("pol1", "German state election sees coalition talks continue", topic_group="OTHER_SYSTEMIC"),
        candidate("nv1", "Nvidia unveils new AI chip for data centers", topic_group="AI_CHIPS"),
    ]
    result = select_review_pool(candidates)
    assert result["unselected_candidate_count"] == 3
    assert result["review_candidate_count"] + result["review_filtered_count"] == 3
    assert review_ids(result) | filtered_ids(result) == {"fed1", "pol1", "nv1"}


def test_does_not_mutate_input_candidates():
    c = candidate("fed1", "Federal Reserve holds interest rates steady", topic_group="US_MARKET_MACRO")
    original = dict(c)
    select_review_pool([c])
    assert c == original


def test_empty_input_yields_empty_pool():
    result = select_review_pool([])
    assert result == {
        "review_candidates": [],
        "filtered_candidates": [],
        "unselected_candidate_count": 0,
        "review_candidate_count": 0,
        "review_filtered_count": 0,
    }
