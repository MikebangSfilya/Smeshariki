"""Generate the opening story without changing the accepted map geometry.

The prose is edited in krosh_lore.json. All political effects live in hidden,
idempotent events; the audience of a literary record never executes them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCUSES = (
    "KRO_project_skameika", "KRO_property_dispute", "KRO_leave_council",
    "KRO_reclaim_bench", "KRO_security_guarantees",
)
TERRITORIES = {"ceded": 41, "occupied": [48, 50], "border_attacker": 50, "garden": 61}
RECORDS = {
    101: ("public", "Открытие С.К.А.М.Е.Й.К.А"),
    102: ("public", "Постановление принято"),
    103: ("personal", "Прекращение участия"),
    104: ("public", "Сложившееся положение"),
    105: ("KRO", "Предложение о защите направлено"),
    106: ("EZH", "Предложение Кроша"),
    107: ("KRO", "Соглашение с Ёжиком"),
    108: ("KRO", "Переговоры прекращены"),
    110: ("KRO", "Инцидент у дальнего огорода"),
    111: ("KOP", "Нарушение границы"),
    112: ("KRO", "Контроль над дальним огородом"),
    113: ("KOP", "Утрата дальнего огорода"),
    114: ("KRO", "Отвод отрядов"),
    115: ("KOP", "Граница удержана"),
    116: ("KRO", "Прекращение столкновения"),
    117: ("KOP", "Прекращение столкновения"),
    121: ("public", "Распад Ромашковой долины"),
}


def blocks(text, keyword):
    """Extract balanced top-level named blocks, ignoring comments and strings."""
    tokens = list(re.finditer(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}]|\b\w+\b|=', text))
    result = []
    for i, match in enumerate(tokens[:-2]):
        if match.group() != keyword or tokens[i + 1].group() != "=" or tokens[i + 2].group() != "{":
            continue
        depth = 0
        for end in tokens[i + 2:]:
            depth += (end.group() == "{") - (end.group() == "}")
            if depth == 0:
                result.append(text[match.start():end.end()])
                break
    return result


def generic_continuation(game, loc):
    """Keep the five opening focuses visible and unlock normal development below."""
    native = (game / "common/national_focus/generic.txt").read_text(encoding="utf-8-sig")
    branch = blocks(native, "focus")
    identifiers = [re.search(r'\bid\s*=\s*(\w+)', block).group(1) for block in branch]
    renamed = {fid: "KRO_post_" + fid for fid in identifiers}
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, identifiers)) + r')\b')
    result = []
    for block, original in zip(branch, identifiers):
        block = pattern.sub(lambda m: renamed[m.group()], block)
        block = re.sub(r'(\by\s*=\s*)(\d+)', lambda m: m.group(1) + str(int(m.group(2)) + 12), block, count=1)
        if not re.search(r'\bprerequisite\s*=', block):
            block = block[:-1] + f'\n prerequisite = {{ focus = {FOCUSES[-1]} }}\n}}'
        loc[renamed[original]] = "$" + original + "$"
        loc[renamed[original] + "_desc"] = "$" + original + "_desc$"
        result.append(block)
    return "\n".join(result)


def record(event_id, picture="GFX_report_event_generic_political"):
    key = f"smesh_krosh.{event_id}"
    audience = RECORDS[event_id][0]
    if audience == "personal":
        trigger = "tag = KRO"
    elif audience == "public":
        trigger = "smesh_story_country_trigger = yes"
    else:
        trigger = "tag = " + audience
    if event_id == 106:
        return f'''country_event = {{
 id = {key} title = {key}.t desc = {key}.d picture = {picture}
 is_triggered_only = yes
 trigger = {{ tag = EZH exists = yes NOT = {{ has_country_flag = smesh_ezh_answered }} }}
 option = {{ name = {key}.a ai_chance = {{ factor = 50 }}
  custom_effect_tooltip = smesh_ezh_accept_tt
  hidden_effect = {{ smesh_ezh_accept_effect = yes }}
 }}
 option = {{ name = {key}.b ai_chance = {{ factor = 50 }}
  custom_effect_tooltip = smesh_ezh_refuse_tt
  hidden_effect = {{ smesh_ezh_refuse_effect = yes }}
 }}
}}
'''
    return f'''country_event = {{
 id = {key} title = {key}.t desc = {key}.d picture = {picture}
 is_triggered_only = yes trigger = {{ {trigger} }}
 option = {{ name = {key}.a }}
}}
'''


def generate_prologue(countries, states, loc, game, write):
    by_id = {state["id"]: state for state in states}
    by_tag = {country["tag"]: country for country in countries}
    for sid, owner in ((41, "KRO"), (48, "EZH"), (50, "EZH"), (61, "KOP")):
        if by_id[sid]["tag"] != owner:
            raise ValueError(f"Prologue territory {sid} no longer belongs to {owner}")
    bench = next(t for t in by_id[40]["towns"] if t["name"] == "С.К.А.М.Е.Й.К.А")
    garden = next(t for t in by_id[61]["towns"] if t["name"] == "Дальний огород Копатыча")
    destination = by_id[50]["hub"]
    tags = [c["tag"] for c in countries]
    subjects = [tag for tag in tags if tag not in ("RDC", "NAT")]
    tag_filter = " ".join("tag = " + tag for tag in tags)
    write("common/scripted_triggers/romashkova_story.txt", f'''smesh_story_country_trigger = {{
 num_of_owned_states > 0 OR = {{ {tag_filter} }}
}}
''')
    effects = '''smesh_public_record_effect = {
 every_country = {
  limit = { smesh_story_country_trigger = yes }
  country_event = { id = $EVENT$ }
 }
}
smesh_ezh_accept_effect = {
 if = {
  limit = { tag = EZH NOT = { has_country_flag = smesh_ezh_answered } }
  set_country_flag = smesh_ezh_answered
  if = { limit = { is_subject = no NOT = { has_war_with = KRO } KRO = { exists = yes } }
   set_country_flag = smesh_ezh_accepted
   KRO = { puppet = EZH country_event = { id = smesh_krosh.107 } }
  }
 }
}
smesh_ezh_refuse_effect = {
 if = {
  limit = { tag = EZH NOT = { has_country_flag = smesh_ezh_answered } }
  set_country_flag = smesh_ezh_answered
  set_country_flag = smesh_ezh_refused
  if = { limit = { is_subject = no KRO = { exists = yes NOT = { has_war_with = EZH } } }
   KRO = { declare_war_on = { target = EZH type = annex_everything }
    country_event = { id = smesh_krosh.108 }
   }
  }
 }
}
smesh_krosh_diplomacy_effect = {
 if = {
  limit = { tag = KRO has_global_flag = smesh_world_fragmented
   has_country_flag = smesh_krosh_guarantees_ready
   NOT = { has_country_flag = smesh_krosh_diplomacy_started }
  }
  set_country_flag = smesh_krosh_diplomacy_started
  if = { limit = { EZH = { exists = yes is_subject = no NOT = { has_war_with = KRO } } }
   country_event = { id = smesh_krosh.105 }
   EZH = { country_event = { id = smesh_krosh.106 hours = 1 } }
  }
  if = { limit = {
   KOP = { exists = yes is_subject = no NOT = { has_war_with = KRO } NOT = { is_in_faction_with = KRO } }
   50 = { is_owned_by = KRO is_fully_controlled_by = KRO }
   61 = { is_owned_by = KOP is_fully_controlled_by = KOP }
  }
   set_global_flag = smesh_garden_conflict_active
   # The defending units march to the disputed area while the states are at peace.
   KOP = { if = { limit = { has_war = no }
    every_owned_state = { teleport_armies = { to_state = 61 limit = { tag = KOP } } }
   } }
   start_border_war = {
    change_state_after_war = no combat_width = 80 minimum_duration_in_days = 14
    attacker = { state = 50 num_provinces = 1
     on_win = smesh_krosh.30 on_lose = smesh_krosh.31 on_cancel = smesh_krosh.32 }
    defender = { state = 61 num_provinces = 1
     on_win = smesh_krosh.31 on_lose = smesh_krosh.30 on_cancel = smesh_krosh.32 }
   }
   country_event = { id = smesh_krosh.110 }
   KOP = { country_event = { id = smesh_krosh.111 } }
   country_event = { id = smesh_krosh.40 days = 90 }
  }
 }
}
'''
    write("common/scripted_effects/romashkova_story.txt", effects)
    startup_effects = "\n".join(f"puppet = {tag}\nadd_to_faction = {tag}" for tag in subjects)
    write("common/on_actions/00_romashkova.txt", f'''on_actions = {{
 on_startup = {{ effect = {{
  if = {{ limit = {{ NOT = {{ has_global_flag = smesh_world_initialized }} }}
   RDC = {{ create_faction = RDC_faction
    {startup_effects}
    country_event = {{ id = smesh_world.2 days = 70 }}
   }}
   set_global_flag = smesh_world_initialized
   log = "SMESH_WORLD_INITIALIZED"
  }}
 }} }}
}}
''')
    free = "\n".join(f"if = {{ limit = {{ {tag} = {{ is_subject_of = RDC }} }} end_puppet = {tag} }}" for tag in subjects)
    write("events/romashkova_world_events.txt", f'''add_namespace = smesh_world
country_event = {{
 id = smesh_world.2 hidden = yes is_triggered_only = yes
 trigger = {{ tag = RDC NOT = {{ has_global_flag = smesh_world_fragmented }} }}
 immediate = {{ country_event = {{ id = smesh_world.1 }} }}
}}
country_event = {{
 id = smesh_world.1 hidden = yes is_triggered_only = yes
 trigger = {{ tag = RDC NOT = {{ has_global_flag = smesh_world_fragmented }} }}
 immediate = {{
  set_global_flag = smesh_world_fragmented
  {free}
  if = {{ limit = {{ is_faction_leader = yes }} dismantle_faction = yes }}
  load_focus_tree = {{ tree = generic_focus keep_completed = no }}
  smesh_public_record_effect = {{ EVENT = smesh_krosh.121 }}
  KRO = {{ smesh_krosh_diplomacy_effect = yes }}
  log = "SMESH_WORLD_FRAGMENTED"
 }}
}}
''')
    council = ['focus_tree = { id = RDC_council country = { factor = 0 modifier = { add = 100 tag = RDC } } default = no']
    for i, (fid, cost) in enumerate(zip(("RDC_common_management", "RDC_central_crisis", "RDC_valley_breakup"), (2, 4, 4))):
        prereq = f"prerequisite = {{ focus = {('RDC_common_management', 'RDC_central_crisis')[i-1]} }}" if i else ""
        council.append(f'''focus = {{ id = {fid} icon = GFX_goal_generic_political_pressure x = 5 y = {i*2}
 cost = {cost} {prereq} available = {{ NOT = {{ has_global_flag = smesh_world_fragmented }} }}
 cancel_if_invalid = no ai_will_do = {{ factor = 1000 }}
 completion_reward = {{ add_political_power = 25 custom_effect_tooltip = smesh_fixed_breakup_tt }}
}}''')
    write("common/national_focus/00_council.txt", "\n".join(council) + "\n}\n")
    focus = ['focus_tree = { id = KRO_prologue country = { factor = 0 modifier = { add = 100 tag = KRO } } default = no',
             f'initial_show_position = {{ focus = {FOCUSES[0]} }}']
    icons = ("GFX_goal_generic_construct_civ_factory", "GFX_goal_generic_political_pressure",
             "GFX_goal_generic_neutrality_focus", "GFX_goal_generic_demand_territory", "GFX_goal_generic_position_armies")
    for i, (fid, icon) in enumerate(zip(FOCUSES, icons), 1):
        prerequisite = f"prerequisite = {{ focus = {FOCUSES[i-2]} }}" if i > 1 else ""
        focus.append(f'''focus = {{ id = {fid} icon = {icon} x = 10 y = {(i-1)*2} cost = 2
 {prerequisite} cancel_if_invalid = no continue_if_invalid = yes
 ai_will_do = {{ factor = 1000 }}
 completion_reward = {{ custom_effect_tooltip = {fid}_tt
  hidden_effect = {{ country_event = {{ id = smesh_krosh.{i} }} }} }}
}}''')
    focus.append(generic_continuation(game, loc))
    write("common/national_focus/01_krosh.txt", "\n".join(focus) + "\n}\n")
    write("common/national_focus/generic.txt", (game / "common/national_focus/generic.txt").read_text(encoding="utf-8-sig"))
    write("common/ai_strategy_plans/01_krosh.txt", f'''KRO_opening = {{
 name = "KRO_opening" desc = "KRO_opening" allowed = {{ original_tag = KRO }}
 enable = {{ NOT = {{ has_country_flag = smesh_krosh_guarantees_ready }} }}
 abort = {{ has_country_flag = smesh_krosh_guarantees_ready }}
 ai_national_focuses = {{ {' '.join(FOCUSES)} }}
}}
''')
    write("common/ai_strategy_plans/00_council.txt", '''RDC_opening = {
 name = "RDC_opening" desc = "RDC_opening" allowed = { original_tag = RDC }
 enable = { NOT = { has_global_flag = smesh_world_fragmented } }
 abort = { has_global_flag = smesh_world_fragmented }
 ai_national_focuses = { RDC_common_management RDC_central_crisis RDC_valley_breakup }
}
''')
    rewards = {
        1: '''every_country = { limit = { smesh_story_country_trigger = yes NOT = { tag = NAT } } add_stability = 0.03 }
KRO = { add_political_power = 25 }
smesh_public_record_effect = { EVENT = smesh_krosh.101 }''',
        2: f'''set_province_name = {{ id = {bench['province']} name = smesh_former_bench_site }}
set_victory_points = {{ province = {bench['province']} value = 0 }}
set_province_name = {{ id = {destination} name = smesh_bench_name }}
set_victory_points = {{ province = {destination} value = 3 }}
smesh_public_record_effect = {{ EVENT = smesh_krosh.102 }}''',
        3: '''KRO = { set_country_flag = smesh_krosh_left_council country_event = { id = smesh_krosh.103 } }''',
        4: '''41 = {
 teleport_armies = { to_state = 50 limit = { tag = KRO } }
 remove_core_of = KRO add_core_of = RDC
}
RDC = { transfer_state = 41 }
KRO = {
 transfer_state = 48 transfer_state = 50
 # The expedition assembles on the occupied southern territory.
 if = { limit = { has_war = no }
  every_owned_state = { teleport_armies = { to_state = 50 limit = { tag = KRO } } }
 }
}
smesh_public_record_effect = { EVENT = smesh_krosh.104 }''',
        5: '''KRO = { set_country_flag = smesh_krosh_guarantees_ready smesh_krosh_diplomacy_effect = yes }''',
    }
    events = ["add_namespace = smesh_krosh\n"]
    for i in range(1, 6):
        events.append(f'''country_event = {{
 id = smesh_krosh.{i} hidden = yes is_triggered_only = yes
 trigger = {{ tag = KRO NOT = {{ has_global_flag = smesh_krosh_stage_{i} }} }}
 immediate = {{ set_global_flag = smesh_krosh_stage_{i}
 {rewards[i]}
 }}
}}
''')
    for event_id, result, pair in ((30, "won", (112, 113)), (31, "lost", (114, 115)), (32, "cancelled", (116, 117))):
        transfer = "if = { limit = { 61 = { is_owned_by = KOP } } KRO = { transfer_state = 61 } }" if result == "won" else ""
        events.append(f'''country_event = {{
 id = smesh_krosh.{event_id} hidden = yes is_triggered_only = yes
 trigger = {{ has_global_flag = smesh_garden_conflict_active NOT = {{ has_global_flag = smesh_garden_conflict_resolved }} }}
 immediate = {{
  set_global_flag = smesh_garden_conflict_resolved
  set_global_flag = smesh_garden_conflict_{result}
  clr_global_flag = smesh_garden_conflict_active
  {transfer}
  KRO = {{ country_event = {{ id = smesh_krosh.{pair[0]} }} }}
  KOP = {{ country_event = {{ id = smesh_krosh.{pair[1]} }} }}
 }}
}}
''')
    events.append('''country_event = {
 id = smesh_krosh.40 hidden = yes is_triggered_only = yes
 trigger = { has_global_flag = smesh_garden_conflict_active }
 immediate = {
  cancel_border_war = { attacker = 50 defender = 61 }
  KRO = { country_event = { id = smesh_krosh.32 } }
 }
}
''')
    for event_id in RECORDS:
        picture = "GFX_report_event_generic_read_write" if event_id in (101, 102, 103, 105, 106, 107) else "GFX_report_event_generic_army"
        events.append(record(event_id, picture))
    write("events/krosh_prologue_events.txt", "\n".join(events))
    for folder in ("common/decisions", "common/ai_focuses"):
        write(folder + "/00_romashkova.txt", "# Vanilla country-specific content is not used in this world.\n")
    lore = json.loads((ROOT / "tools/krosh_lore.json").read_text(encoding="utf-8"))
    for language in ("russian", "english"):
        entries = lore[language]
        text = "l_" + language + ":\n" + "\n".join(
            " " + key + ":0 " + json.dumps(value, ensure_ascii=False) for key, value in entries.items()) + "\n"
        write(f"localisation/replace/krosh_prologue_l_{language}.yml", text, bom=True)
    route = {f"smesh_krosh.{eid}": audience for eid, (audience, _) in RECORDS.items()}
    manifest = {"focus_days": 14, "breakup_after_days": 70, "focuses": list(FOCUSES),
                "territories": TERRITORIES, "bench_original_province": bench["province"],
                "bench_destination_province": destination, "garden_province": garden["province"],
                "audiences": route, "ai_ezh_accept_chance": 50, "border_timeout_days": 90}
    write("build/prologue_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    docs = ["# Пролог Кроша\n", "Пять фокусов по 14 дней. Распад — через 70 дней от начала новой партии.\n",
            "Общие записи получают все 11 активных стран; личную — Крош; дипломатические — только участники.\n",
            "Северо-восточная область 41 передаётся Совету, области 48 и 50 — Крошу. Дом Ёжика остаётся в области 46. Огород — область 61.\n",
            "Референсы: пятилетние планы; постановления о национализации; выход из Лиги Наций; аншлюс; инцидент у моста Марко Поло. Исторические тексты не цитируются. Пограничный конфликт не перерастает в войну автоматически.\n"]
    for fid in FOCUSES:
        docs += ["## " + lore["russian"][fid] + "\n", lore["russian"][fid + "_desc"] + "\n"]
    for eid, (audience, title) in RECORDS.items():
        docs += [f"## {title} — {audience}\n", lore["russian"][f"smesh_krosh.{eid}.d"] + "\n"]
    write("build/krosh_prologue.md", "\n".join(docs))
    startup = " ".join(f"{tag} = {{ is_subject_of = RDC is_in_faction_with = RDC }}" for tag in subjects)
    independent = " ".join(f"{tag} = {{ NOT = {{ is_subject_of = RDC }} }}" for tag in subjects)
    write("tests/romashkova.txt", f'''last_date = 1937.1.2
tests = {{
 smesh_startup_invalid = {{ success = {{ date < 1936.2.1 has_global_flag = smesh_world_initialized {startup} NAT = {{ is_subject = no is_in_faction = no }} }} fail = {{ date > 1936.2.1 }} }}
 smesh_breakup_not_completed_after_70_days = {{ success = {{ date > 1936.3.11 has_global_flag = smesh_world_fragmented {independent} RDC = {{ is_in_faction = no }} }} fail = {{ date > 1936.3.13 }} }}
 smesh_breakup_not_persistent = {{ success = {{ date > 1937.1.1 has_global_flag = smesh_world_fragmented {independent} }} fail = {{ date > 1937.1.2 }} }}
 smesh_ezh_reply_inconsistent = {{ success = {{ date > 1936.4.1
  OR = {{ EZH = {{ has_country_flag = smesh_ezh_accepted is_subject_of = KRO }} EZH = {{ has_country_flag = smesh_ezh_refused }}
   KRO = {{ NOT = {{ has_country_flag = smesh_krosh_diplomacy_started }} }} }}
 }} fail = {{ date > 1936.4.2 }} }}
}}
''')
