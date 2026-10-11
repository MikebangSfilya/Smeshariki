"""Adapt installed mechanics to a world with no historical campaign."""

from pathlib import Path

import re



def mask_literals(text):

    return re.sub(r'#[^\n]*|"(?:\\.|[^"\\])*"',lambda m:" "*len(m[0]),text)



def remove_blocks(text, key, invalid):

    masked=mask_literals(text); edits=[]

    for match in re.finditer(r'\b'+re.escape(key)+r'\s*=\s*\{',masked):

        depth=1;pos=match.end()

        while depth and pos<len(masked):

            depth+=(masked[pos]=="{")-(masked[pos]=="}");pos+=1

        body=text[match.end():pos-1]

        if invalid(body):edits.append((match.start(),pos,""))

    # Nested removals must not invalidate the outer edit's offsets.

    outer=[]

    for edit in edits:

        if not any(a<=edit[0] and b>=edit[1] for a,b,_ in outer):

            outer.append(edit)

    for a,b,r in reversed(outer): text=text[:a]+r+text[b:]

    return text



def sanitise(text):

    valid_focus={"RDC_common_management","RDC_central_crisis","RDC_valley_breakup"}

    # Generic IDs occur in the retained focus tree.

    generic=(Path(__file__).resolve().parents[1]/"common/national_focus/generic.txt").read_text(encoding="utf-8-sig")

    valid_focus.update(re.findall(r'\bid\s*=\s*([A-Za-z0-9_-]+)',generic))

    text=re.sub(r'\bhas_completed_focus\s*=\s*\"?([A-Za-z0-9_-]+)\"?',lambda m:m[0] if m[1] in valid_focus else "always = no",text)

    for key in ("complete_national_focus","unlock_national_focus","uncomplete_national_focus"):

        text=re.sub(r'\b'+key+r'\s*=\s*([A-Za-z0-9_-]+)',lambda m:m[0] if m[1] in valid_focus else "",text)

    for key in ("country_event","news_event"):

        text=remove_blocks(text,key,lambda body:True)

    text=remove_blocks(text,"load_focus_tree",lambda body:True)

    text=remove_blocks(text,"focus_progress",lambda body:not any(re.search(r'\bfocus\s*=\s*'+re.escape(fid)+r'\b',body) for fid in valid_focus))

    for key in ("activate_mission","unlock_decision_tooltip","unlock_decision","remove_decision","activate_decision_category","unlock_decision_category"):
        text=remove_blocks(text,key,lambda body:True)
    # Decisions from the historical campaign do not exist in this scenario.

    for key in ("activate_mission","cancel_mission","remove_mission","unlock_decision_tooltip","unlock_decision","remove_decision","add_days_remove","unlock_decision_category","unlock_decision_category_tooltip","activate_decision_category","remove_decision_category"):

        text=re.sub(r'\b'+key+r'\s*=\s*(?:\"[^\"]+\"|[A-Za-z0-9_-]+)',"",text)

    for key in ("has_active_mission","has_completed_decision","has_decision","has_focus_tree"):

        text=re.sub(r'\b'+key+r'\s*=\s*(?:\"[^\"]+\"|[A-Za-z0-9_-]+)',"always = no",text)

    text=remove_blocks(text,"add_days_mission_timeout",lambda body:True)
    # Old scripted character recruitment has no application to the new countries.

    for key in ("recruit_character","retire_character"):

        text=re.sub(r'\b'+key+r'\s*=\s*(?:\"[^\"]+\"|[A-Za-z0-9_-]+)',"",text)

    return text



def prepare_compatibility(game,root):

    files={}

    # Existing country definitions are passive dependencies, not playable nations.

    for source in (game/"common/country_tags").glob("*.txt"):

        if source.name=="zz_dynamic_countries.txt":continue

        text=source.read_text(encoding="utf-8-sig")


        files[str(source.relative_to(game)).replace("\\","/")]=text

        for tag in re.findall(r'^\s*([A-Z0-9]{3})\s*=\s*"countries/',text,re.M):

            files[f"history/countries/{tag}-inactive.txt"]="set_politics = { ruling_party = neutrality }\nset_popularities = { neutrality = 100 }\n"

    # Retain helpers and national spirits needed by equipment, laws and systems.

    directories=("common/scripted_triggers","common/scripted_effects","common/ideas","common/dynamic_modifiers","common/characters")

    for directory in directories:

        for source in (game/directory).glob("*.txt"):

            files[str(source.relative_to(game)).replace("\\","/")]=sanitise(source.read_text(encoding="utf-8-sig"))

    # Other installed mechanical data can contain historical focus checks.

    # Override those checks without changing equipment, technologies or combat.

    for source in list((game/"common").rglob("*.txt"))+list((game/"music").glob("*.txt")):

        rel=str(source.relative_to(game)).replace("\\","/")

        if any(rel.startswith(d+"/") for d in (

            "common/characters","common/national_focus","common/country_tags",

            "common/bookmarks","common/on_actions","common/ai_focuses","common/ai_strategy",

            "common/ai_strategy_plans","common/decisions",

            "common/scripted_localisation","common/military_industrial_organization/organizations",

            "common/ai_faction_theaters")):continue

        text=source.read_text(encoding="utf-8-sig")

        fixed=sanitise(text)

        if text!=fixed:files[rel]=fixed

    for source in (game/"common/military_industrial_organization/organizations").glob("*.txt"):
        files[str(source.relative_to(game)).replace("\\","/")]=sanitise(source.read_text(encoding="utf-8-sig"))
    # Generic industrial organisations.

    source=game/"common/military_industrial_organization/organizations/00_generic_organization.txt"

    files[str(source.relative_to(game)).replace("\\","/")]=sanitise(source.read_text(encoding="utf-8-sig"))

    # Generic AI economics/production.

    source=game/"common/ai_strategy/default.txt"

    files[str(source.relative_to(game)).replace("\\","/")]=sanitise(source.read_text(encoding="utf-8-sig"))

    # The UI initializes tutorial objectives and hints even in ordinary games.
    # Keep its required structure, but resolve every target against the new world.
    tutorial=(game/"tutorial/tutorial.txt").read_text(encoding="utf-8-sig")
    tutorial=re.sub(r'\bstate\s*=\s*\d+',"state = 1",tutorial)
    tutorial=re.sub(r'\btarget\s*=\s*\d+',"target = 1",tutorial)
    tutorial=re.sub(r'\bdirection_pointer\s*=\s*\d+',"direction_pointer = 1",tutorial)
    tutorial=re.sub(r'\bhighlight_states_trigger\s*=\s*\{[^}]*\}',"highlight_states_trigger = { 1 }",tutorial)
    tutorial=re.sub(r'\bhighlight_provinces\s*=\s*\{[^}]*\}',"highlight_provinces = { 1 }",tutorial)
    tutorial=tutorial.replace('target = "ETH"','target = "NAT"')
    tutorial=re.sub(r'#[^\n]*',"",tutorial)
    files["tutorial/tutorial.txt"]=tutorial
    files["common/achievements.txt"]="# Historical achievements unavailable.\n"

    # Avoid missing generated general/admiral names.

    names=[]

    for c in __import__("json").loads((root/"tools/world.json").read_text(encoding="utf-8-sig"))["countries"]:

        names.append(c["tag"]+' = { male = { names = { "Алексей" "Иван" "Михаил" "Павел" "Сергей" } surnames = { "Лесной" "Речной" "Долинский" "Озерный" "Луговой" } } female = { names = { "Анна" "Мария" "Елена" } surnames = { "Лесная" "Речная" "Долинская" } } surnames = { "Долинский" "Луговой" } callsigns = { "Ромашка" "Ветер" "Волна" } }\n')

    files["common/names/romashkova_names.txt"]="".join(names)

    import zipfile
    for archive in (game/"dlc").rglob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith("music/") and name.endswith(".txt"):
                    original=z.read(name).decode("utf-8-sig")
                    fixed=sanitise(original)
                    if original!=fixed:files[name]=fixed
    # Some installed DLCs are loose directories, not ZIP archives.
    for package_directory in (game/"dlc",game/"integrated_dlc"):
        for source in package_directory.glob("*/music/*.txt"):
            original=source.read_text(encoding="utf-8-sig")
            fixed=sanitise(original)
            if original!=fixed:files["music/"+source.name]=fixed
    # Explicit generic theater for the new strategic-region IDs.

    doc=(game/"common/ai_faction_theaters/ai_faction_theaters.txt").read_text(encoding="utf-8-sig")

    (root/"build/theater_format.txt").write_text(doc,encoding="utf-8")

    files["common/ai_faction_theaters/romashkova_theaters.txt"]='romashkova_theater = { name = RDC_faction regions = { '+ " ".join(map(str,range(1,19)))+' } cancel = { has_war = no } ai_will_do = { base = 10 } }\n'

    from validate_world import parse
    for rel,text in list(files.items()):
        if not rel.endswith(".txt"):continue
        for _ in range(12):
            try:parse(text);break
            except ValueError as e:
                if str(e)!="Unclosed brace":raise
                text+="\n}\n"
        files[rel]=text
    return files

