"""Independent checks of generated HOI4 files and full-resolution geometry.
Run after generate_world.py. Does not modify game inputs.
"""
from __future__ import annotations
import collections, csv, json, re, struct, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def parse(text):
    tokens=re.findall(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}]|<=|>=|!=|=|<|>|[^\s{}=<>!]+',text)
    tokens=[t for t in tokens if not t.startswith("#")]
    pos=0
    def block(closing=False):
        nonlocal pos
        result=[]
        while pos<len(tokens):
            key=tokens[pos]; pos+=1
            if key=="}":
                if not closing: raise ValueError("Unexpected closing brace")
                return result
            if key=="{":
                result.append((None,block(True))); continue
            if pos<len(tokens) and tokens[pos] in ("=","<",">","<=",">=","!="):
                pos+=1
                if pos>=len(tokens): raise ValueError("Missing value")
                value=tokens[pos]; pos+=1
                if value=="{": value=block(True)
                elif value=="}": raise ValueError("Missing value before closing brace")
                result.append((key,value))
            else: result.append((None,key))
        if closing: raise ValueError("Unclosed brace")
        return result
    return block()

def get(node,key,default=None):
    return next((value for name,value in node if name==key),default)

def values(node):
    return [int(value) for key,value in node if key is None]

def graph_of(labels):
    pairs=[]
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        good=a!=b
        pairs.append(np.stack([np.minimum(a[good],b[good]),np.maximum(a[good],b[good])],axis=1))
    graph=collections.defaultdict(set)
    for a,b in np.unique(np.concatenate(pairs),axis=0):
        graph[int(a)].add(int(b)); graph[int(b)].add(int(a))
    return graph

def reachable(graph,start,allowed):
    seen={start}; queue=collections.deque([start])
    while queue:
        for p in graph[queue.popleft()]:
            if p in allowed and p not in seen: seen.add(p); queue.append(p)
    return seen

def connected_counts(labels):
    """Run-length union-find avoids allocating a flood queue for 9 million pixels."""
    parent=[]; run_colors=[]; prev=[]
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for row in labels:
        bounds=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        current=[]
        for a,b in zip(bounds[:-1],bounds[1:]):
            uid=len(parent); parent.append(uid); color=int(row[a]); run_colors.append(color)
            current.append((int(a),int(b),color,uid))
        i=j=0
        while i<len(prev) and j<len(current):
            pa,pb,pc,pu=prev[i]; ca,cb,cc,cu=current[j]
            if pc==cc and min(pb,cb)>max(pa,ca): union(pu,cu)
            if pb<=cb: i+=1
            if cb<=pb: j+=1
        prev=current
    groups=collections.defaultdict(set)
    for uid,color in enumerate(run_colors): groups[color].add(find(uid))
    return {color:len(roots) for color,roots in groups.items()}

def validate():
    checks=[]; errors=[]
    def check(condition,message):
        (checks if condition else errors).append(message)
    manifest=json.loads((ROOT/"build/world_manifest.json").read_text(encoding="utf-8"))
    size=manifest["size"]; config=json.loads((ROOT/"tools/world.json").read_text(encoding="utf-8-sig")); countries={c["tag"]:c for c in manifest["countries"]}
    paths=list((ROOT/"common").rglob("*.txt"))+list((ROOT/"history").rglob("*.txt"))+list((ROOT/"events").glob("*.txt"))+list((ROOT/"tests").glob("*.txt"))+list((ROOT/"map/strategicregions").glob("*.txt"))
    parsed={}
    for path in paths:
        try: parsed[path]=parse(path.read_text(encoding="utf-8-sig"))
        except ValueError as e: errors.append(str(path.relative_to(ROOT))+": "+str(e))
    check(len(countries)==11,"11 playable countries")
    definitions={}
    for row in csv.reader((ROOT/"map/definition.csv").open(encoding="utf-8"),delimiter=";"):
        if not row: continue
        pid=int(row[0]); check(pid not in definitions,f"Unique province ID {pid}")
        definitions[pid]={"rgb":tuple(map(int,row[1:4])),"type":row[4],"coastal":row[5]=="true","terrain":row[6],"continent":int(row[7])}
    check(sum(p["type"]=="land" for i,p in definitions.items() if i)==config["land_provinces"],f'{config["land_provinces"]} land provinces')
    check(sum(p["type"]=="sea" for p in definitions.values())==120,"120 sea provinces")
    check(sum(p["type"]=="lake" for p in definitions.values())==2,"2 lake provinces")
    rgb=np.asarray(Image.open(ROOT/"map/provinces.bmp"),dtype=np.uint32)
    check(rgb.shape==(size,size,3),"Province bitmap dimensions and 24-bit RGB")
    packed=rgb[:,:,0]+(rgb[:,:,1]<<8)+(rgb[:,:,2]<<16)
    color_to_id={r["rgb"][0]+(r["rgb"][1]<<8)+(r["rgb"][2]<<16):pid for pid,r in definitions.items()}
    colors,pixel_counts=np.unique(packed,return_counts=True)
    check(len(color_to_id)==len(definitions),"Province RGB colours are unique")
    check(set(map(int,colors))<=set(color_to_id),"Every pixel colour is registered")
    check({color_to_id[int(c)] for c in colors}==set(definitions)-{0},"Every province is painted; no dummy pixels")
    tiny={color_to_id[int(c)]:int(area) for c,area in zip(colors,pixel_counts) if area<64}
    check(not tiny,f"Province areas remain playable; no tiny forced centres: {tiny}")
    keys=np.array(sorted(color_to_id),dtype=np.uint32); ids=np.array([color_to_id[int(k)] for k in keys],dtype=np.int32)
    labels=ids[np.searchsorted(keys,packed)]
    a,b,c,d=labels[:-1,:-1],labels[:-1,1:],labels[1:,:-1],labels[1:,1:]
    check(not np.any((a!=b)&(a!=c)&(a!=d)&(b!=c)&(b!=d)&(c!=d)),"No four-way province junctions")
    physical=graph_of(labels)
    print("Checking full-resolution province connectivity...",flush=True)
    disconnected={pid:count for pid,count in connected_counts(labels).items() if count!=1}
    check(not disconnected,f"All provinces are four-connected: {disconnected}")
    water={pid for pid,p in definitions.items() if p["type"]=="sea"}
    land={pid for pid,p in definitions.items() if p["type"]=="land" and pid}
    check(all(int(x) in water for x in np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]),"Ocean surrounds every map edge")
    check(reachable(physical,min(water),water)==water,"Sea and both navigable rivers form one connected network")
    for pid,p in definitions.items():
        if p["type"]=="lake": check(not bool(physical[pid]&water),f"Lake {pid} is enclosed by land, not joined to the sea")
    for pid in land:
        check(definitions[pid]["coastal"]==bool(physical[pid]&water),f"Coastal flag for province {pid}")
    sizes={"heightmap.bmp":(size,size),"terrain.bmp":(size,size),"rivers.bmp":(size,size),"cities.bmp":(size,size),"trees.bmp":(round(size*600/2048),)*2,"world_normal.bmp":(size//2,)*2}
    for name,expected in sizes.items():
        image=Image.open(ROOT/"map"/name)
        check(image.size==expected,f"Dimensions of {name}")
        with (ROOT/"map"/name).open("rb") as source: header=source.read(54)
        bits=struct.unpack_from("<H",header,28)[0]
        check(bits==(24 if name=="world_normal.bmp" else 8),f"Bit depth of {name}")
    tree_types=set(np.unique(np.asarray(Image.open(ROOT/"map/trees.bmp"))).tolist())
    check(tree_types<={0,5,6,28,29} and len(tree_types)>1,"Trees use existing temperate forest models, not palm or missing slots")
    for name in ["colormap_rgb_cityemissivemask_a.dds"]+[f"colormap_water_{i}.dds" for i in range(3)]:
        image=Image.open(ROOT/"map/terrain"/name)
        expected=(size,size) if name.startswith("colormap_rgb") else (size//2**(int(name[-5])+1),)*2
        check(image.size==expected,f"DDS dimensions {name}")
    memberships={}; state_nodes={}; state_owner={}; ports={}; capital_states={}
    state_files=sorted((ROOT/"history/states").glob("*.txt"))
    check(len(state_files)==72,"72 state files")
    for path in state_files:
        state=get(parsed[path],"state"); sid=int(get(state,"id")); history=get(state,"history")
        owner=get(history,"owner"); ps=values(get(state,"provinces"))
        check(owner in countries,f"Valid owner of state {sid}")
        state_nodes[sid]=set(ps); state_owner[sid]=owner
        for pid in ps:
            check(pid in land and pid not in memberships,f"Unique land membership of province {pid}")
            memberships[pid]=sid
        building=get(history,"buildings",[])
        for key,block in building:
            if key and key.isdigit():
                pid=int(key); check(pid in ps,f"Building {pid} belongs to state {sid}")
                if get(block,"naval_base") is not None:
                    check(pid in land and bool(physical[pid]&water),f"Port {pid} touches sea")
                    ports[pid]=sid
        for _,vp in [item for item in history if item[0]=="victory_points"]:
            pid=int(vp[0][1]); check(pid in ps,f"VP {pid} belongs to state {sid}")
    check(set(memberships)==land,"Every land province belongs to exactly one state")
    naval_spawns=set()
    for row in csv.reader((ROOT/"map/buildings.txt").open(encoding="utf-8"),delimiter=";"):
        sid=int(row[0]); x,z=float(row[2]),float(row[4]); px,py=int(x),size-1-int(z)
        check(0<=px<size and 0<=py<size,f"Building coordinates remain in map for state {sid}")
        if not (0<=px<size and 0<=py<size): continue
        pid=int(labels[py,px])
        check(memberships.get(pid)==sid,f"Building {row[1]} maps to land in its state {sid}")
        if row[1]=="naval_base_spawn":
            sea_id=int(row[6]); naval_spawns.add(pid)
            check(pid in land and definitions[pid]["coastal"],f"Naval spawn maps to coastal land {pid}")
            check(sea_id in physical[pid]&water,f"Naval spawn {pid} uses adjacent sea {sea_id}")
    check(naval_spawns=={pid for pid in land if definitions[pid]["coastal"]},"Every coastal province has an actual pixel-correct naval spawn")
    for row in csv.reader((ROOT/"map/unitstacks.txt").open(encoding="utf-8"),delimiter=";"):
        pid=int(row[0]);x,z=float(row[2]),float(row[4])
        check(int(labels[size-1-int(z),int(x)])==pid,f"Unit position maps to its actual province {pid}")
    river_pixels=np.asarray(Image.open(ROOT/"map/rivers.bmp"))
    river_mask=river_pixels<12
    check(not np.any(river_mask[:-1,:-1]&river_mask[:-1,1:]&river_mask[1:,:-1]&river_mask[1:,1:]),"Ordinary rivers use single-pixel paths without solid squares")
    for index,path in enumerate(config["tributaries"]):
        x,y=(round(v*(size-1)) for v in path[0])
        check(river_pixels[y,x]==0,f"Tributary {index+1} has a source at its sketch coordinate")
        x,y=(round(v*(size-1)) for v in path[len(path)//2])
        check(np.any(river_mask[max(0,y-5):y+6,max(0,x-5):x+6]),f"Tributary {index+1} is present through the central sketch waypoint")
    counts=collections.Counter(state_owner.values())
    for tag,country in countries.items(): check(counts[tag]==country["states"],f"State count for {tag}")
    regions={}; assigned=set()
    for path in (ROOT/"map/strategicregions").glob("*.txt"):
        region=get(parsed[path],"strategic_region"); rid=int(get(region,"id")); ps=values(get(region,"provinces"))
        check(bool(ps),f"Strategic region {rid} is nonempty")
        for pid in ps:
            check(pid in definitions and pid not in assigned,f"Unique strategic region for province {pid}")
            assigned.add(pid); regions[pid]=rid
        types={definitions[pid]["type"]!="sea" for pid in ps}
        check(len(types)==1,f"Land and water do not mix in strategic region {rid}")
    check(assigned==set(definitions)-{0},"Every province belongs to exactly one strategic region")
    for sid,ps in state_nodes.items(): check(len({regions[pid] for pid in ps})==1,f"State {sid} does not cross air regions")
    movement={pid:set(edges) for pid,edges in physical.items()}
    crossings=list(csv.reader((ROOT/"map/adjacencies.csv").open(encoding="utf-8"),delimiter=";"))[1:-1]
    check(len(crossings)==6,"Exactly six river crossings")
    for row in crossings:
        a,b,through=int(row[0]),int(row[1]),int(row[3])
        check(a in land and b in land and through in water,f"Valid crossing {a}-{b}")
        check(through in physical[a] and through in physical[b],f"Both banks touch through-province {through}")
        for pid,x,z in ((a,int(row[4]),int(row[5])),(b,int(row[6]),int(row[7]))):
            check(int(labels[size-1-z,x])==pid,f"Crossing endpoint position maps to province {pid}")
        movement[a].add(b); movement[b].add(a)
    rail_graph=collections.defaultdict(set)
    for line in (ROOT/"map/railways.txt").read_text().splitlines():
        level,count,*route=map(int,line.split())
        check(len(route)==count and len(route)>1,"Railway node count")
        for a,b in zip(route,route[1:]):
            check(a in land and b in land and b in movement[a],f"Railway {a}-{b} follows adjacency")
            check(state_owner[memberships[a]]==state_owner[memberships[b]],f"Railway {a}-{b} stays inside country")
            rail_graph[a].add(b); rail_graph[b].add(a)
    supply_nodes=set()
    for line in (ROOT/"map/supply_nodes.txt").read_text().splitlines():
        _,pid=map(int,line.split()); check(pid in land,f"Valid supply hub {pid}"); supply_nodes.add(pid)
    for tag,country in countries.items():
        path=ROOT/"history/countries"/f"{tag}-Romashkova.txt"
        history=parsed[path]; sid=int(get(history,"capital")); capital_states[tag]=sid
        px,py=country["anchor"]
        actual=manifest["provinces"][country["capital_province"]-1]["point"]
        check(((actual[0]/manifest["grid_size"]-py)**2+(actual[1]/manifest["grid_size"]-px)**2)**.5<.003,f"Capital of {tag} stays at the sketch anchor")
        check(state_owner.get(sid)==tag,f"Capital state of {tag} is owned")
        check(country["capital_province"] in state_nodes[sid],f"Capital province of {tag} is owned")
        owned={p for p in land if state_owner[memberships[p]]==tag}
        connected=reachable(movement,country["capital_province"],owned)
        disconnected=owned-connected
        declared=[island for island in config.get("maritime_islands",[]) if island["tag"]==tag]
        permitted=set()
        for island in declared:
            xs,ys=zip(*island["outline"]);lo_x,hi_x=min(xs)-.006,max(xs)+.006;lo_y,hi_y=min(ys)-.006,max(ys)+.006
            enclave={pid for pid in owned if lo_x<=manifest["provinces"][pid-1]["point"][1]/manifest["grid_size"]<=hi_x and lo_y<=manifest["provinces"][pid-1]["point"][0]/manifest["grid_size"]<=hi_y}
            check(bool(enclave),f"Declared island {island['key']} is owned by {tag}")
            if enclave:
                component=reachable(movement,min(enclave),owned)
                check(component==enclave,f"Island {island['key']} is one separate land component")
                check(bool(enclave&set(ports)),f"Island {island['key']} has an operating national port")
                check(bool(connected&set(ports)),f"{tag} mainland has a port for overseas supply")
                yy,xx=np.nonzero(np.isin(labels,list(enclave)))
                check(bool(np.all((xx/size>=lo_x)&(xx/size<=hi_x)&(yy/size>=lo_y)&(yy/size<=hi_y))),f"Island {island['key']} has no detached mainland pixels")
                check(not any(q in land-enclave for pid in enclave for q in physical[pid]),f"Island {island['key']} has no land contact")
                permitted.update(enclave)
        check(disconnected==permitted,f"{tag} mainland is connected; overseas holdings match declared islands")
        if tag=="NAT":
            ix,iy,rx,ry=config["island"]
            yy,xx=np.nonzero(np.isin(labels,list(owned)))
            inside=((xx/size-ix)/rx)**2+((yy/size-iy)/ry)**2<1.6
            check(bool(len(xx)) and bool(np.all(inside)),"All NATO pixels lie on the intended island; no mainland territory")
            check(not any(q in land-owned for p in owned for q in physical[p]),"NATO island has no land contact with the mainland")
            check(bool(owned&set(ports)),"NATO island has its own naval supply port")
            check(all((manifest["provinces"][pid-1]["tag"]=="NAT") for pid in owned),"NATO capitals, hubs and provinces stay on the island")
        connected_rails=reachable(rail_graph,country["capital_province"],owned)
        for hub in supply_nodes&owned:
            check(hub in connected_rails or bool(reachable(rail_graph,hub,owned)&set(ports)),f"Hub {hub} can reach capital or national port")
        oob=parsed[ROOT/"history/units"/f"{tag}_1936.txt"]
        units=get(oob,"units")
        divisions=[d for key,d in units if key=="division"]
        check(len(divisions)==country["divisions"],f"Starting division count for {tag}")
        for division in divisions: check(int(get(division,"location")) in owned,f"{tag} army spawns on owned land")
        for sub,dim in (("",(82,52)),("medium/",(41,26)),("small/",(10,7))):
            check(Image.open(ROOT/"gfx/flags"/sub/f"{tag}.tga").size==dim,f"{tag} flag dimensions")
    for field in ("divisions","civilian","military","dockyards","manpower"):
        valley=sum(c[field] for tag,c in countries.items() if tag!="NAT"); nato=countries["NAT"][field]
        check(abs(valley-nato)/max(valley,nato)<=.15,f"Starting {field} balance within 15% ({valley}:{nato})")
    actual_civ=actual_mil=actual_dock=0
    for path in state_files:
        block=get(get(get(parsed[path],"state"),"history"),"buildings")
        actual_civ+=int(get(block,"industrial_complex",0)); actual_mil+=int(get(block,"arms_factory",0)); actual_dock+=int(get(block,"dockyard",0))
    check(actual_civ==54 and actual_mil==54 and actual_dock==8,"Factories and dockyards match planned budgets")
    event=(ROOT/"events/romashkova_world_events.txt").read_text()
    check("NOT = { has_global_flag = smesh_world_fragmented }" in event and "set_global_flag = smesh_world_fragmented" in event,"Breakup is guarded against repeat/save-load")
    subjects=set(countries)-{"RDC","NAT"}
    check(set(re.findall(r"end_puppet\s*=\s*(\w+)",event))==subjects,"Exactly nine original countries become independent")
    check("dismantle_faction = yes" in event,"Shared faction is dissolved")
    check(not any(word in event for word in ("transfer_state","declare_war_on","annex_country")),"Breakup preserves borders and does not declare war")
    focus=(ROOT/"common/national_focus/00_council.txt").read_text()
    check(list(map(int,re.findall(r"cost\s*=\s*(\d+)\b",focus)))==[2,4,4],"Council opening fits the 70-day shared timeline")
    startup=(ROOT/"common/on_actions/00_romashkova.txt").read_text()
    check("id = smesh_world.2 days = 70" in startup,"Breakup deadline is scheduled at startup, independently of focus selection")
    krosh=get(parsed[ROOT/"common/national_focus/01_krosh.txt"],"focus_tree")
    opening=[node for key,node in krosh if key=="focus" and str(get(node,"id")).startswith("KRO_") and not str(get(node,"id")).startswith("KRO_post_")]
    check(len(opening)==5 and all(get(node,"cost")=="2" for node in opening),"Krosh has five 14-day prologue focuses")
    from krosh_prologue import FOCUSES
    check([get(node,"id") for node in opening]==list(FOCUSES),"Krosh opening sequence matches the authored story")
    check(all(get(get(node,"prerequisite",[]),"focus")==FOCUSES[i-1] for i,node in enumerate(opening) if i),"Opening focuses form one continuous chain")
    for language in ("russian","english"):
        check((ROOT/f"localisation/replace/krosh_prologue_l_{language}.yml").read_bytes().startswith(b"\xef\xbb\xbf"),f"{language} prologue localisation has UTF-8 BOM")
    bookmark=(ROOT/"common/bookmarks/00_romashkova.txt").read_text()
    check(all('"'+tag+'"' in bookmark for tag in countries),"Every country appears in the bookmark")
    for language in ("russian","english"):
        content=(ROOT/f"localisation/replace/romashkova_world_l_{language}.yml").read_bytes()
        check(content.startswith(b"\xef\xbb\xbf"),f"{language} localisation has UTF-8 BOM")

    elevation=np.asarray(Image.open(ROOT/"map/heightmap.bmp"))
    normals=np.asarray(Image.open(ROOT/"map/world_normal.bmp"))
    check(int(elevation.max())>=220,"Mountain peaks reach visible native elevation range")
    check(np.all(normals[:,:,2]>=128),"Normal up-axis is blue, using native RGB tangent layout")
    flat_water=elevation[::2,::2]==90
    sample=normals[flat_water]
    check(np.median(sample[:,2])==255 and abs(np.median(sample[:,0])-128)<=1 and abs(np.median(sample[:,1])-128)<=1,"Flat sea normals are 128,128,255")
    surface=np.asarray(Image.open(ROOT/"map/terrain/colormap_rgb_cityemissivemask_a.dds"))
    check(np.mean(surface[:,:,:3])>50 and np.std(surface[:,:,:3])>20,"DDS surface colours survive zero-emission alpha resampling")
    terrain_indices=set(np.unique(np.asarray(Image.open(ROOT/"map/terrain.bmp"))).tolist())
    check(terrain_indices<={0,4,5,6,13,14,15,16,17,20},"Surface texture indices exist in installed HOI4 terrain registry")
    color_defs=parse((ROOT/"common/countries/colors.txt").read_text(encoding="utf-8-sig"))
    for tag,country in countries.items():
        check(values(next(value for key,value in get(color_defs,tag) if key is None and isinstance(value,list)))==country["color"],f"Political renderer colour is registered for {tag}")

    towns=manifest.get("towns",[])
    check(len(towns)==len(config.get("towns",[])) and len(towns)>=20,"Named towns are all placed")
    check(len({town["province"] for town in towns})==len(towns),"Town provinces are unique")
    check(any(town["name"]=="С.К.А.М.Е.Й.К.А" and town["tag"]=="KRO" for town in towns),"Requested С.К.А.М.Е.Й.К.А is placed in Krosh territory")
    city_mask=np.asarray(Image.open(ROOT/"map/cities.bmp"))
    localised=(ROOT/"localisation/replace/romashkova_world_l_russian.yml").read_text(encoding="utf-8-sig")
    for town in towns:
        pid=town["province"]; y,x=town["point"]; scale=size//manifest["grid_size"]
        py,px=y*scale+scale//2,x*scale+scale//2
        check(int(labels[py,px])==pid and state_owner[memberships[pid]]==town["tag"],f"Town {town['name']} sits on its owned province")
        if town.get("city_sprawl",True):
            check((city_mask[py,px]==config['relief']['settlements']['city_selector'] if config.get('layout_snapshot') else city_mask[py,px]>0),f"Town {town['name']} has visible city-sprawl pixels")
        else:
            check((city_mask[py,px]==15 if config.get('layout_snapshot') else city_mask[py,px]==0),f"Rural or natural place {town['name']} has no urban sprawl")
        if town.get("coastal"):
            check(pid in ports and definitions[pid]["coastal"],f"Harbour {town['name']} has a coastal naval base")
        for border in town.get("borders_with",[])+([town["border_with"]] if town.get("border_with") else []):
            check(any(state_owner.get(memberships.get(q))==border for q in physical[pid]),f"Far garden {town['name']} touches {border}")
        if town.get("min_height"):
            check(definitions[pid]["terrain"]=="mountain",f"Mountain landmark {town['name']} keeps mountain terrain")
        if town.get("terrain_override")==17:
            check(definitions[pid]["terrain"]=="hills",f"Fenced garden {town['name']} has difficult hill movement")
        check(f'VICTORY_POINTS_{pid}:0 '+json.dumps(town["name"],ensure_ascii=False) in localised,f"Town {town['name']} has Russian localisation")
        history=get(get(parsed[ROOT/f"history/states/{town['state']}-Romashkova.txt"],"state"),"history")
        pairs=[values(value) for key,value in history if key=="victory_points"]
        check([pid,town["value"]] in pairs,f"Town {town['name']} has an actual in-game victory point")
        if town.get("fort"):
            province_block=get(get(history,"buildings"),str(pid),[])
            check(int(get(province_block,"bunker",0))==town["fort"],f"Fort at {town['name']} is present")
    gardens=[town for town in towns if town.get("kind")=="garden" and town["tag"]=="KOP"]
    check(len(gardens)>=8,"Kopatych has a network of named vegetable gardens")
    home=countries["KOP"]["anchor"]
    check(sum(np.hypot(town["point"][1]/manifest["grid_size"]-home[0],town["point"][0]/manifest["grid_size"]-home[1])<.04 for town in gardens)>=6,"Gardens surround Kopatych's house")
    check(any(town.get("fenced") and town.get("fort") and np.hypot(town["point"][1]/manifest["grid_size"]-home[0],town["point"][0]/manifest["grid_size"]-home[1])<.025 for town in gardens),"A fenced, fortified garden is beside Kopatych's house")
    villages=manifest.get("villages",[])
    check(len(villages)>=60,"Compact map contains secondary inhabited hamlets")
    check(not ({t["province"] for t in towns}&{v["province"] for v in villages}),"Background hamlets do not replace named sites")
    for village in villages:
        pid=village["province"];y,x=village["point"];scale=size//manifest["grid_size"]
        check((city_mask[y*scale+scale//2,x*scale+scale//2]==config['relief']['settlements']['city_selector'] if config.get('layout_snapshot') else city_mask[y*scale+scale//2,x*scale+scale//2]>0) and memberships[pid]==village["state"],f"Background hamlet {pid} has native houses in its state")
    graphics=(ROOT/"common/defines/zz_romashkova_graphics.lua").read_text(encoding="utf-8")
    visibility=config["label_visibility"]
    check(0.8*visibility["capital_max"]<=visibility["text_min"]<=visibility["text_max"]<visibility["capital_max"]<3000,"Settlement names hide shortly before capitals, below maximum camera distance")
    check(visibility["dot_min"]>visibility["text_max"] and visibility["dot_max"]>=visibility["dot_min"],"Settlement dots remain visible longer than names")
    check("TEXT_CUTOFF = {"+", ".join(map(str,visibility["text_cutoffs"]))+"}" in graphics and "CAPITAL_CUTOFF_MAX = "+str(visibility["capital_max"])+".0" in graphics,"Generated graphical cutoffs match the graduated label profile")
    far=[town for town in towns if town["name"]=="Дальний огород Копатыча"]
    check(len(far)==1 and far[0]["tag"]=="KOP" and set(far[0].get("borders_with",[]))=={"KRO","EZH"},"Far garden belongs to Kopatych at the three-country border")
    check("CAPITAL_ICON_CUTOFF = "+str(visibility["capital_max"])+".0" in graphics,"Separate capital icon cutoff does not hide the name early")
    priorities=dict((key,int(value)) for key,value in re.findall(r"NMapIcons\.(\w+_PRIORITY_(?:CAPITAL|VICTORY_POINTS)) = (\d+)",graphics))
    check(bool(priorities) and all(priorities.get(key.replace("VICTORY_POINTS","CAPITAL"),0)>value for key,value in priorities.items() if key.endswith("VICTORY_POINTS")),"Capitals take priority over settlement labels in every map mode")
    check("CITY_SPRAWL_SHRINK_DISTANCE = 1200.0" in graphics,"Expanded village models remain visible at wider zoom")
    preserved_path=ROOT/"build/preserved_user_files.json"
    if preserved_path.exists():
        import hashlib
        for name,digest in json.loads(preserved_path.read_text(encoding="utf-8")).items():
            check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f"Preserved user file {name}")
    features=json.loads((ROOT/"build/natural_features.json").read_text(encoding="utf-8"))
    check(features["ridge_chains"]>=8 and features["passes"]>=2,"Landscape has connected ridge chains and mountain passes")
    check(features["ordinary_rivers"]>=6,"Central river is retained with a branching tributary network")
    check(15<features["forest_land_percent"]<70,"Forest cover includes open valley and woodland clearings")
    check(features["farmland_land_percent"]>1,"Kopatych fields occupy a visible agricultural region")
    check(tree_types=={0,5,6,28,29},"Northern conifers and southern deciduous groves are both represented")
    from validate_relief import validate_relief
    validate_relief(ROOT,config,manifest,labels,check)
    report={"passed":len(checks),"failed":len(errors),"errors":errors,"countries":11,"states":72,"land_provinces":config["land_provinces"],"sea_provinces":120,"lakes":2,"crossings":6}
    (ROOT/"build/validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if errors else 0

if __name__=="__main__": sys.exit(validate())
