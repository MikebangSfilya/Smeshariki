"""Generate a complete HOI4 world. Installed game files are read-only inputs.
Requires Pillow and NumPy. Design coordinates are top-left; game positions are
converted to bottom-left. Only files listed in generated_files.json are owned.
"""
from __future__ import annotations
import argparse, collections, heapq, json, math, shutil, struct, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from relief import terrain_layers, world_normals, shaded_preview, noise_field
from settlements import place_towns, town_surface, place_villages, village_surface
ROOT = Path(__file__).resolve().parents[1]
GENERATED = []

def write(relative, content, bom=False):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig" if bom else "utf-8")
    GENERATED.append(relative)

def save_image(relative, image, **kwargs):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)
    if path.suffix.lower()==".bmp" and image.mode=="P":
        data=bytearray(path.read_bytes())
        # HOI4 requires biClrUsed/biClrImportant=0 (full 256-entry palette).
        offset=struct.unpack_from("<I",data,10)[0]
        palette_start=14+struct.unpack_from("<I",data,14)[0]
        if offset-palette_start<1024:
            padding=1024-(offset-palette_start)
            data[offset:offset]=bytes(padding)
            struct.pack_into("<I",data,10,offset+padding)
            struct.pack_into("<I",data,2,len(data))
        struct.pack_into("<II",data,46,0,0)
        path.write_bytes(data)
    GENERATED.append(relative)

def components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h,w = mask.shape; groups=[]
    for y,x in zip(*np.nonzero(mask)):
        if seen[y,x]: continue
        queue=collections.deque([(int(y),int(x))]); seen[y,x]=True; pts=[]
        while queue:
            cy,cx=queue.popleft(); pts.append((cy,cx))
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; queue.append((ny,nx))
        groups.append(np.asarray(pts,dtype=np.int32))
    return groups

def nearest(mask, xy):
    pts=np.column_stack(np.nonzero(mask))
    goal=np.array([xy[1],xy[0]])*mask.shape[0]
    p=pts[np.argmin(((pts-goal)**2).sum(axis=1))]
    return tuple(map(int,p))

def choose_seeds(mask,count,preferred=None,importance=None):
    groups=components(mask)
    if count<len(groups): raise ValueError(f"{count} seeds cannot cover {len(groups)} components")
    selected=[] if preferred is None else [preferred]
    for pts in groups:
        if any(np.any(np.all(pts==p,axis=1)) for p in selected): continue
        mean=pts.mean(axis=0)
        selected.append(tuple(map(int,pts[np.argmin(((pts-mean)**2).sum(axis=1))])))
    candidates=np.column_stack(np.nonzero(mask)).astype(np.float32)
    distance=np.full(len(candidates),np.inf,dtype=np.float32)
    for p in selected: distance=np.minimum(distance,((candidates-p)**2).sum(axis=1))
    score_weight=np.ones(len(candidates),dtype=np.float32) if importance is None else importance[candidates[:,0].astype(int),candidates[:,1].astype(int)]
    while len(selected)<count:
        p=tuple(map(int,candidates[np.argmax(distance*score_weight)])); selected.append(p)
        distance=np.minimum(distance,((candidates-p)**2).sum(axis=1))
    return selected

def flood(regions,seeds,noise,bridges=(),diagonals=False):
    h,w=regions.shape
    labels=np.zeros((h,w),dtype=np.int32)
    distance=np.full((h,w),np.inf,dtype=np.float64)
    heap=[]; extra=collections.defaultdict(list)
    for a,b in bridges: extra[a].append(b); extra[b].append(a)
    for label,(y,x) in seeds:
        labels[y,x]=label; distance[y,x]=0; heapq.heappush(heap,(0.0,label,y,x))
    while heap:
        dist,label,y,x=heapq.heappop(heap)
        if dist!=distance[y,x] or label!=labels[y,x]: continue
        neighbors=[(y-1,x,1.),(y+1,x,1.),(y,x-1,1.),(y,x+1,1.)]
        if diagonals:
            for oy,ox in ((-1,-1),(-1,1),(1,-1),(1,1)):
                ny,nx=y+oy,x+ox
                if 0<=ny<h and 0<=nx<w and regions[y,nx]==regions[y,x] and regions[ny,x]==regions[y,x]:
                    neighbors.append((ny,nx,math.sqrt(2)))
        neighbors.extend((ny,nx,math.hypot(ny-y,nx-x)) for ny,nx in extra.get((y,x),()))
        for ny,nx,step in neighbors:
            if not (0<=ny<h and 0<=nx<w) or not regions[ny,nx] or regions[ny,nx]!=regions[y,x]: continue
            nd=dist+step*(1.0+float(noise[ny,nx]))
            if nd<distance[ny,nx]:
                distance[ny,nx]=nd; labels[ny,nx]=label
                heapq.heappush(heap,(nd,label,ny,nx))
    if np.any((regions!=0)&(labels==0)): raise ValueError("Uncovered terrain: check islands and bridges")
    return labels

def repair_province_connectivity(labels,regions,provinces,point_scale=1):
    # Diagonal paths can leave corner-only pixels. Keep each seed's component
    # and merge detached pieces into a touching province of the same state/type.
    from validate_world import connected_counts
    province_regions=np.array([0]+[p["state"] if p["type"]=="land" else -1 if p["type"]=="sea" else -p["id"] for p in provinces])
    for _ in range(6):
        broken=[pid for pid,count in connected_counts(labels).items() if count>1]
        if not broken:return
        for pid in broken:
            ys,xs=np.nonzero(labels==pid)
            y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
            groups=components(labels[y0:y1,x0:x1]==pid)
            sy,sx=provinces[pid-1]["point"]
            sy=sy*point_scale+point_scale//2; sx=sx*point_scale+point_scale//2
            largest=max(range(len(groups)),key=lambda i:len(groups[i]))
            for group_index,group in enumerate(groups):
                points=group+np.array([y0,x0])
                if point_scale>1:
                    if group_index==largest:continue
                elif np.any(np.all(points==[sy,sx],axis=1)):continue
                choices=[]
                for oy,ox in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=points[:,0]+oy,points[:,1]+ox
                    valid=(ny>=0)&(ny<labels.shape[0])&(nx>=0)&(nx<labels.shape[1])
                    ny,nx=ny[valid],nx[valid]
                    valid=(labels[ny,nx]!=pid)&(province_regions[labels[ny,nx]]==province_regions[pid])
                    choices.extend(labels[ny[valid],nx[valid]].tolist())
                if not choices and point_scale>1:
                    # Rounded state corners can leave tiny isolated coastal tips.
                    # Merge only those tips into touching land of the same type.
                    for oy,ox in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny,nx=points[:,0]+oy,points[:,1]+ox
                        valid=(ny>=0)&(ny<labels.shape[0])&(nx>=0)&(nx<labels.shape[1])
                        for q in labels[ny[valid],nx[valid]]:
                            if q!=pid and provinces[int(q)-1]["type"]==provinces[pid-1]["type"]: choices.append(int(q))
                    same_country=[q for q in choices if provinces[q-1]["tag"]==provinces[pid-1]["tag"]]
                    choices=same_country
                if not choices:raise ValueError(f"Detached province {pid} has no compatible neighbour")
                counts=collections.Counter(choices)
                target=min(counts,key=lambda q:(-counts[q],q))
                labels[points[:,0],points[:,1]]=target
    raise ValueError("Province connectivity repair did not converge")


def allocate(weights,total,minima=None):
    minimum=np.ones(len(weights),dtype=int) if minima is None else np.array(minima,dtype=int)
    rest=total-int(minimum.sum())
    if rest<0: raise ValueError("Too few provinces for physical components")
    exact=np.array(weights,dtype=float)/sum(weights)*rest
    counts=minimum+np.floor(exact).astype(int)
    for i in np.argsort(-(exact-np.floor(exact)))[:total-int(counts.sum())]: counts[i]+=1
    return counts.tolist()


def curve_points(points, closed=False):
    pts=np.array(points,dtype=float)
    result=[]
    if closed:
        padded=np.vstack([pts[-1],pts,pts[0],pts[1]])
        segments=len(pts)
    else:
        padded=np.vstack([pts[0],pts,pts[-1]])
        segments=len(pts)-1
    for i in range(segments):
        a,b,c,d=padded[i:i+4]
        for t in np.linspace(0,1,12,endpoint=False):
            q=.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t)
            result.append(q.tolist())
    if not closed: result.append(pts[-1].tolist())
    return result

def river_layers(config,land):
    size=config["size"]
    raster=np.full((size,size),255,dtype=np.uint8)
    visual=Image.new("L",(size,size),0); pen=ImageDraw.Draw(visual)
    for index,waypoints in enumerate(config["tributaries"]):
        points=[tuple(round(v*(size-1)) for v in p) for p in curve_points(waypoints)]
        path=[points[0]]
        for (ax,ay),(bx,by) in zip(points,points[1:]):
            dx,dy=abs(bx-ax),abs(by-ay); sx=1 if bx>ax else -1; sy=1 if by>ay else -1
            x,y=ax,ay; ix=iy=0
            while ix<dx or iy<dy:
                if iy==dy or (ix<dx and (2*ix+1)*dy<(2*iy+1)*dx): x+=sx; ix+=1
                else: y+=sy; iy+=1
                if (x,y)!=path[-1]: path.append((x,y))
        width=config.get("tributary_widths",[6,3,2])[index]
        visible=Image.new("L",(size,size),0)
        ImageDraw.Draw(visible).line(points,fill=255,width=width,joint="curve")
        visual=Image.fromarray(np.maximum(np.asarray(visual),np.asarray(visible)))
        previous_rivers=raster<12
        for step,(x,y) in enumerate(path):
            if not land[y,x]: break
            if index and not previous_rivers[y,x] and any(previous_rivers[ny,nx] for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0<=ny<size and 0<=nx<size):
                raster[y,x]=1
                break
            if previous_rivers[y,x]:
                # A tributary ends one pixel before the already drawn trunk.
                if step: px,py=path[step-1]; raster[py,px]=1
                break
            raster[y,x]=(6 if step<len(path)*.55 else 9) if index==0 else (4 if index==1 else 3)
        x,y=path[0]; raster[y,x]=0
    # Pixel rounding on smaller maps can close a tiny 2x2 bend. Thin only
    # redundant river pixels while retaining every source/join and 4-way connectivity.
    wet=raster<12
    for _ in range(12):
        squares=wet[:-1,:-1]&wet[1:,:-1]&wet[:-1,1:]&wet[1:,1:]
        if not np.any(squares):break
        changed=False
        for y,x in zip(*np.nonzero(squares)):
            if not np.all(wet[y:y+2,x:x+2]):continue
            candidates=[(y+dy,x+dx) for dy,dx in ((0,0),(0,1),(1,0),(1,1)) if 3<=raster[y+dy,x+dx]<12]
            for py,px in candidates:
                if pixel_can_be_removed(wet,int(py),int(px),True):
                    raster[py,px]=255;wet[py,px]=False;changed=True;break
        if not changed:raise ValueError("A rounded river bend cannot be thinned without splitting")
    if np.any(wet[:-1,:-1]&wet[1:,:-1]&wet[:-1,1:]&wet[1:,1:]):raise ValueError("Unresolved river square")
    raster[~land]=254
    visual_array=np.asarray(visual).copy(); visual_array[~land]=0
    return raster,visual_array


def geometry(config):
    n=config["grid_size"]
    point=lambda p:tuple(round(v*(n-1)) for v in p)
    image=Image.new("L",(n,n),0); draw=ImageDraw.Draw(image)
    draw.polygon([point(p) for p in curve_points(config["continent"],True)],fill=1)
    x,y,rx,ry=config["island"]
    if config.get("island_outline"):
        draw.polygon([point(p) for p in curve_points(config["island_outline"],True)],fill=2)
    else:
        draw.ellipse([int((x-rx)*n),int((y-ry)*n),int((x+rx)*n),int((y+ry)*n)],fill=2)
    for domain,island in enumerate(config.get("maritime_islands",[]),3):
        draw.polygon([point(p) for p in curve_points(island["outline"],True)],fill=domain)
    base=np.asarray(image).copy(); river_masks=[]
    for river in config["navigable_rivers"]:
        layer=Image.new("L",(n,n),0); pen=ImageDraw.Draw(layer)
        pts=[point(p) for p in curve_points(river["points"])]
        factors=np.interp(np.linspace(0,len(river["points"])-1,len(pts)),np.arange(len(river["points"])),river.get("width_factors",[1]*len(river["points"])))
        widths=[max(7,round(river["width"]*n*factor)) for factor in factors]
        for (a,b),wa,wb in zip(zip(pts,pts[1:]),widths,widths[1:]):
            pen.line((a,b),fill=1,width=max(wa,wb))
        for (px,py),width in zip(pts,widths):
            pen.ellipse([px-width//2,py-width//2,px+width//2,py+width//2],fill=1)
        mask=np.asarray(layer).astype(bool); river_masks.append(mask); base[mask]=0
    lakes=np.zeros((n,n),dtype=np.uint8)
    for index,(x,y,rx,ry) in enumerate(config["lakes"],1):
        layer=Image.new("L",(n,n),0)
        lake_rng=np.random.default_rng(config["seed"]+index*91)
        contour=[]
        for angle in np.linspace(0,2*math.pi,18,endpoint=False):
            radial=lake_rng.uniform(.87,1.04)
            contour.append([x+rx*radial*math.cos(angle),y+ry*radial*math.sin(angle)])
        ImageDraw.Draw(layer).polygon([point(p) for p in curve_points(contour,True)],fill=index)
        lakes=np.maximum(lakes,np.asarray(layer))
    base[lakes>0]=0
    margin=round(n*.025)
    base[:margin,:]=0; base[-margin:,:]=0; base[:,:margin]=0; base[:,-margin:]=0
    land=base>0; crossings=[]
    for cross in config["crossings"]:
        cx,cy=point(cross["point"]); normal=np.array(cross["normal"],dtype=float)
        normal/=np.linalg.norm(normal); endpoints=[]
        for sign in (-1,1):
            for r in range(1,100):
                px,py=np.rint(np.array([cx,cy])+normal*sign*r).astype(int)
                if 0<=px<n and 0<=py<n and land[py,px]:
                    endpoints.append((int(py),int(px))); break
        if len(endpoints)!=2 or endpoints[0]==endpoints[1]: raise ValueError("Invalid crossing: "+cross["name"])
        crossings.append({**cross,"a":endpoints[0],"b":endpoints[1],"center":(cy,cx)})
    return base,lakes,river_masks,crossings

def adjacency(labels):
    pairs=[]
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        good=(a!=b)&(a>0)&(b>0)
        pairs.append(np.stack([np.minimum(a[good],b[good]),np.maximum(a[good],b[good])],axis=1))
    graph=collections.defaultdict(set)
    for a,b in np.unique(np.concatenate(pairs),axis=0):
        graph[int(a)].add(int(b)); graph[int(b)].add(int(a))
    return graph

def shortest(graph,source,target,allowed):
    queue=collections.deque([source]); parent={source:None}
    while queue:
        node=queue.popleft()
        if node==target:
            result=[]
            while node is not None: result.append(node); node=parent[node]
            return result[::-1]
        for other in sorted(graph[node]):
            if other in allowed and other not in parent:
                parent[other]=node; queue.append(other)
    return None

def palette_image(array,source):
    image=Image.fromarray(array.astype(np.uint8)).convert("P")
    palette = Image.open(source).getpalette()
    image.putpalette(palette if palette is not None else [v for i in range(256) for v in (i,i,i)])
    return image

def pixel_can_be_removed(array,y,x,value):
    if y==0 or x==0 or y==array.shape[0]-1 or x==array.shape[1]-1:return False
    mask=array[y-1:y+2,x-1:x+2]==value;mask[1,1]=False
    neighbours=[(dy,dx) for dy,dx in ((0,1),(2,1),(1,0),(1,2)) if mask[dy,dx]]
    if len(neighbours)<2:return True
    seen={neighbours[0]};queue=[neighbours[0]]
    while queue:
        a,b=queue.pop()
        for na,nb in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0<=na<3 and 0<=nb<3 and mask[na,nb] and (na,nb) not in seen:seen.add((na,nb));queue.append((na,nb))
    return all(p in seen for p in neighbours)


def round_country_corners(labels,country_ids,provinces,scale,radius):
    # Smooth national outlines while keeping every province and country connected.
    owners=country_ids[labels]; size=labels.shape[0]
    desired=owners.copy(); best=np.zeros(labels.shape,dtype=np.uint8)
    land=owners<=11
    for cid in range(1,12):
        probability=np.asarray(Image.fromarray(np.uint8(owners==cid)*255).filter(ImageFilter.GaussianBlur(radius*scale)))
        selected=(probability>best)&land
        best[selected]=probability[selected];desired[selected]=cid
    protected={(p["point"][0]*scale+scale//2,p["point"][1]*scale+scale//2) for p in provinces}
    candidates=np.column_stack(np.nonzero((desired!=owners)&land))
    for _ in range(3):
        changed=0
        for y,x in candidates:
            y,x=int(y),int(x)
            if y==0 or x==0 or y==size-1 or x==size-1 or (y,x) in protected:continue
            target=int(desired[y,x]);old=int(owners[y,x]);pid=int(labels[y,x])
            if target==old:continue
            choices=[int(labels[ny,nx]) for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if owners[ny,nx]==target]
            if not choices or not pixel_can_be_removed(owners,y,x,old) or not pixel_can_be_removed(labels,y,x,pid):continue
            labels[y,x]=min(choices);owners[y,x]=target;changed+=1
        if not changed:break


def smooth_province_boundaries(labels,provinces,scale,radius):
    """Round categorical outlines without interpolating registered RGB colours."""
    n=labels.shape[0]; count=len(provinces)+1; size=n*scale
    ids=labels.ravel(); yy=np.repeat(np.arange(n),n); xx=np.tile(np.arange(n),n)
    min_y=np.full(count,n,dtype=np.int32); max_y=np.zeros(count,dtype=np.int32)
    min_x=np.full(count,n,dtype=np.int32); max_x=np.zeros(count,dtype=np.int32)
    np.minimum.at(min_y,ids,yy); np.maximum.at(max_y,ids,yy)
    np.minimum.at(min_x,ids,xx); np.maximum.at(max_x,ids,xx)
    tags=list(dict.fromkeys(p["tag"] for p in provinces if p["type"]=="land"))
    country_ids={tag:i+1 for i,tag in enumerate(tags)}
    lake_codes={p["id"]:13+i for i,p in enumerate(p for p in provinces if p["type"]=="lake")}
    classes=np.array([0]+[country_ids[p["tag"]] if p["type"]=="land" else 12 if p["type"]=="sea" else lake_codes[p["id"]] for p in provinces],dtype=np.uint8)
    domain=np.repeat(np.repeat(classes[labels],scale,0),scale,1)
    result=np.zeros((size,size),dtype=np.int32); best=np.zeros((size,size),dtype=np.float32)
    pad=int(math.ceil(radius*3))+2
    for p in provinces:
        pid=p["id"]; y0=max(0,int(min_y[pid])-pad); y1=min(n,int(max_y[pid])+pad+1)
        x0=max(0,int(min_x[pid])-pad); x1=min(n,int(max_x[pid])+pad+1)
        mask=Image.fromarray(np.uint8(labels[y0:y1,x0:x1]==pid)*255)
        soft=mask.filter(ImageFilter.GaussianBlur(radius)).resize(((x1-x0)*scale,(y1-y0)*scale),Image.Resampling.BILINEAR)
        probability=np.asarray(soft,dtype=np.float32)
        ys=slice(y0*scale,y1*scale); xs=slice(x0*scale,x1*scale)
        selected=(probability>best[ys,xs])&(domain[ys,xs]==classes[pid])
        best[ys,xs][selected]=probability[selected]; result[ys,xs][selected]=pid
    if np.any(result==0): raise ValueError("Boundary smoothing left unassigned pixels")
    regions=np.array([0]+[p["state"] if p["type"]=="land" else -1 if p["type"]=="sea" else -p["id"] for p in provinces])
    repair_province_connectivity(result,regions[result],provinces,scale)
    # Move a centre into the surviving shape instead of painting an isolated pixel.
    for p in provinces:
        pid=p["id"]; y0=max(0,int(min_y[pid])-pad)*scale;y1=min(n,int(max_y[pid])+pad+1)*scale
        x0=max(0,int(min_x[pid])-pad)*scale;x1=min(n,int(max_x[pid])+pad+1)*scale
        mask=result[y0:y1,x0:x1]==pid
        interior=mask&np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1)
        ys,xs=np.nonzero(interior);ys=ys+y0;xs=xs+x0
        selected=(ys%scale==scale//2)&(xs%scale==scale//2)
        if not np.any(selected):
            ys,xs=np.nonzero(mask);ys=ys+y0;xs=xs+x0;selected=(ys%scale==scale//2)&(xs%scale==scale//2)
        if not np.any(selected):raise ValueError(f"Province {p['id']} has no interior centre")
        ys,xs=ys[selected],xs[selected]
        oy,ox=p["point"];goal=(oy*scale+scale//2,ox*scale+scale//2)
        at=np.argmin((ys-goal[0])**2+(xs-goal[1])**2)
        p["point"]=(int(ys[at])//scale,int(xs[at])//scale)
    round_country_corners(result,classes,provinces,scale,radius)
    return result


def grow_small_provinces(labels,provinces,scale,minimum=96):
    """Grow real connected outlines instead of forcing isolated centre pixels."""
    counts=np.bincount(labels.ravel(),minlength=len(provinces)+1)
    protected={(p["point"][0]*scale+scale//2,p["point"][1]*scale+scale//2) for p in provinces}
    for pid in np.flatnonzero((counts<minimum)&(counts>0)):
        if pid==0:continue
        province=provinces[int(pid)-1]
        ys,xs=np.nonzero(labels==pid)
        y0,y1=max(1,int(ys.min())-4),min(labels.shape[0]-1,int(ys.max())+5)
        x0,x1=max(1,int(xs.min())-4),min(labels.shape[1]-1,int(xs.max())+5)
        while counts[pid]<minimum:
            patch=labels[y0:y1,x0:x1]; mask=patch==pid
            border=(np.roll(mask,1,0)|np.roll(mask,-1,0)|np.roll(mask,1,1)|np.roll(mask,-1,1))&~mask
            candidates=[]
            for ay,ax in np.column_stack(np.nonzero(border)):
                y,x=int(ay)+y0,int(ax)+x0; old=int(labels[y,x]); neighbour=provinces[old-1]
                if (y,x) in protected or counts[old]<=minimum*2:continue
                if neighbour["tag"]!=province["tag"] or neighbour["type"]!=province["type"]:continue
                if not pixel_can_be_removed(labels,y,x,old):continue
                cy,cx=province["point"]
                candidates.append((neighbour["state"]!=province["state"],(y-cy*scale)**2+(x-cx*scale)**2,y,x,old))
            if not candidates:
                if counts[pid]<64:raise ValueError(f"Cannot grow narrow province {pid} safely")
                break
            _,_,y,x,old=min(candidates)
            labels[y,x]=pid;counts[pid]+=1;counts[old]-=1

def fix_junctions(labels,provinces=None,scale=1):
    countries=None;protected=set()
    if provinces is not None:
        tags=list(dict.fromkeys(p["tag"] for p in provinces if p["type"]=="land"))
        country_ids={tag:i+1 for i,tag in enumerate(tags)}
        codes=np.array([0]+[country_ids[p["tag"]] if p["type"]=="land" else -1 if p["type"]=="sea" else -p["id"] for p in provinces])
        countries=codes[labels]
        protected={(p["point"][0]*scale+scale//2,p["point"][1]*scale+scale//2) for p in provinces}
    for _ in range(8):
        a,b,c,d=labels[:-1,:-1],labels[:-1,1:],labels[1:,:-1],labels[1:,1:]
        bad=(a!=b)&(a!=c)&(a!=d)&(b!=c)&(b!=d)&(c!=d)
        ys,xs=np.nonzero(bad)
        # Longitude wraps in the engine. Stagger seam junctions by one pixel
        # without assigning any province a detached piece on the opposite edge.
        sa,sb,sc,sd=labels[:-1,-1],labels[:-1,0],labels[1:,-1],labels[1:,0]
        seam=(sa!=sb)&(sa!=sc)&(sa!=sd)&(sb!=sc)&(sb!=sd)&(sc!=sd)
        seam_y=np.nonzero(seam)[0]
        if not len(ys) and not len(seam_y): return
        labels[seam_y,0]=labels[seam_y+1,0]
        if provinces is None:
            labels[ys,xs]=labels[ys+1,xs]
        else:
            for y,x in zip(ys,xs):
                y,x=int(y),int(x)
                corners=((y,x),(y,x+1),(y+1,x),(y+1,x+1))
                if len({int(labels[a,b]) for a,b in corners})<4:continue
                choices=[]
                for ay,ax in corners:
                    if (ay,ax) in protected:continue
                    old=int(labels[ay,ax]);old_country=int(countries[ay,ax])
                    if not pixel_can_be_removed(labels,ay,ax,old):continue
                    for by,bx in corners:
                        if abs(ay-by)+abs(ax-bx)!=1:continue
                        new=int(labels[by,bx]);new_country=int(countries[by,bx])
                        if provinces[old-1]["type"]!=provinces[new-1]["type"]:continue
                        if old_country!=new_country and not pixel_can_be_removed(countries,ay,ax,old_country):continue
                        choices.append((old_country!=new_country,ay,ax,new,new_country))
                if not choices:raise ValueError(f"No connected repair for junction {x},{y}")
                _,ay,ax,new,new_country=min(choices)
                labels[ay,ax]=new;countries[ay,ax]=new_country
    raise ValueError("Unresolved four-way junction")

def generate(config,game):
    if config.get("layout_snapshot"):
        from relief_pass import generate_relief
        return generate_relief(config,game)
    rng=np.random.default_rng(config["seed"]); n=config["grid_size"]; size=config["size"]; scale=size//n
    if size%n or size%256: raise ValueError("Invalid map/grid dimensions")
    countries=config["countries"]
    print("1/7: geography and navigable rivers",flush=True)
    base,lakes,river_masks,crossings=geometry(config); land=base>0; sea=(~land)&(lakes==0)
    noise=.03*(noise_field(rng,n,48)+1)+rng.random((n,n)).astype(np.float32)*.003
    bridges=[(c["a"],c["b"]) for c in crossings]
    capitals={c["tag"]:nearest(base==(2 if c["tag"]=="NAT" else 1),c["anchor"]) for c in countries}
    height,forest,terrain,coastal,rgb,landscape=terrain_layers(config,land,sea,lakes,capitals)
    dy,dx=np.gradient(height)
    geographic_cost=1.5*np.clip(landscape["ridge"]/110,0,1.5)+.35*np.clip(np.hypot(dy,dx),0,4)+1.6*(landscape["valley"]>.965)
    noise+=geographic_cost
    importance=1+.38*landscape["valley"]+.35*np.clip(np.hypot(dy,dx),0,2)+.28*(noise_field(rng,n,76)+1)
    country_seeds=[(i+1,capitals[c["tag"]]) for i,c in enumerate(countries)]
    # Separate physical domains enforce island-only NATO ownership.
    for domain,island in enumerate(config.get("maritime_islands",[]),3):
        label=next(i for i,c in enumerate(countries,1) if c["tag"]==island["tag"])
        country_seeds.append((label,nearest(base==domain,island["anchor"])))
    owner=flood(base,country_seeds,noise,bridges,diagonals=True)
    # Authored border adjustments remain part of the mainland before states/provinces grow.
    for adjustment in config.get("territory_adjustments",[]):
        label=next(i for i,c in enumerate(countries,1) if c["tag"]==adjustment["tag"])
        patch=Image.new("L",(n,n),0)
        ImageDraw.Draw(patch).polygon([(round(x*n),round(y*n)) for x,y in adjustment["outline"]],fill=1)
        owner[(np.asarray(patch)>0)&(base==1)]=label
    for i,c in enumerate(countries,1):
        if owner[capitals[c["tag"]]]!=i:
            raise ValueError("Border adjustment moved a capital: "+c["tag"])
    print("2/7: states and connected provinces",flush=True)
    states=[]; seeds=[]
    for i,c in enumerate(countries,1):
        for j,point in enumerate(choose_seeds(owner==i,c["states"],capitals[c["tag"]])):
            sid=len(states)+1; states.append({"id":sid,"tag":c["tag"],"index":j,"seed":point}); seeds.append((sid,point))
    state_map=flood(owner,seeds,noise,bridges)
    weights=[int(np.count_nonzero(state_map==s["id"])) for s in states]
    minima=[max(len(components(state_map==s["id"])),config.get("province_minima",{}).get(s["tag"],1)) for s in states]
    counts=allocate(weights,config["land_provinces"],minima)
    province_seeds=[]; provinces=[]
    for state,count in zip(states,counts):
        for point in choose_seeds(state_map==state["id"],count,state["seed"],importance):
            pid=len(provinces)+1
            provinces.append({"id":pid,"state":state["id"],"tag":state["tag"],"point":point,"type":"land"})
            province_seeds.append((pid,point))
    province_map=flood(state_map,province_seeds,noise,diagonals=True)
    water_seeds=[]
    for point in choose_seeds(sea,config["sea_provinces"]):
        pid=len(provinces)+1; provinces.append({"id":pid,"state":None,"tag":None,"point":point,"type":"sea"}); water_seeds.append((pid,point))
    province_map+=flood(sea.astype(np.int32),water_seeds,noise,diagonals=True)
    for i in range(1,len(config["lakes"])+1):
        pid=len(provinces)+1; point=nearest(lakes==i,config["lakes"][i-1][:2])
        provinces.append({"id":pid,"state":None,"tag":None,"point":point,"type":"lake"}); province_map[lakes==i]=pid
    region_constraints=state_map.copy()
    region_constraints[sea]=-1
    region_constraints[lakes>0]=-2-lakes[lakes>0].astype(np.int32)
    repair_province_connectivity(province_map,region_constraints,provinces)
    graph=adjacency(province_map)
    for cross in crossings:
        cross["from"]=int(province_map[cross["a"]]); cross["to"]=int(province_map[cross["b"]]); cross["through"]=int(province_map[cross["center"]])
        if provinces[cross["through"]-1]["type"]!="sea": raise ValueError("Crossing through land")
        graph[cross["from"]].add(cross["to"]); graph[cross["to"]].add(cross["from"])
    for c in countries:
        c["capital_province"]=int(province_map[capitals[c["tag"]]])
        c["capital_state"]=provinces[c["capital_province"]-1]["state"]
    for p in provinces:
        p["coastal"]=(p["type"]=="land" and any(provinces[q-1]["type"]=="sea" for q in graph[p["id"]])) or (p["type"]=="sea" and any(provinces[q-1]["type"]=="land" for q in graph[p["id"]]))
        y,x=p["point"]; p["position"]=[x*scale+scale//2+.5,size-(y*scale+scale//2+.5)]
    print("3/7: bitmap and DDS layers",flush=True)
    full_labels=smooth_province_boundaries(province_map,provinces,scale,config.get("border_smoothing_radius",2.3))
    for _ in range(3):
        grow_small_provinces(full_labels,provinces,scale)
        fix_junctions(full_labels,provinces,scale)
        if np.bincount(full_labels.ravel())[1:].min()>=64:break
    for cross in crossings:
        cross["from"]=int(full_labels[cross["a"][0]*scale+scale//2,cross["a"][1]*scale+scale//2])
        cross["to"]=int(full_labels[cross["b"][0]*scale+scale//2,cross["b"][1]*scale+scale//2])
        cross["through"]=int(full_labels[cross["center"][0]*scale+scale//2,cross["center"][1]*scale+scale//2])
    # Junction repair can change a one-pixel land/sea contact. Engine metadata
    # and routes must use the final full-resolution bitmap, not the design grid.
    graph=adjacency(full_labels)
    for cross in crossings:
        graph[cross["from"]].add(cross["to"]); graph[cross["to"]].add(cross["from"])
    for p in provinces:
        p["coastal"]=(p["type"]=="land" and any(provinces[q-1]["type"]=="sea" for q in graph[p["id"]])) or (p["type"]=="sea" and any(provinces[q-1]["type"]=="land" for q in graph[p["id"]]))
    for p in provinces:
        y,x=p["point"];p["position"]=[x*scale+scale//2+.5,size-(y*scale+scale//2+.5)]
    towns=place_towns(config,provinces,height,graph)
    town_surface(towns,terrain,landscape["trees"],rgb,full_labels,scale)
    villages=place_villages(config,states,provinces,towns,height)
    village_surface(villages,landscape["trees"],rgb,full_labels,scale)
    colors=np.zeros((len(provinces)+1,3),dtype=np.uint8)
    for p in provinces:
        value=(p["id"]*104729+9187)&0xffffff; colors[p["id"]]=[value&255,(value>>8)&255,(value>>16)&255]; p["color"]=colors[p["id"]].tolist()
        p["terrain"]={0:"plains",5:"plains",4:"forest",6:"mountain",13:"urban",16:"mountain",17:"hills",20:"mountain"}[int(terrain[p["point"]])] if p["type"]=="land" else ("ocean" if p["type"]=="sea" else "lakes")
    save_image("map/provinces.bmp",Image.fromarray(colors[full_labels]))
    up=lambda a:np.repeat(np.repeat(a,scale,0),scale,1)
    height_image=Image.fromarray(np.uint8(np.clip(height,0,255))).resize((size,size),Image.Resampling.BILINEAR)
    save_image("map/heightmap.bmp",palette_image(np.asarray(height_image),game/"map/heightmap.bmp"))
    save_image("map/terrain.bmp",palette_image(up(terrain),game/"map/terrain.bmp"))
    river_array,river_visual=river_layers(config,up(land))
    save_image("map/rivers.bmp",palette_image(river_array,game/"map/rivers.bmp"))
    trees=landscape["trees"]
    tree_size=round(size*600/2048); tree_image=Image.fromarray(trees).resize((tree_size,tree_size),Image.Resampling.NEAREST)
    save_image("map/trees.bmp",palette_image(np.asarray(tree_image),game/"map/trees.bmp"))
    cities=Image.new("L",(size,size),0); pen=ImageDraw.Draw(cities)
    for c in countries:
        y,x=capitals[c["tag"]]; x*=scale; y*=scale;radius=config.get("capital_city_radius",7)
        block=Image.new("L",(size,size),0);ImageDraw.Draw(block).ellipse((x-radius,y-radius,x+radius,y+radius),fill=15)
        mask=np.asarray(block).copy();mask[full_labels!=c["capital_province"]]=0
        cities=Image.fromarray(np.maximum(np.asarray(cities),mask))
    for town in towns+villages:
        if not town.get("city_sprawl",True): continue
        y,x=town["point"]; x=x*scale+scale//2; y=y*scale+scale//2
        radius=town.get("radius",6)
        # Standard city-sprawl mask, clipped to the actual town province.
        block=Image.new("L",(size,size),0); urban=ImageDraw.Draw(block)
        urban.rectangle((x-radius,y-radius//2,x+radius,y+radius//2),fill=15)
        urban.rectangle((x-radius//2,y-radius,x+radius//2,y+radius),fill=15)
        mask=np.asarray(block).copy(); mask[full_labels!=town["province"]]=0
        cities=Image.fromarray(np.maximum(np.asarray(cities),mask))
    save_image("map/cities.bmp",palette_image(np.asarray(cities),game/"map/cities.bmp"))
    save_image("map/world_normal.bmp",Image.fromarray(world_normals(height_image)))
    # Resize RGB before attaching the zero city-emission alpha channel.
    # Pillow premultiplies RGBA during resize, erasing colours when all alpha is zero.
    colormap=Image.fromarray(rgb[:,:,:3]).resize((size,size),Image.Resampling.BILINEAR)
    # Keep tributaries readable at the same map scale as the reference sketch.
    colormap.paste((43,91,132),(0,0,size,size),Image.fromarray(river_visual))
    colormap=colormap.convert("RGBA")
    colormap.putalpha(0)
    save_image("map/terrain/colormap_rgb_cityemissivemask_a.dds",colormap,pixel_format="DXT5")
    for level in range(3):
        dim=size//2**(level+1)
        save_image(f"map/terrain/colormap_water_{level}.dds",Image.new("RGBA",(dim,dim),(43,91,132,255)),pixel_format="DXT5")
    print("4/7: map definitions and supply",flush=True)
    (ROOT/"build").mkdir(exist_ok=True)
    loc={"romashkova_valley":"Ромашковая долина","RDC_faction":"Ромашковая долина"}
    for town in towns: loc[f"VICTORY_POINTS_{town['province']}"]=town["name"]
    for state in states: state["towns"]=[town for town in towns if town["state"]==state["id"]]
    generate_map_data(config,game,countries,states,provinces,crossings,graph,province_map,full_labels,height,river_masks,loc)
    print("5/7: playable countries and armed forces",flush=True)
    generate_countries(countries,states,loc,game)
    print("6/7: scripted independence",flush=True)
    generate_scenario(countries,states,loc,game)
    for language in ("russian","english"):
        write(f"localisation/replace/romashkova_world_l_{language}.yml",f"l_{language}:\n"+"".join(f" {key}:0 "+json.dumps(value,ensure_ascii=False)+"\n" for key,value in loc.items()),bom=True)
    generate_compatibility(game,config)
    print("7/7: preview and manifests",flush=True)
    tag_ids={c["tag"]:i+1 for i,c in enumerate(countries)}
    pixel_owners=np.array([0]+[tag_ids.get(p["tag"],0) for p in provinces])[full_labels]
    rendered_owners=pixel_owners[::scale,::scale]
    political=np.zeros((n,n,3),dtype=np.uint8); political[:]=[44,106,160]
    for i,c in enumerate(countries,1): political[rendered_owners==i]=c["color"]
    political[lakes>0]=[79,147,193]
    preview=Image.fromarray(political).resize((1536,1536),Image.Resampling.NEAREST); draw=ImageDraw.Draw(preview)
    font_path=Path("C:/Windows/Fonts/arial.ttf"); font=ImageFont.truetype(str(font_path),19) if font_path.exists() else ImageFont.load_default()
    for c in countries:
        y,x=capitals[c["tag"]]; px=x*1536/n; py=y*1536/n
        draw.ellipse((px-5,py-5,px+5,py+5),fill="white",outline="black",width=2)
        draw.text((px+9,py-13),"Совет" if c["tag"]=="RDC" else c["name"],font=font,fill="white",stroke_width=2,stroke_fill="black")
    for cross in crossings:
        ay,ax=cross["a"]; by,bx=cross["b"]; draw.line((ax*1536/n,ay*1536/n,bx*1536/n,by*1536/n),fill=(255,222,148),width=4)
    save_image("build/political_preview.png",preview)
    save_image("build/terrain_preview.png",shaded_preview(height_image,colormap))
    detail_image=shaded_preview(height_image,colormap)
    detail=np.asarray(detail_image).copy()
    rendered=full_labels[::2,::2]; dry=rendered<=config["land_provinces"]
    borders=np.zeros(rendered.shape,dtype=bool)
    borders[1:,:]|=(rendered[1:,:]!=rendered[:-1,:])&dry[1:,:]&dry[:-1,:]
    borders[:,1:]|=(rendered[:,1:]!=rendered[:,:-1])&dry[:,1:]&dry[:,:-1]
    detail[borders]=np.uint8(detail[borders]*.65)
    detail_image=Image.fromarray(detail); draw=ImageDraw.Draw(detail_image)
    for c in countries:
        y,x=capitals[c["tag"]]; px=x*detail_image.width/n; py=y*detail_image.height/n
        draw.ellipse((px-4,py-4,px+4,py+4),fill="white",outline="black",width=2)
        draw.text((px+8,py-13),c["name"],font=font,fill="white",stroke_width=2,stroke_fill="black")
    save_image("build/map_detail_preview.png",detail_image)
    from settlement_previews import render_settlement_previews
    for name,preview in render_settlement_previews(shaded_preview(height_image,colormap),config,towns,countries,font_path).items():
        save_image(name,preview)
    write("build/settlements.json",json.dumps(towns,ensure_ascii=False,indent=2))
    table="# Места Ромашковой долины\n\n"+config["town_reference_note"]+"\n\n| Место | Страна | VP | Роль | Отсылка и примечание |\n|---|---|---|---|---|\n"
    for town in towns:
        reference="["+town["episode"]+"]("+town["source"]+")" if town.get("episode") else "Авторское название"
        note="; "+town["continuity"] if town.get("continuity") else ""
        note+="; "+town["reference_kind"] if town.get("reference_kind") else ""
        table+=f'| {town["name"]} | {town["tag"]} | {town["value"]} | {town.get("role", "поселение")} | {reference}{note} |\n'
    write("build/settlements.md",table)

    feature_report={
        "geometry_version":config["geometry_version"],
        "ridge_chains":len(config["relief"]["ridges"]),
        "passes":len(config["relief"]["passes"]),
        "ordinary_rivers":len(config["tributaries"]),
        "forest_land_percent":round(float(np.mean(forest[land])*100),1),
        "tree_coverage_land_percent":round(float(np.mean((trees>0)[land])*100),1),
        "farmland_land_percent":round(float(np.mean((landscape["farmland"]>.35)[land])*100),1),
        "hill_land_percent":round(float(np.mean((terrain==17)[land])*100),1),
        "mountain_land_percent":round(float(np.mean(np.isin(terrain,[6,16,20])[land])*100),1),
        "height_percentiles":np.percentile(height[land],[5,25,50,75,95,100]).round(1).tolist(),
        "nato_land_provinces":sum(p["tag"]=="NAT" for p in provinces),
        "ambient_villages":len(villages),
        "city_sprawl_pixels":int(np.count_nonzero(np.asarray(cities))),
        "label_visibility":config["label_visibility"]
    }
    write("build/natural_features.json",json.dumps(feature_report,ensure_ascii=False,indent=2))
    manifest={"seed":config["seed"],"geometry_version":config["geometry_version"],"size":size,"grid_size":n,"countries":countries,"states":states,"provinces":provinces,"crossings":crossings,"towns":towns,"villages":villages}
    write("build/world_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    np.savez_compressed(ROOT/"build/design_grid.npz",provinces=province_map,owners=owner,states=state_map)
    write("build/generated_files.json",json.dumps(sorted(set(GENERATED)),indent=2))
    print(f"Done: {len(provinces)} provinces, {len(states)} states, {len(countries)} countries",flush=True)

def generate_map_data(config,game,countries,states,provinces,crossings,graph,grid,full_labels,height,river_masks,loc):
    n=config["grid_size"]; size=config["size"]; scale=size//n
    write("map/default.map",'definitions = "definition.csv"\nprovinces = "provinces.bmp"\npositions = "positions.txt"\nterrain = "terrain.bmp"\nrivers = "rivers.bmp"\nheightmap = "heightmap.bmp"\ntree_definition = "trees.bmp"\ncontinent = "continent.txt"\nadjacency_rules = "adjacency_rules.txt"\nadjacencies = "adjacencies.csv"\nambient_object = "ambient_object.txt"\nseasons = "seasons.txt"\ntree = { 5 6 28 29 }\n')
    write("map/continent.txt",(game/"map/continent.txt").read_text(encoding="utf-8-sig"))
    for file in ("positions.txt","adjacency_rules.txt","ambient_object.txt"):
        write("map/"+file,"# Custom world; no vanilla province references.\n")
    write("map/seasons.txt",(game/"map/seasons.txt").read_text(encoding="utf-8-sig"))
    write("map/supplyareas/00_romashkova.txt","# Supply is provided by railways and hubs.\n")
    shutil.copyfile(game/"map/cities.txt",ROOT/"map/cities.txt"); GENERATED.append("map/cities.txt")
    rows=["0;0;0;0;land;false;unknown;0"]
    for p in provinces:
        rows.append(";".join(map(str,[p["id"],*p["color"],p["type"],str(p["coastal"]).lower(),p["terrain"],1 if p["type"]=="land" else 0])))
    write("map/definition.csv","\n".join(rows)+"\n")
    rows=["From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment"]
    for c in crossings:
        ay,ax=c["a"]; by,bx=c["b"]
        rows.append(f'{c["from"]};{c["to"]};sea;{c["through"]};{ax*scale+scale//2};{size-1-(ay*scale+scale//2)};{bx*scale+scale//2};{size-1-(by*scale+scale//2)};;{c["name"]}')
    rows.append("-1;-1;;-1;-1;-1;-1;-1;;"); write("map/adjacencies.csv","\n".join(rows)+"\n")
    regions=[{"id":i,"name":"Земли: "+c["name"],"provinces":[p["id"] for p in provinces if p["tag"]==c["tag"]]} for i,c in enumerate(countries,1)]
    for j,name in enumerate(["Западный океан","Северный океан","Восточный океан","Южный океан","Ромашковое море","Западная судоходная река","Восточная судоходная река"]):
        regions.append({"id":12+j,"name":name,"provinces":[]})
    # Connected sea regions: geodesic Voronoi on the sea-province graph.
    water={p["id"] for p in provinces if p["type"]=="sea"}
    anchors=[(.04,.35),(.50,.02),(.985,.45),(.43,.98),(.78,.71),(.085,.39),(.605,.15)]
    assigned={}; heap=[]
    for rid,anchor in enumerate(anchors,12):
        goal=np.array([anchor[1]*n,anchor[0]*n])
        pid=min(water,key=lambda q:sum((np.array(provinces[q-1]["point"])-goal)**2))
        heapq.heappush(heap,(0.,rid,pid))
    while heap:
        distance,rid,pid=heapq.heappop(heap)
        if pid in assigned: continue
        assigned[pid]=rid
        for q in sorted(graph[pid]&water):
            if q not in assigned:
                step=max(1.,math.dist(provinces[pid-1]["point"],provinces[q-1]["point"]))
                heapq.heappush(heap,(distance+step,rid,q))
    for p in provinces:
        if p["type"]=="sea": regions[assigned[p["id"]]-1]["provinces"].append(p["id"])
        elif p["type"]=="lake":
            c=min(countries,key=lambda c:math.dist(p["point"],(c["anchor"][1]*n,c["anchor"][0]*n)))
            rid=next(i for i,country in enumerate(countries) if c==country)
            regions[rid]["provinces"].append(p["id"])
    weather=[]
    for region in regions:
        rid=region["id"]; loc[f"SMESH_REGION_{rid}"]=region["name"]
        text=f'strategic_region = {{ id = {rid} name = "SMESH_REGION_{rid}" provinces = {{ '+" ".join(map(str,region["provinces"]))+" }\n"
        if rid>11: text+=" naval_terrain = ocean\n"
        text+=" weather = {\n"
        for month in range(12):
            winter=month in (0,1,11)
            text+=f' period = {{ between = {{ 0.{month} {[30,27,30,29,30,29,30,30,29,30,29,30][month]}.{month} }} temperature = {{ {"-6.0 8.0" if winter else "8.0 24.0"} }} no_phenomenon = 0.8 rain_light = 0.2 rain_heavy = 0.05 snow = {0.1 if winter else 0.0} blizzard = 0.0 arctic_water = 0.0 mud = 0.05 sandstorm = 0.0 min_snow_level = 0.0 }}\n'
        text+=" } }\n"; write(f"map/strategicregions/{rid}-Romashkova.txt",text)
        if region["provinces"]:
            p=provinces[region["provinces"][0]-1]; x,z=p["position"]; weather.append(f"{rid};{x};12.0;{z};small")
    write("map/weatherpositions.txt","\n".join(weather)+"\n")
    by_tag={c["tag"]:c for c in countries}; railways=set(); nodes=set(); buildings=[]
    allocations={}
    for c in countries:
        national=[s for s in states if s["tag"]==c["tag"]]
        for kind,key in (("civilian","civilian"),("military","military")):
            distribution={s["id"]:c[key]//c["states"]+int(s["index"]<c[key]%c["states"]) for s in national}
            # Move an existing factory to each named industrial district; national totals stay fixed.
            targets={s["id"] for s in national if any(t.get("kind")=="industry" for t in s.get("towns",[]))}
            for sid in sorted(targets):
                if distribution[sid]: continue
                donors=[other for other,amount in distribution.items() if amount and (other not in targets or amount>1)]
                if donors:
                    donor=max(donors,key=lambda other:(distribution[other],-other))
                    distribution[donor]-=1;distribution[sid]+=1
            for sid,amount in distribution.items(): allocations.setdefault(sid,{})[kind]=amount
    for state in states:
        c=by_tag[state["tag"]]; sid=state["id"]
        ps=[p for p in provinces if p["state"]==sid]
        state["provinces"]=[p["id"] for p in ps]
        transport=next((t for t in state.get("towns",[]) if t.get("kind")=="transport"),None)
        state["hub"]=transport["province"] if transport else int(grid[state["seed"]]); nodes.add(state["hub"])
        harbour=next((t for t in state.get("towns",[]) if t.get("port_level")),None)
        state["port"]=harbour["province"] if harbour else next((p["id"] for p in ps if p["coastal"]),None)
        state["port_level"]=harbour["port_level"] if harbour else 3 if sid==c["capital_state"] else 1
        if state["port"]:
            state["port_sea"]=next(q for q in sorted(graph[state["port"]]) if provinces[q-1]["type"]=="sea")
        allowed={p["id"] for p in provinces if p["tag"]==c["tag"]}
        route=shortest(graph,c["capital_province"],state["hub"],allowed)
        if route and len(route)>1: railways.add(tuple(route))
        if not route and not state["port"]: raise ValueError(f"Isolated state {sid} has no port")
        if state["port"]:
            route=shortest(graph,state["hub"],state["port"],set(state["provinces"]))
            if route and len(route)>1: railways.add(tuple(route))
        civ=allocations[sid]["civilian"];mil=allocations[sid]["military"]
        dock=0
        if c["dockyards"] and state["port"] and not c.get("dockyard_assigned"):
            dock=c["dockyards"]; c["dockyard_assigned"]=True
        state.update(civilian=civ,military=mil,dockyards=dock)
        capital=sid==c["capital_state"]
        text=f'state = {{\n id = {sid}\n name = "SMESH_STATE_{sid}"\n manpower = {c["manpower"]//c["states"]}\n state_category = {"metropolis" if c["tag"]=="NAT" or dock else "large_city"}\n'
        text+=f' resources = {{ steel = {8 if c["tag"]=="NAT" else 2} oil = {6 if c["tag"]=="NAT" else 1} aluminium = {4 if c["tag"]=="NAT" else 1} rubber = {4 if c["tag"]=="NAT" else 1} tungsten = {2 if c["tag"]=="NAT" else 1} chromium = 1 }}\n history = {{ owner = {c["tag"]} add_core_of = {c["tag"]}\n'
        if capital: text+=f' victory_points = {{ {c["capital_province"]} {config.get("capital_victory_points",20)} }}\n'
        for town in state.get("towns",[]):
            text+=f' victory_points = {{ {town["province"]} {town["value"]} }}\n'
        text+=f' buildings = {{ infrastructure = 3 industrial_complex = {civ} arms_factory = {mil} dockyard = {dock} air_base = {2 if capital else 1}\n'
        province_buildings={state["hub"]:["supply_node = 1"]}
        if state["port"]: province_buildings.setdefault(state["port"],[]).append(f'naval_base = {state["port_level"]}')
        for town in state.get("towns",[]):
            if town.get("fort"): province_buildings.setdefault(town["province"],[]).append(f'bunker = {town["fort"]}')
        for pid,entries in province_buildings.items(): text+=f' {pid} = {{ '+ " ".join(entries)+" }\n"
        text+=" } }\n provinces = { "+" ".join(map(str,state["provinces"]))+" }\n local_supplies = 0.0\n}\n"
        write(f"history/states/{sid}-Romashkova.txt",text)
        loc[f"SMESH_STATE_{sid}"]=c["capital"] if capital else f'{c["name"]}: область {state["index"]+1}'
        loc[f'VICTORY_POINTS_{c["capital_province"]}']=c["capital"]
        hub=provinces[state["hub"]-1]; x,z=hub["position"]; y=float(height[hub["point"]])/10
        # Construction positions exist for every state even if no factory is built yet.
        industry=next((t for t in state.get("towns",[]) if t.get("kind")=="industry"),None)
        for kind,total in (("industrial_complex",max(civ,1)),("arms_factory",max(mil,1)),("air_base",1),("supply_node",1),("rocket_site",1),("mega_gun_emplacement",1),("dockyard",dock)):
            spot=provinces[industry["province"]-1] if industry and kind in ("industrial_complex","arms_factory") else hub
            bx,bz=spot["position"];by=float(height[spot["point"]])/10
            for j in range(total): buildings.append(f"{sid};{kind};{bx:.2f};{by:.2f};{bz:.2f};0.00;0")
        for town in state.get("towns",[]):
            if town.get("fort"):
                spot=provinces[town["province"]-1];bx,bz=spot["position"];by=float(height[spot["point"]])/10
                buildings.append(f"{sid};bunker;{bx:.2f};{by:.2f};{bz:.2f};0.00;0")
    # Every coastal province requires a naval spawn, including unbuilt ports.
    for p in provinces:
        if p["type"]!="land" or not p["coastal"]: continue
        sea_id=next(q for q in sorted(graph[p["id"]]) if provinces[q-1]["type"]=="sea")
        mask=grid==p["id"]; wet=grid==sea_id
        adjacent=mask&(np.roll(wet,1,0)|np.roll(wet,-1,0)|np.roll(wet,1,1)|np.roll(wet,-1,1))
        if np.any(adjacent):
            cy,cx=np.column_stack(np.nonzero(adjacent))[0]
            px=int(cx)*scale+scale//2; pz=size-(int(cy)*scale+scale//2)
        else:
            mask=full_labels==p["id"]; wet=full_labels==sea_id
            adjacent=mask&(np.roll(wet,1,0)|np.roll(wet,-1,0)|np.roll(wet,1,1)|np.roll(wet,-1,1))
            cy,cx=np.column_stack(np.nonzero(adjacent))[0]
            px=int(cx)+.5; pz=size-int(cy)-.5
        # The engine floors bottom-left coordinates; avoid pixel-edge ambiguity.
        if int(full_labels[size-1-math.floor(pz),math.floor(px)])!=p["id"]:
            mask=full_labels==p["id"]; wet=full_labels==sea_id
            adjacent=mask&(np.roll(wet,1,0)|np.roll(wet,-1,0)|np.roll(wet,1,1)|np.roll(wet,-1,1))
            cy,cx=np.column_stack(np.nonzero(adjacent))[0]
            px=int(cx)+.5; pz=size-int(cy)-.5
        buildings.append(f'{p["state"]};naval_base_spawn;{px};9.7;{pz};0.0;{sea_id}')
    # A trailing blank CSV record is rejected by the engine.
    write("map/buildings.txt","\n".join(buildings))
    write("map/supply_nodes.txt","".join(f"1 {pid}\n" for pid in sorted(nodes)))
    write("map/railways.txt","".join(f"2 {len(route)} "+" ".join(map(str,route))+"\n" for route in sorted(railways)))
    write("map/unitstacks.txt","".join(f'{p["id"]};0;{p["position"][0]};{float(height[p["point"]])/10:.2f};{p["position"][1]};0.0;1.0\n' for p in provinces))
    write("build/strategic_regions.json",json.dumps(regions,ensure_ascii=False,indent=2))

def generate_countries(countries,states,loc,game):
    write("common/country_tags/00_romashkova.txt","".join(f'{c["tag"]} = "countries/{c["tag"]}.txt"\n' for c in countries))
    write("common/country_tags/zz_dynamic_countries.txt",(game/"common/country_tags/zz_dynamic_countries.txt").read_text(encoding="utf-8-sig"))
    write("country_metadata/00_romashkova.txt",'default = { playthrough_stats_background = "GFX_playthrough_stats_bg_default" career_profile_background = "GFX_career_profile_bg_default" }\n')
    write("common/country_tag_aliases/00_romashkova.txt","# No vanilla aliases.\n")
    base_colors=(game/"common/countries/colors.txt").read_text(encoding="utf-8-sig")
    custom_colors="".join(c["tag"]+" = { color = rgb { "+" ".join(map(str,c["color"]))+" } color_ui = rgb { "+" ".join(map(str,c["color"]))+" } }\n" for c in countries)
    write("common/countries/colors.txt",base_colors+"\n"+custom_colors)
    # Native dormant-country flags are still loaded into the engine atlas.
    for sub in ("","medium/","small/"):
        for name in ("SCO.tga","SCO_neutrality.tga","CSA_neutrality.tga"):
            source=game/"gfx/flags"/sub/name
            if source.exists(): save_image("gfx/flags/"+sub+name,Image.open(source).convert("RGBA"))
    chars=["characters = {"]
    bookmark=['bookmarks = { bookmark = { name = "ROMASHKOVA_BOOKMARK" desc = "ROMASHKOVA_BOOKMARK_DESC" date = 1936.1.1.12 picture = "GFX_select_date_1936" default_country = "RDC" default = yes']
    loc.update(ROMASHKOVA_BOOKMARK="Ромашковая долина",ROMASHKOVA_BOOKMARK_DESC="Совет управляет девятью землями жителей долины. На острове в юго-восточном заливе располагается Миссия НАТО. Через семь месяцев кризис общего управления приведёт к независимости государств.")
    for c in countries:
        tag=c["tag"]; democratic=tag=="NAT"
        for suffix in ("","_DEF","_ADJ"): loc[tag+suffix]=c["name"]
        for ideology in ("neutrality","democratic","communism","fascism"):
            for suffix in ("","_DEF","_ADJ"): loc[tag+"_"+ideology+suffix]=c["name"]
        loc[tag+"_leader"]=c["leader"]
        loc[tag+"_history"]="Независимая военная миссия на острове в юго-восточном заливе долины." if democratic else ("Центральное управление долины. Стартовая ветка фокусов приведёт к независимости девяти стран." if tag=="RDC" else "Государство жителя долины под управлением Совета. После распада оно самостоятельно определяет свою политику.")
        write(f"common/countries/{tag}.txt",'graphical_culture = western_european_gfx\ngraphical_culture_2d = western_european_2d\ncolor = { '+" ".join(map(str,c["color"]))+" }\n")
        chars.append(f' {tag}_leader = {{ name = {tag}_leader portraits = {{ civilian = {{ large = "gfx/leaders/Europe/Portrait_Europe_Generic_land_1.dds" }} }} country_leader = {{ ideology = {"liberalism" if democratic else "despotism"} expire = "1965.1.1" }} }}')
        tech="infantry_weapons = 1 infantry_weapons1 = 1 tech_support = 1 tech_engineers = 1 tech_trucks = 1 basic_train = 1 gw_artillery = 1 synth_oil_experiments = 1"
        history=f'capital = {c["capital_state"]}\noob = "{tag}_1936"\nset_research_slots = {3 if tag in ("RDC","NAT","LOS") else 2}\nset_stability = 0.65\nset_war_support = 0.35\nset_politics = {{ ruling_party = {"democratic" if democratic else "neutrality"} elections_allowed = no }}\nset_popularities = {{ democratic = {80 if democratic else 10} neutrality = {10 if democratic else 80} communism = 5 fascism = 5 }}\nrecruit_character = {tag}_leader\nadd_ideas = {{ extensive_conscription civilian_economy export_focus }}\nset_technology = {{ {tech} }}\n'
        history+='if = { limit = { has_dlc = "Man the Guns" } set_technology = { early_ship_hull_light = 1 basic_battery = 1 basic_torpedo = 1 mtg_transport = 1 } else = { set_technology = { early_destroyer = 1 transport = 1 } } }\n'
        for eq,amount in (("infantry_equipment_1",c["divisions"]*600),("support_equipment_1",c["divisions"]*30),("train_equipment_1",100),("motorized_equipment_1",200),("convoy_1",100)):
            history+=f"add_equipment_to_stockpile = {{ type = {eq} amount = {amount} producer = {tag} }}\n"
        cs=[s for s in states if s["tag"]==tag]; port=next((s["port"] for s in cs if s["port"]),None); c["naval_base"]=port
        if tag in ("RDC","NAT"):
            if port is None: raise ValueError(tag+" needs a port")
            history+=f'if = {{ limit = {{ has_dlc = "Man the Guns" }} create_equipment_variant = {{ name = "Romashkova Class" type = ship_hull_light_1 parent_version = 0 modules = {{ fixed_ship_battery_slot = ship_light_battery_1 fixed_ship_anti_air_slot = empty fixed_ship_fire_control_system_slot = empty fixed_ship_radar_slot = empty fixed_ship_engine_slot = light_ship_engine_1 fixed_ship_torpedo_slot = ship_torpedo_1 mid_1_custom_slot = empty rear_1_custom_slot = empty }} }} set_naval_oob = "{tag}_naval_mtg" else = {{ set_naval_oob = "{tag}_naval_legacy" }} }}\n'
            for modern in (False,True):
                eq="ship_hull_light_1" if modern else "destroyer_1"
                ships="\n".join(f' ship = {{ name = "{tag} {j+1}" definition = destroyer equipment = {{ {eq} = {{ amount = 1 owner = {tag} '+('version_name = "Romashkova Class"' if modern else '')+" } } }" for j in range(4))
                write(f'history/units/{tag}_naval_{"mtg" if modern else "legacy"}.txt',f'units = {{ fleet = {{ name = "{c["name"]}" naval_base = {port} task_force = {{ name = "Речная флотилия" location = {port}\n{ships}\n }} }} }}\n')
        write(f"history/countries/{tag}-Romashkova.txt",history)
        regiments=" ".join(f"infantry = {{ x = {i//3} y = {i%3} }}" for i in range(6))
        oob=f'division_template = {{ name = "Пехотная дивизия" regiments = {{ {regiments} }} support = {{ engineer = {{ x = 0 y = 0 }} }} }}\nunits = {{\n'
        hubs=[s["hub"] for s in cs]
        for j in range(c["divisions"]):
            oob+=f' division = {{ name = "{j+1}-я дивизия" location = {hubs[j%len(hubs)]} division_template = "Пехотная дивизия" start_experience_factor = 0.2 start_equipment_factor = 1.0 }}\n'
        oob+='}\ninstant_effect = { add_equipment_production = { equipment = { type = infantry_equipment_1 creator = '+tag+" } requested_factories = 1 progress = 0.5 efficiency = 50 } }\n"
        write(f"history/units/{tag}_1936.txt",oob)
        bookmark.append(f' "{tag}" = {{ history = "{tag}_history" ideology = {"democratic" if democratic else "neutrality"} }}')
        flag=Image.new("RGBA",(82,52),tuple(c["color"])); pen=ImageDraw.Draw(flag)
        pen.rectangle((0,0,81,7),fill=(240,229,191)); pen.rectangle((0,44,81,51),fill=(38,63,79))
        if tag=="NAT":
            pen.line((41,12,41,40),fill="white",width=2); pen.line((27,26,55,26),fill="white",width=2)
            pen.polygon([(41,12),(45,23),(55,26),(45,29),(41,40),(37,29),(27,26),(37,23)],outline="white")
        else:
            for a in range(6):
                angle=a*math.pi/3; cx=41+14*math.cos(angle); cy=26+14*math.sin(angle)
                pen.ellipse((cx-5,cy-5,cx+5,cy+5),fill=(245,240,224))
            pen.ellipse((34,19,48,33),fill=(240,229,191))
        for sub,dim in (("",(82,52)),("medium/",(41,26)),("small/",(10,7))):
            for suffix in ("","_neutrality","_democratic","_communism","_fascism"):
                save_image(f"gfx/flags/{sub}{tag}{suffix}.tga",flag.resize(dim,Image.Resampling.LANCZOS))
    chars.append("}"); write("common/characters/00_romashkova.txt","\n".join(chars)+"\n")
    bookmark.append("} }"); write("common/bookmarks/00_romashkova.txt","\n".join(bookmark)+"\n")

def generate_scenario(countries,states,loc,game):
    from krosh_prologue import generate_prologue
    generate_prologue(countries,states,loc,game,write)


def regenerate_scenario(game):
    """Rebuild authored content using the accepted map, without regenerating it."""
    manifest=json.loads((ROOT/"build/world_manifest.json").read_text(encoding="utf-8"))
    loc={}
    generate_scenario(manifest["countries"],manifest["states"],loc,game)
    for language in ("russian","english"):
        relative=f"localisation/replace/romashkova_world_l_{language}.yml"
        content=(ROOT/relative).read_text(encoding="utf-8-sig")
        existing={m.group(1):json.loads(m.group(2)) for m in re.finditer(r'^\s+(\S+):\d+\s+(".*")\s*$',content,re.MULTILINE)}
        existing.update(loc)
        write(relative,"l_"+language+":\n"+"\n".join(" "+key+":0 "+json.dumps(value,ensure_ascii=False) for key,value in existing.items())+"\n",bom=True)
    previous=json.loads((ROOT/"build/generated_files.json").read_text(encoding="utf-8"))
    write("build/generated_files.json",json.dumps(sorted(set(previous+GENERATED)),indent=2))


def generate_compatibility(game,config):
    """Keep generic engine data, neutralising hooks into the historical campaign.
    Vanilla tags remain dormant to satisfy engine-wide tag references. They have
    no owned states, armies, bookmarks, focus trees or historical events.
    """
    from compatibility import prepare_compatibility
    for relative,content in prepare_compatibility(game,ROOT).items():
        write(relative,content)
    write("common/national_focus/00_titlebar_styles.txt",(game/"common/national_focus/00_titlebar_styles.txt").read_text(encoding="utf-8-sig"))
    write("common/defines/00_romashkova.lua",'NDefines.NGame.HANDS_OFF_START_TAG = "RDC"\n')
    visibility=config["label_visibility"]
    cutoffs=", ".join(map(str,visibility["text_cutoffs"]))
    graphics=f"NDefines_Graphics.NGraphics.VICTORY_POINT_MAP_ICON_TEXT_CUTOFF = {{{cutoffs}}}\n"
    for field,key in (("TEXT_CUTOFF_MIN","text_min"),("TEXT_CUTOFF_MAX","text_max"),
                      ("DOT_CUTOFF_MIN","dot_min"),("DOT_CUTOFF_MAX","dot_max"),
                      ("CAPITAL_CUTOFF_MAX","capital_max"),("MAX_VICTORY_POINTS_FOR_PERCENT","max_vp_for_percent")):
        graphics+=f"NDefines_Graphics.NGraphics.VICTORY_POINT_MAP_ICON_{field} = {visibility[key]}.0\n"
    graphics+="\n".join([
        "NDefines_Graphics.NGraphics.VICTORY_POINT_MAP_ICON_AFTER = {0, 2, 9}",
        f"NDefines_Graphics.NGraphics.PROVINCE_NAME_DRAW_DISTANCE = {visibility['capital_max']}.0",
        f"NDefines_Graphics.NGraphics.CAPITAL_ICON_CUTOFF = {visibility['capital_max']}.0",
        "NDefines_Graphics.NGraphics.CITY_SPRAWL_SHRINK_DISTANCE = 1200.0",
        "NDefines_Graphics.NGraphics.MAP_BUILDINGS_SHRINK_DISTANCE = 700.0",
        "NDefines_Graphics.NGraphics.DRAW_MAP_OBJECTS_CUTOFF = 4000.0",
        "NDefines_Graphics.NGraphics.TREE_FADE_NEAR = 450.0",
        "NDefines_Graphics.NGraphics.TREE_FADE_FAR = 850.0"
    ])+"\n"
    native_graphics=(game/"common/defines/00_graphics.lua").read_text(encoding="utf-8-sig")
    for key in re.findall(r"^\s*(\w+_PRIORITY_VICTORY_POINTS)\s*=",native_graphics,re.MULTILINE):
        graphics+=f"NDefines_Graphics.NMapIcons.{key} = 12\n"
    for key,native_priority in re.findall(r"^\s*(\w+_PRIORITY_CAPITAL)\s*=\s*(\d+)",native_graphics,re.MULTILINE):
        graphics+=f"NDefines_Graphics.NMapIcons.{key} = {max(13,int(native_priority))}\n"
    write("common/defines/zz_romashkova_graphics.lua",graphics)
    gui=(game/"interface/mapicons.gui").read_text(encoding="utf-8-sig")
    for name in ("victory_point_mapicon","capital_mapicon"):
        begin=gui.index('name = "'+name+'"');end=gui.find('\n\t####',begin)
        if end<0:end=len(gui)
        block=gui[begin:end].replace('width=160 height=30','width=320 height=40').replace('maxWidth = 149','maxWidth = 300').replace('maxHeight = 30','maxHeight = 40')
        gui=gui[:begin]+block+gui[end:]
    write("interface/mapicons.gui",gui)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game",type=Path,default=Path("C:/Program Files (x86)/Steam/steamapps/common/Hearts of Iron IV"))
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("world.json"))
    parser.add_argument("--scenario-only",action="store_true",help="Regenerate the story on the accepted map without rebuilding geometry")
    args=parser.parse_args()
    if not (args.game/"map/provinces.bmp").exists(): parser.error("--game must point to installed HOI4")
    if args.scenario_only:
        regenerate_scenario(args.game)
    else:
        generate(json.loads(args.config.read_text(encoding="utf-8-sig")),args.game)
if __name__=="__main__": main()

