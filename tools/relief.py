"""Deterministic landforms, vegetation and HOI4-compatible world normals."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter

def smoothstep(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)

def noise_field(rng,n,cells):
    sample=rng.uniform(-1,1,(cells,cells)).astype(np.float32)
    return np.asarray(Image.fromarray(sample).resize((n,n),Image.Resampling.BICUBIC))

def distance_from_water(land,limit=48):
    distance=np.where(land,limit,0).astype(np.float32)
    for _ in range(limit):
        distance=np.minimum(distance,1+np.minimum.reduce([
            np.roll(distance,1,0),np.roll(distance,-1,0),
            np.roll(distance,1,1),np.roll(distance,-1,1)]))
    return distance

def sampled_path(points,values,steps=6):
    points=np.asarray(points,dtype=np.float32)
    padded=np.vstack([points[0],points,points[-1]])
    path=[]; positions=[]
    for i in range(len(points)-1):
        a,b,c,d=padded[i:i+4]
        for t in np.linspace(0,1,steps,endpoint=False):
            path.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
            positions.append(i+t)
    path.append(points[-1]); positions.append(len(points)-1)
    knots=np.linspace(0,len(points)-1,len(values))
    return np.asarray(path),np.interp(positions,knots,values)

def path_field(n,points,width,values=(1,1),warp=None):
    """Closest path influence, evaluated only in small segment bounding boxes."""
    points,values=sampled_path(points,values)
    influence=np.zeros((n,n),dtype=np.float32)
    along=np.zeros_like(influence)
    padding=width*3.2+.005
    for (ax,ay),(bx,by),va,vb in zip(points,points[1:],values,values[1:]):
        x0=max(0,int((min(ax,bx)-padding)*n)); x1=min(n,int((max(ax,bx)+padding)*n)+1)
        y0=max(0,int((min(ay,by)-padding)*n)); y1=min(n,int((max(ay,by)+padding)*n)+1)
        if x0>=x1 or y0>=y1: continue
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        xx=(xx+.5)/n; yy=(yy+.5)/n
        if warp is not None:
            xx+=warp[0][y0:y1,x0:x1]; yy+=warp[1][y0:y1,x0:x1]
        dx,dy=bx-ax,by-ay
        t=np.clip(((xx-ax)*dx+(yy-ay)*dy)/max(dx*dx+dy*dy,1e-10),0,1)
        squared=(xx-ax-t*dx)**2+(yy-ay-t*dy)**2
        weight=np.exp(-squared/(width*width))
        selected=weight>influence[y0:y1,x0:x1]
        influence[y0:y1,x0:x1][selected]=weight[selected]
        along[y0:y1,x0:x1][selected]=(va+(vb-va)*t)[selected]
    return influence,along

def oval(xx,yy,center,radii):
    cx,cy=center; rx,ry=radii
    return np.exp(-(((xx-cx)/rx)**2+((yy-cy)/ry)**2))

def terrain_layers(config,land,sea,lakes,capitals):
    n=land.shape[0]; yy,xx=np.mgrid[0:n,0:n].astype(np.float32)/n
    profile=config["relief"]
    rng=np.random.default_rng(config["seed"]+730)
    broad=noise_field(rng,n,12); medium=noise_field(rng,n,42)
    fine=noise_field(rng,n,116); detail=noise_field(rng,n,240)
    micro=noise_field(rng,n,384)
    ridges=np.zeros_like(broad); foothills=np.zeros_like(broad)
    warp=(.0024*medium,.0024*broad)
    for chain in profile["ridges"]:
        envelope,crest=path_field(n,chain["points"],chain["width"],chain["heights"],warp)
        ridges=np.maximum(ridges,envelope*crest)
        foothills=np.maximum(foothills,np.sqrt(envelope)*np.minimum(crest*.28,28))
    for saddle in profile["passes"]:
        ridges-=saddle["depth"]*oval(xx,yy,saddle["center"],saddle["radii"])
    ridges=np.maximum(ridges,0)*np.clip(.90+.10*medium+.035*fine,.72,1.08)
    plateau=profile["northwest_plateau"]
    plateau=oval(xx,yy,plateau[:2],plateau[2:])
    rolling=5.5*broad+3.5*medium+1.3*fine+.65*micro
    low_hills=np.zeros_like(broad)
    for belt in profile["hills"]:
        envelope=oval(xx+.006*medium,yy+.005*broad,belt["center"],belt["radii"])
        low_hills+=belt["height"]*.65*envelope*(.82+.18*medium)
    height=111+rolling+low_hills+24*plateau+ridges+foothills

    # Tributaries carve continuous valleys; channel floors descend downstream.
    valley=np.zeros_like(broad); riparian=np.zeros_like(broad)
    for points,spec in zip(config["tributaries"],profile["river_valleys"]):
        envelope,floor=path_field(n,points,spec["width"],spec["heights"])
        weight=envelope*.94
        height=height*(1-weight)+floor*weight
        valley=np.maximum(valley,envelope)
        riparian=np.maximum(riparian,envelope*(1-smoothstep((height-148)/30)))

    farm=profile["farm_region"]
    farmland=oval(xx+.002*medium,yy+.001*broad,farm["center"],farm["radii"])
    farmland*=1-smoothstep((ridges-8)/20)
    # The farm and the central basin remain usable lowlands.
    center=oval(xx,yy,[.475,.410],[.200,.200])
    height=height*(1-.38*center)+104*.38*center
    height=height*(1-.65*farmland)+110*.65*farmland
    distance=distance_from_water(land,64)
    # Cliffs meet the water sharply; floodplains have broad, low slopes.
    cliff=(ridges>46)&((xx>.88)|(yy<.15))
    shore_width=np.where(cliff,5.0,17+5*medium+5*valley)
    shore=smoothstep((distance-1)/shore_width)
    height=97+(height-97)*shore
    height=np.where(land,np.clip(height,97,246),90).astype(np.float32)
    coastal=land&(distance==1)

    west=smoothstep((.365-xx+.055*broad+.024*medium)/.19)
    south=smoothstep((yy-.650+.047*broad+.030*medium)/.20)
    north=smoothstep((.315-yy+.040*broad)/.18)
    east=smoothstep((xx-.700+.040*broad+.020*medium)/.20)
    density=np.maximum.reduce([.89*west,.94*south,.83*north,.82*east])
    density+=.17*medium+.10*fine+.05*detail
    density=np.maximum(density,.54*smoothstep((medium+.36*broad-.06)/.67))
    density*=1-.62*plateau
    clearings=smoothstep((fine+.32*medium-.39)/.47)
    density*=1-.47*clearings
    for clearing in profile["woodland_clearings"]:
        density*=1-clearing["strength"]*oval(xx,yy,clearing["center"],clearing["radii"])
    density*=1-.93*farmland
    density*=1-.37*valley
    density=np.maximum(density,.35*riparian)
    density*=1-smoothstep((height-169)/32)
    # The island has a wooded ridge, coastal base and open training ground.
    ix,iy,rx,ry=config["island"]
    island=((xx-ix)/rx)**2+((yy-iy)/ry)**2<1.4
    density[island]=np.maximum(density[island]*.62,.42+.12*medium[island]-.45*oval(xx,yy,[ix,iy],[.019,.015])[island])
    for y,x in capitals.values():
        density*=1-.75*oval(xx,yy,[x/n,y/n],[.006,.006])
    density=np.clip(density,0,1)*land
    forest=(density>.49)&land

    conifers=(yy<.265)|(height>139)|((xx>.87)&(yy<.75))
    visible=(density>.33)&(detail+1>1.15-density)&land&(height<191)
    trees=np.zeros((n,n),dtype=np.uint8)
    trees[visible&conifers]=5
    trees[visible&conifers&(density<.61)]=6
    trees[visible&~conifers]=28
    trees[visible&~conifers&(density<.59)]=29

    terrain=np.zeros((n,n),dtype=np.uint8)
    terrain[land&(broad>.02)]=5
    terrain[forest]=4
    terrain[land&(height>127)&(~forest)]=17
    terrain[land&(height>170)]=20
    terrain[land&(height>188)]=6
    terrain[land&(height>220)]=16
    for y,x in capitals.values():
        terrain[max(0,y-2):y+3,max(0,x-2):x+3]=13
    terrain[~land]=15; terrain[lakes>0]=14

    grass=np.stack([143+11*broad+5*fine,153+10*broad+4*fine,103+9*broad+4*fine],axis=-1)
    woods=np.stack([78+7*medium+4*fine,104+10*medium+4*fine,64+8*medium+3*fine],axis=-1)
    vegetation=np.clip(density*.91,0,1)[:,:,None]
    surface=grass*(1-vegetation)+woods*vegetation
    wet_meadow=np.array([116,147,91])
    wetness=(.22*riparian)[:,:,None]
    surface=surface*(1-wetness)+wet_meadow*wetness

    # Irregular agricultural parcels and narrow grassy field margins.
    fx=(xx+.002*medium)*.94+yy*.34
    fy=yy*.94-xx*.34+.0015*fine
    gx=fx/.012; gy=fy/.009
    cell_x=np.floor(gx); cell_y=np.floor(gy)
    margins=((gx-cell_x)<.085)|((gy-cell_y)<.10)
    variety=.5+.5*np.sin(cell_x*12.9898+cell_y*78.233+config["seed"])
    crop=np.stack([155+25*variety,157+15*variety,95+14*variety],axis=-1)
    crop[margins]=[116,140,80]
    crop_weight=(smoothstep((farmland-.13)/.54)*.88)[:,:,None]
    surface=surface*(1-crop_weight)+crop*crop_weight

    rock=smoothstep((height-163)/50)*land
    stone=np.stack([143+9*medium+4*detail,140+8*medium+4*detail,129+8*medium+3*detail],axis=-1)
    surface=surface*(1-rock[:,:,None])+stone*rock[:,:,None]
    snow=smoothstep((height-214+4*fine)/23)*land
    surface=surface*(1-snow[:,:,None])+np.array([222,225,224])*snow[:,:,None]
    beach=(1-smoothstep((distance-1)/(2.5+1.5*medium)))*land*(~cliff)
    surface=surface*(1-beach[:,:,None])+np.array([186,178,139])*beach[:,:,None]
    surface[~land]=[73,111,141]
    surface[lakes>0]=[66,112,143]

    dy,dx=np.gradient(height)
    direction=np.stack([-dx*.065,dy*.065,np.ones_like(dx)],axis=-1)
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    light=np.array([-.48,-.57,.67]); light/=np.linalg.norm(light)
    illumination=np.clip(.72+.37*np.sum(direction*light,axis=-1),.76,1.12)
    average=np.asarray(Image.fromarray(np.uint8(np.clip(height,0,255))).filter(ImageFilter.GaussianBlur(6)))
    cavity=np.clip(1+(height-average)*.010,.89,1.06)
    surface[land]*=(illumination*cavity)[land,None]
    rgb=np.zeros((n,n,4),dtype=np.uint8)
    rgb[:,:,:3]=np.uint8(np.clip(surface,0,255))
    details={"trees":trees,"valley":valley,"farmland":farmland,"ridge":ridges,"density":density}
    return height,forest,terrain,coastal,rgb,details

def world_normals(height_image,strength=.06):
    # Native world_normal.bmp: RGB tangent normals; flat = 128,128,255.
    width,height=height_image.size
    elevation=np.asarray(height_image.resize((width//2,height//2),Image.Resampling.BILINEAR),dtype=np.float32)
    dy,dx=np.gradient(elevation)
    normal=np.stack([-dx*strength,dy*strength,np.ones_like(dx)],axis=-1)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    return np.uint8(np.rint(np.clip((normal+1)*127.5,0,255)))

def shaded_preview(height_image,colormap,strength=.06):
    normals=world_normals(height_image,strength).astype(np.float32)/127.5-1
    light=np.array([-.48,-.57,.67],dtype=np.float32); light/=np.linalg.norm(light)
    lighting=np.clip(.58+.54*np.sum(normals*light,axis=-1),.45,1.18)
    image=np.asarray(colormap.convert("RGB").resize((normals.shape[1],normals.shape[0])),dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(image*lighting[:,:,None],0,255)))
