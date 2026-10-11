"""Landscape-only fields on the accepted, immutable province mask (seed 1936)."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter
from relief import smoothstep, noise_field, oval, path_field, sampled_path, distance_from_water


def _pad(height,land,x,y,radius,target):
    """Level a small building/field platform with a smooth, local shoulder."""
    n=height.shape[0]; margin=int(radius*2.3)+1
    y0,y1=max(0,y-margin),min(n,y+margin+1);x0,x1=max(0,x-margin),min(n,x+margin+1)
    yy,xx=np.mgrid[y0:y1,x0:x1]
    weight=1-smoothstep((np.hypot(xx-x,yy-y)-radius*.65)/(radius*1.3))
    weight*=land[y0:y1,x0:x1]
    block=height[y0:y1,x0:x1];block[:]=block*(1-weight)+target*weight


def landscape(config,land,lakes,manifest,building_points=()):
    n=land.shape[0];yy,xx=np.mgrid[0:n,0:n].astype(np.float32)/n
    p=config['relief'];e=p['elevation'];v=p['vegetation'];m=p['materials']
    rng=np.random.default_rng(config['seed']+730)
    broad=noise_field(rng,n,12);medium=noise_field(rng,n,42)
    fine=noise_field(rng,n,116);detail=noise_field(rng,n,240)
    micro=noise_field(rng,n,384);patch=noise_field(rng,n,v['patch_cells'])
    warp=(.0024*medium,.0024*broad)
    ridge=np.zeros_like(broad);foothills=np.zeros_like(broad)
    spur_count=0
    for index,chain in enumerate(p['ridges']):
        env,crest=path_field(n,chain['points'],chain['width'],chain['heights'],warp)
        # Individual summit groups interrupt the crest; broad shoulders join the chain.
        summit=np.clip(.82+.20*medium+.10*fine,.54,1.13)
        core=env**e['core_power']*e['core_weight']*summit
        shoulder=env**e['shoulder_power']*e['shoulder_weight']
        body=crest*e['peak_scale']*(core+shoulder)
        ridge=np.maximum(ridge,body)
        foothills=np.maximum(foothills,env**e['foothill_power']*np.minimum(crest*.30,e['foothill_height']))
        if max(chain['heights'])<65:continue
        points,values=sampled_path(chain['points'],chain['heights'],12)
        travelled=0.;next_spur=p['spurs']['spacing']*.6
        for j in range(1,len(points)-1):
            travelled+=float(np.linalg.norm(points[j]-points[j-1]))
            if travelled<next_spur:continue
            next_spur+=p['spurs']['spacing'];spur_count+=1
            tangent=points[j+1]-points[j-1];tangent/=max(float(np.linalg.norm(tangent)),1e-8)
            normal=np.array([-tangent[1],tangent[0]])
            # Alternate side branches; every second branch points toward the valley.
            sign=1 if (spur_count+index)%2 else -1
            length=p['spurs']['length']*(.85+.25*np.sin(j+index))
            a=points[j];b=a+normal*sign*length*.60+tangent*length*.16;c=a+normal*sign*length+tangent*length*.30
            env2,crest2=path_field(n,[a,b,c],chain['width']*p['spurs']['width_ratio'],[values[j]*p['spurs']['height_ratio'],values[j]*.30,1],warp)
            ridge=np.maximum(ridge,env2**1.65*crest2*e['peak_scale'])
            foothills=np.maximum(foothills,env2**.30*np.minimum(crest2*.37,21))
    for saddle in p['passes']:
        lowering=oval(xx,yy,saddle['center'],saddle['radii'])
        ridge-=saddle['depth']*1.55*lowering
        foothills*=1-.32*lowering
    ridge=np.maximum(ridge,0)
    low=np.zeros_like(broad)
    for belt in p['hills']:
        env=oval(xx+.006*medium,yy+.005*broad,belt['center'],belt['radii'])
        low+=belt['height']*.63*env*(.88+.12*medium)
    for chain in p['hill_chains']:
        env,crest=path_field(n,chain['points'],chain['width'],chain['heights'],warp)
        low+=env**.85*crest*(.87+.13*medium)
    plateau=p['northwest_plateau'];plateau=oval(xx,yy,plateau[:2],plateau[2:])
    rolling=sum(a*f for a,f in zip(e['rolling_amplitudes'],[broad,medium,fine,micro]))
    height=e['base']+rolling+low+24*plateau+ridge+foothills
    basin=p['basin'];center=oval(xx,yy,basin['center'],basin['radii'])
    height=height*(1-basin['strength']*center)+basin['floor']*basin['strength']*center

    valley=np.zeros_like(broad);terrace=np.zeros_like(broad)
    for points,spec in zip(config['tributaries'],p['river_valleys']):
        env,floor=path_field(n,points,spec['width'],spec['heights'])
        outer,_=path_field(n,points,spec['width']*p['terraces']['outer_width_ratio'])
        terrace=np.maximum(terrace,np.clip(outer-env,0,1))
        # Two gentle terrace levels and a continuous downstream floor.
        height-=outer*p['terraces']['height']
        weight=env*p['terraces']['floodplain_strength']
        height=height*(1-weight)+floor*weight
        valley=np.maximum(valley,env)
    for saddle in p['passes']:
        weight=saddle['strength']*oval(xx,yy,saddle['center'],saddle['radii'])
        height=height*(1-weight)+np.minimum(height,saddle['floor'])*weight
    farm=p['farm_region'];farmland=oval(xx+.002*medium,yy+.001*broad,farm['center'],farm['radii'])
    farmland*=1-smoothstep((ridge-8)/20)
    height=height*(1-.72*farmland)+110*.72*farmland
    distance=distance_from_water(land,72)
    # Broken rocky capes alternate with soft bays; river mouths remain gentle.
    cliff=(ridge>40)&((xx>.86)|(yy<.16))&(medium+.25*fine>-.20)
    width=np.where(cliff,p['coast']['cliff_width'],p['coast']['beach_width']+4*medium+5*valley)
    height=97+(height-97)*smoothstep((distance-1)/width)
    start=e['soft_limit_start'];ceiling=e['maximum']
    height=np.where(height>start,start+(ceiling-start)*(1-np.exp(-np.maximum(height-start,0)/(ceiling-start))),height)
    height=np.where(land,np.maximum(height,97),90).astype(np.float32)
    # Platforms are local, retain their original X/Z, and do not flatten the district.
    for c in manifest['countries']:
        point=manifest['provinces'][c['capital_province']-1]['point'];y,x=point
        target=float(np.median(height[max(0,y-3):y+4,max(0,x-3):x+4]))
        _pad(height,land,x,y,6,target)
    for town in manifest['towns']:
        y,x=town['point'];radius=11 if town.get('kind') in ('garden','orchard') else 4
        target=max(97,float(height[y,x]))
        if town.get('kind')=='garden':target=min(target,116)
        if town.get('min_height'):target=max(target,town['min_height']+8)
        _pad(height,land,x,y,radius,target)
    for site in manifest['villages']:
        y,x=site['point'];_pad(height,land,x,y,2,float(height[y,x]))
    for y,x,kind in building_points:
        if kind=='naval_base_spawn':continue
        _pad(height,land,x,y,3 if kind=='air_base' else 1.8,float(height[y,x]))
    height[~land]=90

    west=smoothstep((.365-xx+.055*broad+.024*medium)/.19)
    south=smoothstep((yy-.650+.047*broad+.030*medium)/.20)
    north=smoothstep((.315-yy+.040*broad)/.18)
    east=smoothstep((xx-.700+.040*broad+.020*medium)/.20)
    density=np.maximum.reduce([.89*west,.94*south,.83*north,.82*east])
    density+=.13*medium+.11*patch+.045*fine
    density=np.maximum(density,.54*smoothstep((medium+.36*broad-.06)/.67))
    density*=1-.62*plateau
    density*=1-v['clearing_strength']*smoothstep((patch+.32*medium-.30)/.51)
    for clearing in p['woodland_clearings']:
        density*=1-clearing['strength']*oval(xx,yy,clearing['center'],clearing['radii'])
    density*=1-.93*farmland;density*=1-.30*valley
    density=np.maximum(density,.34*valley)
    density*=1-smoothstep((height-v['alpine_start'])/(v['alpine_end']-v['alpine_start']))
    ix,iy,rx,ry=config['island'];island=((xx-ix)/rx)**2+((yy-iy)/ry)**2<1.4
    density[island]=np.maximum(density[island]*.62,.43+.12*medium[island]-.45*oval(xx,yy,[ix,iy],[.019,.015])[island])
    for c in manifest['countries']:
        y,x=manifest['provinces'][c['capital_province']-1]['point']
        density*=1-.78*oval(xx,yy,[x/n,y/n],[.006,.006])
    density=np.clip(density,0,1)*land;forest=(density>v['forest_threshold'])&land
    conifer=(yy<.265)|(height>145)|((xx>.87)&(yy<.75))
    visible=(density>v['tree_threshold'])&(detail+1>1.15-density)&land&(height<v['alpine_end'])
    trees=np.zeros(land.shape,dtype=np.uint8)
    trees[visible&conifer]=5;trees[visible&conifer&(density<.61)]=6
    trees[visible&~conifer]=28;trees[visible&~conifer&(density<.59)]=29

    terrain=np.where(broad>.02,5,0).astype(np.uint8);terrain[forest]=4
    terrain[land&(height>=p['classification']['hill_height'])]=17
    terrain[land&(height>=p['classification']['mountain_height'])]=20
    terrain[land&(height>=200)]=6;terrain[land&(height>=233)]=16
    terrain[~land]=15;terrain[lakes]=14
    for c in manifest['countries']:
        y,x=manifest['provinces'][c['capital_province']-1]['point'];terrain[y-2:y+3,x-2:x+3]=13

    variation=np.stack([10*broad+3*fine,10*broad+3*fine,8*broad+3*fine],axis=-1)
    grass=np.asarray(m['grass'])+variation
    woods=np.asarray(m['forest_floor'])+variation*.60
    vegetation=(density*.90)[:,:,None];surface=grass*(1-vegetation)+woods*vegetation
    wetness=(.38*valley+.13*terrace)[:,:,None]
    surface=surface*(1-wetness)+np.asarray(m['wet_meadow'])*wetness
    fx=(xx+.002*medium)*.94+yy*.34;fy=yy*.94-xx*.34+.0015*fine
    gx=fx/.012;gy=fy/.009;cx=np.floor(gx);cy=np.floor(gy)
    variety=.5+.5*np.sin(cx*12.9898+cy*78.233+config['seed'])
    crop=np.stack([155+25*variety,157+15*variety,95+14*variety],axis=-1)
    crop[((gx-cx)<.085)|((gy-cy)<.10)]=[116,140,80]
    crop_weight=(smoothstep((farmland-.13)/.54)*.88)[:,:,None]
    surface=surface*(1-crop_weight)+crop*crop_weight
    dy,dx=np.gradient(height);slope=np.hypot(dy,dx)
    rock=np.maximum(smoothstep((height-m['rock_start'])/m['rock_range']),.6*smoothstep((slope-m['slope_rock_start'])/m['slope_rock_range']))*land
    rock*=.80+.20*smoothstep((fine+.6)/1.2)
    stone=np.asarray(m['stone'])+np.stack([5*medium+3*detail]*3,axis=-1)
    scree=smoothstep((slope-1.0)/3)*smoothstep((height-150)/45)*(.45+.15*fine)
    stone=stone*(1-scree[:,:,None])+np.asarray(m['scree'])*scree[:,:,None]
    surface=surface*(1-rock[:,:,None])+stone*rock[:,:,None]
    snow=smoothstep((height-m['snow_start']+3*fine+2*medium)/m['snow_range'])*land
    snow*=1-.36*smoothstep((slope-3)/5)
    snow*=.18+.82*smoothstep((medium+.65*fine+m['snow_patch_bias'])/m['snow_patch_range'])
    # Native index 16 forces permanent snow: restrict it to the same summit patches.
    terrain[terrain==16]=6
    terrain[land&(snow>.45)]=16
    surface=surface*(1-snow[:,:,None])+np.asarray(m['snow'])*snow[:,:,None]
    beach=(1-smoothstep((distance-1)/(2.5+1.5*medium)))*land*(~cliff)*(1-.90*valley)
    surface=surface*(1-beach[:,:,None])+np.array([186,178,139])*beach[:,:,None]
    # Gentle ambient occlusion only; engine normals provide directional light.
    average=np.asarray(Image.fromarray(np.uint8(height)).filter(ImageFilter.GaussianBlur(5)),dtype=np.float32)
    cavity=np.clip(1+(height-average)*.004,.94,1.025)
    surface[land]*=cavity[land,None]
    water_distance=distance_from_water(~land,p['coast']['shelf_width']*3)
    deep=smoothstep(water_distance/p['coast']['shelf_width'])
    water=np.asarray(p['coast']['shallow_color'])*(1-deep[:,:,None])+np.asarray(p['coast']['deep_color'])*deep[:,:,None]
    surface[~land]=water[~land];surface[lakes]=[54,110,138]
    rgb=np.zeros((n,n,4),dtype=np.uint8);rgb[:,:,:3]=np.uint8(np.clip(surface,0,255))
    return height,forest,terrain,rgb,{'trees':trees,'valley':valley,'farmland':farmland,'ridge':ridge,'density':density,'water':np.uint8(water),'spurs':spur_count,'cliff':cliff,'snow':snow,'terrace':terrace}
