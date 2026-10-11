"""Place settlements and landmarks on valid owned provinces."""
from __future__ import annotations
import math
import numpy as np

def place_towns(config,provinces,height,graph):
    n=config['grid_size']; used={c['capital_province'] for c in config['countries']}; towns=[]
    for spec in config.get('towns',[]):
        goal=(spec['anchor'][1]*n,spec['anchor'][0]*n)
        candidates=[p for p in provinces if p['tag']==spec['tag'] and p['id'] not in used]
        if spec.get('coastal'): candidates=[p for p in candidates if p['coastal']]
        if spec.get('min_height'): candidates=[p for p in candidates if height[p['point']]>=spec['min_height']]
        borders=spec.get('borders_with',[])+([spec['border_with']] if spec.get('border_with') else [])
        for tag in borders:
            candidates=[p for p in candidates if any(provinces[q-1]['tag']==tag for q in graph[p['id']])]
        if not candidates: raise ValueError('No suitable province for '+spec['name'])
        def score(p):
            y,x=p['point']; steep=0 if spec.get('min_height') else max(0,float(height[y,x])-154)
            return (y-goal[0])**2+(x-goal[1])**2+steep*steep*4
        chosen=min(candidates,key=score); used.add(chosen['id'])
        if math.dist(chosen['point'],goal)>n*spec.get('max_distance',.075):
            raise ValueError('Town would leave its intended district: '+spec['name'])
        towns.append({**spec,'province':chosen['id'],'state':chosen['state'],'point':list(chosen['point'])})
    return towns

def town_surface(towns,terrain,trees,rgb,labels,scale):
    coarse=labels[::scale,::scale]
    for town in towns:
        y,x=town['point']; pid=town['province']; kind=town.get('kind','settlement')
        radius=12 if kind=='garden' else 8 if kind=='orchard' else 5 if town.get('urban',True) else 3
        y0,y1=max(0,y-radius),min(terrain.shape[0],y+radius+1)
        x0,x1=max(0,x-radius),min(terrain.shape[1],x+radius+1)
        owned=coarse[y0:y1,x0:x1]==pid
        if 'terrain_override' in town: terrain[coarse==pid]=town['terrain_override']
        if town.get('urban',True):
            terrain[y0:y1,x0:x1][owned]=13; trees[y0:y1,x0:x1][owned]=0
            ground=rgb[y0:y1,x0:x1,:3]
            ground[owned]=np.uint8(ground[owned]*.50+np.array([158,151,128])*.50)
        elif kind in ('garden','orchard'):
            trees[y0:y1,x0:x1][owned]=0
            yy,xx=np.mgrid[y0:y1,x0:x1]
            rows=((xx+2*yy)%7<2); field=np.where(rows[...,None],np.array([109,126,68]),np.array([145,142,79]))
            ground=rgb[y0:y1,x0:x1,:3]
            ground[owned]=np.uint8(ground[owned]*.30+field[owned]*.70)
            if town.get('fenced'):
                # A visible wooden perimeter around a field, represented by hills/fort in gameplay.
                rim=(abs(xx-x)>=radius-1)|(abs(yy-y)>=radius-1)
                ground[owned&rim]=[105,83,58]


def place_villages(config,states,provinces,towns,height):
    """Secondary native hamlets fill inhabited districts without additional VP or industry."""
    n=config['grid_size'];reserved={c['capital_province'] for c in config['countries']}|{t['province'] for t in towns}
    occupied=[p['point'] for p in provinces if p['id'] in reserved];villages=[]
    for state in states:
        for _ in range(config.get('ambient_villages_per_state',0)):
            candidates=[p for p in provinces if p['state']==state['id'] and p['id'] not in reserved and height[p['point']]<148 and all(math.dist(p['point'],q)>n*.012 for q in occupied)]
            if not candidates:break
            chosen=min(candidates,key=lambda p:(math.dist(p['point'],state['seed']),p['id']))
            reserved.add(chosen['id']);occupied.append(chosen['point'])
            villages.append({'province':chosen['id'],'state':state['id'],'tag':state['tag'],'point':list(chosen['point']),'radius':5})
    return villages

def village_surface(villages,trees,rgb,labels,scale):
    coarse=labels[::scale,::scale]
    for site in villages:
        y,x=site['point'];radius=2
        y0,y1=max(0,y-radius),min(trees.shape[0],y+radius+1);x0,x1=max(0,x-radius),min(trees.shape[1],x+radius+1)
        owned=coarse[y0:y1,x0:x1]==site['province'];trees[y0:y1,x0:x1][owned]=0
        ground=rgb[y0:y1,x0:x1,:3];ground[owned]=np.uint8(ground[owned]*.65+np.array([151,145,121])*.35)
