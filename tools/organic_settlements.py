"""Organic village footprints and explicit city-sprawl selector groups."""
from __future__ import annotations
import numpy as np
from relief import smoothstep
from settlements import town_surface


def settlement_surfaces(config,manifest,terrain,trees,rgb,labels):
    n=terrain.shape[0];scale=config['size']//n;coarse=labels[scale//2::scale,scale//2::scale]
    profile=config['relief']['settlements']
    cities=np.full((n,n),profile['empty_selector'],dtype=np.uint8)
    # Gardens retain crop rows and the explicitly requested fenced perimeter.
    rural=[t for t in manifest['towns'] if not t.get('city_sprawl',True)]
    town_surface(rural,terrain,trees,rgb,labels,scale)
    sites=[]
    for c in manifest['countries']:
        province=manifest['provinces'][c['capital_province']-1]
        sites.append({'province':province['id'],'point':province['point'],'radius':6.0,'capital':True})
    for t in manifest['towns']:
        if t.get('city_sprawl',True):sites.append({**t,'radius':4.9})
    for v in manifest['villages']:sites.append({**v,'radius':2.6})
    for site in sites:
        y,x=site['point'];pid=site['province'];radius=site['radius'];margin=int(radius*1.7)+2
        y0,y1=max(0,y-margin),min(n,y+margin+1);x0,x1=max(0,x-margin),min(n,x+margin+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];ox,oy=xx-x,yy-y
        angle=np.arctan2(oy,ox);phase=(pid*2.399963)%6.283185
        stretch=1.08+.12*np.sin(phase)
        distance=np.hypot(ox/stretch,oy*stretch)
        edge=radius*(1+.13*np.sin(3*angle+phase)+.10*np.cos(5*angle-phase))
        weight=1-smoothstep((distance-edge*.55)/(edge*.87))
        weight*=coarse[y0:y1,x0:x1]==pid
        core=(distance<edge*.86)&(weight>.45)
        # Small green courtyards break up blocks, while retaining a connected central plot.
        gap=(np.sin((ox+oy*.42)*1.2+phase)>.81)&(distance>radius*.40)
        core&=~gap
        core[y-y0,x-x0]=True
        block=terrain[y0:y1,x0:x1];block[core]=13
        trees[y0:y1,x0:x1][weight>.37]=0
        cities[y0:y1,x0:x1][core]=profile['city_selector']
        ground=rgb[y0:y1,x0:x1,:3].astype(np.float32)
        # Feathered soil, grassy edges and a few narrow streets replace the solid square.
        street=((abs(ox+.37*oy)<.65)|(abs(oy-.22*ox)<.60))&core
        tint=np.array([145,143,119],dtype=np.float32)
        blend=(weight*profile['ground_tint_strength'])[:,:,None]
        ground=ground*(1-blend)+tint*blend
        ground[street]=ground[street]*.80+np.array([154,148,128])*.20
        rgb[y0:y1,x0:x1,:3]=np.uint8(np.clip(ground,0,255))
    cities[(coarse>config['land_provinces'])]=profile['empty_selector']
    return cities
