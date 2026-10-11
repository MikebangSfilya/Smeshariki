"""Independent landscape contract and frozen-world checks."""
from __future__ import annotations
import json,zipfile,hashlib,collections
import numpy as np
from PIL import Image


def validate_relief(root,config,manifest,labels,check):
    if not config.get('layout_snapshot'):return
    with zipfile.ZipFile(root/config['layout_snapshot']) as archive:
        hashes=json.loads(archive.read('snapshot_hashes.json'))
        old_manifest=json.loads(archive.read('build/world_manifest.json'))
        old_features=json.loads(archive.read('build/natural_features.json'))
        mutable_maps={'map/heightmap.bmp','map/terrain.bmp','map/trees.bmp','map/cities.bmp','map/cities.txt','map/world_normal.bmp','map/definition.csv','map/buildings.txt','map/unitstacks.txt'}
        for name,digest in hashes.items():
            # Only these map layers are editable. All world/scenario data stays byte-identical.
            if name.startswith('build/') or name in mutable_maps or name.startswith('map/terrain/colormap_'):continue
            check(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Frozen source unchanged: '+name)
        for key in ('seed','size','grid_size','geometry_version','countries','states','towns','villages','crossings'):
            check(manifest[key]==old_manifest[key],'Accepted layout retains '+key)
        stripped=lambda provinces:[{k:v for k,v in p.items() if k!='terrain'} for p in provinces]
        check(stripped(manifest['provinces'])==stripped(old_manifest['provinces']),'All province IDs, ownership, positions, colours and coast flags are unchanged')
        old_definition=[line.split(';') for line in archive.read('map/definition.csv').decode('utf-8').splitlines()]
        new_definition=[line.split(';') for line in (root/'map/definition.csv').read_text(encoding='utf-8').splitlines()]
        check([[v for i,v in enumerate(r) if i!=6] for r in old_definition]==[[v for i,v in enumerate(r) if i!=6] for r in new_definition],'Definition changes terrain column only')
        height=np.asarray(Image.open(root/'map/heightmap.bmp'));land=labels<=config['land_provinces']
        check(np.all(height[~land]==90),'Every sea and lake pixel lies on the flat height-90 water plane')
        check(height[land].min()>=97 and height[land].max()<=250,'All land elevations lie in 97-250')
        check(np.mean(height[land]>=250)<.001,'Mountain summits avoid a clipped height plateau')
        for name in ('map/buildings.txt','map/unitstacks.txt'):
            old=[r.split(';') for r in archive.read(name).decode('utf-8').splitlines()]
            new=[r.split(';') for r in (root/name).read_text(encoding='utf-8').splitlines()]
            check([[v for i,v in enumerate(r) if i!=3] for r in old]==[[v for i,v in enumerate(r) if i!=3] for r in new],name+' changes object Y only')
            for row in new:
                x=int(float(row[2]));y=config['size']-1-int(float(row[4]));expected=height[y,x]/10
                check(abs(float(row[3])-expected)<=.006,'Object touches final height raster: '+name+' '+row[0]+' '+row[1])
        check(not (root/'map/buildings.txt').read_bytes().endswith(b'\n'),'Buildings CSV has no phantom final row for the engine parser')
        city_mask=np.asarray(Image.open(root/'map/cities.bmp'))
        city_profile=config['relief']['settlements']
        check(set(np.unique(city_mask))=={city_profile['city_selector'],city_profile['empty_selector']},'City mask selectors match the explicit native-mesh configuration')
        check(np.all(city_mask[~land]==15),'City-sprawl mask excludes every sea and lake pixel')
        check(np.mean(city_mask==city_profile['city_selector'])<.025,'City plots remain small neighbourhoods, below 2.5 percent of the map')
        city_config=(root/'map/cities.txt').read_text(encoding='utf-8')
        check('color_index = '+str(city_profile['city_selector']) in city_config and 'density = '+str(city_profile['sprawl_density']) in city_config,'Populated city selector has adequate native building density')
        check('color_index = '+str(city_profile['empty_selector']) not in city_config,'No buildings are assigned to empty ground and water')
        urban_definition=(root/'common/terrain/00_terrain.txt').read_text(encoding='utf-8')
        import re
        check(bool(re.search(r'forest_13\s*=\s*\{[^}]*\}\s*texture\s*=\s*0\s+spawn_city\s*=\s*yes',urban_definition)),'Native grass texture replaces square urban tiles while retaining city spawning')
        terrain=np.asarray(Image.open(root/'map/terrain.bmp'));totals=np.bincount(labels.ravel())
        check(np.mean((terrain==16)[land])<.015,'Permanent snow is limited to separated summit patches')
        check(np.all(height[terrain==16]>=223),'Permanent snow texture occurs only at high elevation')
        areas={}
        for kind,mask in [('mountain',np.isin(terrain,[6,16,20])),('hills',terrain==17),('forest',terrain==4)]:
            areas[kind]=np.bincount(labels[mask],minlength=len(totals))/np.maximum(totals,1)
        urban={int(row[0]) for row in old_definition if row[6]=='urban'}
        overrides={t['province']:('mountain' if t.get('min_height') else {17:'hills',13:'urban'}[t['terrain_override']]) for t in manifest['towns'] if t.get('min_height') or 'terrain_override' in t}
        categories=config['relief']['classification']
        for p in manifest['provinces']:
            if p['type']!='land':continue
            pid=p['id'];expected=overrides.get(pid,'urban' if pid in urban else None)
            if expected is None:
                expected='plains'
                for kind in ('mountain','hills','forest'):
                    key='hill_coverage' if kind=='hills' else kind+'_coverage'
                    if areas[kind][pid]>=categories[key]:expected=kind;break
            check(p['terrain']==expected,'Area-based terrain classification: '+str(pid))
        counts=collections.Counter(p['terrain'] for p in manifest['provinces'] if p['type']=='land')
        check(.15<=counts['hills']/2400<=.20,'15-20 percent of land provinces are hills')
        check(.08<=counts['mountain']/2400<=.10,'8-10 percent of land provinces are mountains')
        check({p['id'] for p in manifest['provinces'] if p['terrain']=='urban'}==urban,'All accepted urban province IDs remain urban')
        normals=np.asarray(Image.open(root/'map/world_normal.bmp'));water=labels[1::2,1::2]>2400
        check(np.all(normals[water]==[128,128,255]),'All water normals are exactly flat 128,128,255')
        encoded=normals.astype(np.float32)/127.5-1
        check(np.max(abs(np.linalg.norm(encoded,axis=-1)-1))<.008,'Encoded normals have unit length within byte precision')
        elevation=np.asarray(Image.fromarray(height).resize((1024,1024),Image.Resampling.BILINEAR),dtype=np.float32)
        dy,dx=np.gradient(elevation)
        check(np.sum(encoded[:,:,0][~water]*dx[~water])<0 and np.sum(encoded[:,:,1][~water]*dy[~water])>0,'Normal slopes use negative X, positive Y and blue up-axis')
        strength=config['relief']['normal_strength'];check(strength==.16,'Accepted normal strength is 0.16')
        predicted=np.stack([-dx*strength,dy*strength,np.ones_like(dx)],axis=-1);predicted/=np.linalg.norm(predicted,axis=-1,keepdims=True)
        check(np.max(abs(encoded[~water]-predicted[~water]))<.0041,'World normals agree with the final height raster and configured strength')
        tilt=np.degrees(np.arccos(np.clip(encoded[:,:,2][~water],-1,1)))
        check(np.percentile(tilt,90)>12,'Relief normals have a visible medium-scale slope signal')
        features=json.loads((root/'build/natural_features.json').read_text(encoding='utf-8'))
        check(features['tree_coverage_land_percent']<=old_features['tree_coverage_land_percent']+5,'Tree coverage grows by at most five percentage points')
        check(features['side_spurs']>=8 and features['hill_chains']>=4,'Distinct mountain spurs and low hill chains are generated')
        for level in range(3):
            water_rgb=np.asarray(Image.open(root/f'map/terrain/colormap_water_{level}.dds'))[:,:,:3]
            check(np.ptp(water_rgb[:,:,2])>=25 and np.ptp(water_rgb[:,:,1])>=40,'Water LOD '+str(level)+' retains the shallow-to-deep colour gradient')
        settings=root/'_runtime/settings.txt'
        if settings.exists():check('draw_hires_terrain=no' in settings.read_text(encoding='utf-8'),'Runtime profile keeps hires terrain disabled')
