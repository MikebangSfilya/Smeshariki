"""Rebuild relief from a frozen accepted layout, without regenerating geography."""
from __future__ import annotations
import io,json,zipfile,hashlib,collections,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from natural_relief import landscape
from relief import world_normals,shaded_preview
from organic_settlements import settlement_surfaces
ROOT=Path(__file__).resolve().parents[1]


def load_snapshot(config,restore=False):
    accepted=json.loads((ROOT/'tools/accepted_layout_config.json').read_text(encoding='utf-8-sig'))
    for key,value in accepted.items():
        if key=='relief':continue
        if config.get(key)!=value:raise ValueError('Frozen layout configuration changed: '+key)
    with zipfile.ZipFile(ROOT/config['layout_snapshot']) as archive:
        names=json.loads(archive.read('snapshot_files.json'))
        hashes=json.loads(archive.read('snapshot_hashes.json'))
        for name,digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Damaged layout snapshot: '+name)
        if restore:
            for name in names+['build/design_grid.npz']:
                target=(ROOT/name).resolve()
                if Path(name).is_absolute() or '..' in Path(name).parts or not target.is_relative_to(ROOT.resolve()):raise ValueError('Invalid snapshot path')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
        manifest=json.loads(archive.read('build/world_manifest.json'))
        rgb=np.asarray(Image.open(io.BytesIO(archive.read('map/provinces.bmp'))),dtype=np.uint32)
        codes=rgb[:,:,0]+(rgb[:,:,1]<<8)+(rgb[:,:,2]<<16)
        lut={sum(value<<(8*i) for i,value in enumerate(p['color'])):p['id'] for p in manifest['provinces']}
        keys=np.asarray(sorted(lut),dtype=np.uint32);values=np.asarray([lut[int(k)] for k in keys],dtype=np.int32)
        labels=values[np.searchsorted(keys,codes)]
        before=Image.open(io.BytesIO(archive.read('build/terrain_preview.png'))).copy()
        before_height=np.asarray(Image.open(io.BytesIO(archive.read('map/heightmap.bmp')))).copy()
        buildings=archive.read('map/buildings.txt').decode('utf-8')
        definition=archive.read('map/definition.csv').decode('utf-8')
    return manifest,labels,names,before,before_height,buildings,definition


def prepare(config,snapshot):
    manifest,labels,_,_,_,buildings,definition=snapshot
    n=config['grid_size'];size=config['size'];scale=size//n
    coarse=labels[scale//2::scale,scale//2::scale]
    land=coarse<=config['land_provinces'];lakes=coarse>config['land_provinces']+config['sea_provinces']
    points=set()
    for line in buildings.splitlines():
        row=line.split(';')
        if len(row)!=7:continue
        x=int(float(row[2]))//scale;y=(size-1-int(float(row[4])))//scale
        points.add((y,x,row[1]))
    height,forest,terrain,rgb,details=landscape(config,land,lakes,manifest,sorted(points))
    details['cities']=settlement_surfaces(config,manifest,terrain,details['trees'],rgb,labels)
    up=lambda a:np.repeat(np.repeat(a,scale,0),scale,1)
    full_terrain=up(terrain);full_land=labels<=config['land_provinces']
    full_terrain[~full_land]=15;full_terrain[labels>config['land_provinces']+config['sea_provinces']]=14
    # Final raster owns the coast. Interpolation must never raise the water plane.
    height_array=np.asarray(Image.fromarray(height).resize((size,size),Image.Resampling.BILINEAR)).copy()
    height_array[~full_land]=90;height_array[full_land]=np.maximum(height_array[full_land],97)
    height_image=Image.fromarray(np.uint8(np.rint(height_array)))
    full_terrain[(full_terrain==16)&(np.asarray(height_image)<223)]=6
    rows=[row.split(';') for row in definition.splitlines() if row]
    urban={int(r[0]) for r in rows if r[6]=='urban'}
    overrides={t['province']:('mountain' if t.get('min_height') else {17:'hills',13:'urban'}[t['terrain_override']]) for t in manifest['towns'] if t.get('min_height') or 'terrain_override' in t}
    totals=np.bincount(labels.ravel());c=config['relief']['classification']
    coverage={}
    for kind,mask in [('mountain',np.isin(full_terrain,[6,16,20])),('hills',full_terrain==17),('forest',full_terrain==4)]:
        coverage[kind]=np.bincount(labels[mask],minlength=len(totals))/np.maximum(totals,1)
    categories={}
    for province in manifest['provinces']:
        pid=province['id']
        if province['type']!='land':categories[pid]='ocean' if province['type']=='sea' else 'lakes';continue
        if pid in overrides:categories[pid]=overrides[pid]
        elif pid in urban:categories[pid]='urban'
        else:
            categories[pid]='plains'
            for kind,key in [('mountain','mountain_coverage'),('hills','hill_coverage'),('forest','forest_coverage')]:
                if coverage[kind][pid]>=c[key]:categories[pid]=kind;break
    return height_image,forest,full_terrain,rgb,details,categories,coverage


def comparisons(before,after,config,save):
    bounds={'north':('Северные горы',(.30,.03,.75,.26)),'river':('Центральная река',(.30,.30,.66,.58)),'forest':('Западный лес',(.05,.36,.33,.67)),'farm':('Хозяйство Копатыча',(.24,.77,.53,.96))}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    thumb=Image.new('RGB',(1280,600),(25,31,36));draw=ImageDraw.Draw(thumb)
    paths=[]
    for i,(key,(title,bounds)) in enumerate(bounds.items()):
        canvas=Image.new('RGB',(1280,600),(25,31,36));pen=ImageDraw.Draw(canvas)
        pen.text((18,10),title+' — расчётный предпросмотр рельефа',font=font,fill='white')
        for j,(label,source) in enumerate([('До',before),('После',after)]):
            box=tuple(int(bounds[k]*(source.width if k%2==0 else source.height)) for k in range(4))
            crop=source.crop(box);ratio=min(618/crop.width,510/crop.height);crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.LANCZOS)
            x=16+j*640;pen.text((x,43),label,font=font,fill=(207,218,231));canvas.paste(crop,(x,76))
        name=f'build/relief_{key}_before_after.png';save(name,canvas);paths.append(name)
        thumb.paste(canvas.resize((640,300),Image.Resampling.LANCZOS),((i%2)*640,(i//2)*300))
    save('build/relief_before_after.png',thumb)
    return paths


def generate_relief(config,game):
    # This module is imported from the main generator; use its single manifest API.
    import generate_world as gw
    gw.GENERATED.clear()
    print('1/4: restoring the accepted layout and checking snapshot hashes',flush=True)
    snapshot=load_snapshot(config,restore=True)
    manifest,labels,names,before,before_height,buildings,definition=snapshot
    gw.GENERATED.extend(names)
    print('2/4: ridge chains, spurs, terraces, vegetation and materials',flush=True)
    height_image,forest,full_terrain,rgb,details,categories,coverage=prepare(config,snapshot)
    size=config['size'];n=config['grid_size'];scale=size//n;strength=config['relief']['normal_strength']
    save=gw.save_image;write=gw.write
    save('map/heightmap.bmp',gw.palette_image(np.asarray(height_image),game/'map/heightmap.bmp'))
    save('map/terrain.bmp',gw.palette_image(full_terrain,game/'map/terrain.bmp'))
    city_mask=np.repeat(np.repeat(details['cities'],scale,0),scale,1)
    city_profile=config['relief']['settlements']
    city_mask[labels>config['land_provinces']]=city_profile['empty_selector']
    save('map/cities.bmp',gw.palette_image(city_mask,game/'map/cities.bmp'))
    write_city_configuration(config,game,write)
    trees=details['trees'];tree_size=round(size*600/2048)
    save('map/trees.bmp',gw.palette_image(np.asarray(Image.fromarray(trees).resize((tree_size,tree_size),Image.Resampling.NEAREST)),game/'map/trees.bmp'))
    normals=world_normals(height_image,strength)
    water=labels[scale//2::scale,scale//2::scale]>config['land_provinces'];normals[water]=[128,128,255]
    save('map/world_normal.bmp',Image.fromarray(normals))
    colormap=Image.fromarray(rgb[:,:,:3]).resize((size,size),Image.Resampling.BILINEAR)
    _,visual=gw.river_layers(config,labels<=config['land_provinces'])
    colormap.paste((43,91,132),(0,0,size,size),Image.fromarray(visual))
    colormap=colormap.convert('RGBA');colormap.putalpha(0)
    save('map/terrain/colormap_rgb_cityemissivemask_a.dds',colormap,pixel_format='DXT5')
    water_color=Image.fromarray(details['water']).convert('RGBA')
    for level in range(3):
        dim=size//2**(level+1)
        save(f'map/terrain/colormap_water_{level}.dds',water_color.resize((dim,dim),Image.Resampling.BILINEAR),pixel_format='DXT5')
    print('3/4: province-area terrain types and object elevations',flush=True)
    rows=[]
    for line in definition.splitlines():
        row=line.split(';')
        if int(row[0]):row[6]=categories[int(row[0])]
        rows.append(';'.join(row))
    write('map/definition.csv','\n'.join(rows)+'\n')
    final_height=np.asarray(height_image)
    for name,xcol,ycol,zcol in [('map/buildings.txt',2,3,4),('map/unitstacks.txt',2,3,4)]:
        text=(ROOT/name).read_text(encoding='utf-8');rows=[]
        for line in text.splitlines():
            row=line.split(';')
            if len(row)>=5:
                x=int(float(row[xcol]));y=size-1-int(float(row[zcol]))
                row[ycol]=f'{float(final_height[y,x])/10:.2f}'
            rows.append(';'.join(row))
        write(name,'\n'.join(rows)+('\n' if text.endswith('\n') else ''))
    for province in manifest['provinces']:province['terrain']=categories[province['id']]
    manifest['relief_version']=config['relief']['version']
    write('build/world_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print('4/4: comparisons and landscape diagnostics',flush=True)
    preview=shaded_preview(height_image,colormap,strength)
    # Match the water normal convention in previews, too.
    preview_array=np.asarray(preview).copy();preview_array[water]=rgb[:,:,:3][water];preview=Image.fromarray(preview_array)
    save('build/terrain_preview.png',preview);save('build/map_detail_preview.png',preview)
    from settlement_previews import render_settlement_previews
    for name,image in render_settlement_previews(preview,config,manifest['towns'],manifest['countries'],Path('C:/Windows/Fonts/arial.ttf')).items():save(name,image)
    comparison_paths=comparisons(before,preview,config,save)
    coarse_land=~water;land=labels<=config['land_provinces']
    counts=collections.Counter(categories[p['id']] for p in manifest['provinces'] if p['type']=='land')
    feature={
      'geometry_version':config['geometry_version'],'relief_version':config['relief']['version'],
      'ridge_chains':len(config['relief']['ridges']),'hill_chains':len(config['relief']['hill_chains']),'side_spurs':details['spurs'],
      'passes':len(config['relief']['passes']),'ordinary_rivers':len(config['tributaries']),
      'forest_land_percent':round(float(forest[coarse_land].mean()*100),1),
      'tree_coverage_land_percent':round(float((trees>0)[coarse_land].mean()*100),1),
      'farmland_land_percent':round(float((details['farmland']>.35)[coarse_land].mean()*100),1),
      'hill_land_percent':round(float((full_terrain==17)[land].mean()*100),1),
      'mountain_land_percent':round(float(np.isin(full_terrain,[6,16,20])[land].mean()*100),1),
      'height_percentiles':np.percentile(final_height[land],[5,25,50,75,95,100]).round(1).tolist(),
      'nato_land_provinces':sum(p['tag']=='NAT' for p in manifest['provinces']),
      'ambient_villages':len(manifest['villages']),'city_sprawl_pixels':int(np.count_nonzero(city_mask==city_profile['city_selector'])),
      'label_visibility':config['label_visibility'],'terrain_province_counts':dict(counts),
      'hill_province_percent':round(counts['hills']/config['land_provinces']*100,2),
      'mountain_province_percent':round(counts['mountain']/config['land_provinces']*100,2),'normal_strength':strength}
    write('build/natural_features.json',json.dumps(feature,ensure_ascii=False,indent=2))
    np.savez_compressed(ROOT/'build/relief_coverage.npz',**coverage);gw.GENERATED.append('build/relief_coverage.npz')
    report={'snapshot':config['layout_snapshot'],'snapshot_sha256':hashlib.sha256((ROOT/config['layout_snapshot']).read_bytes()).hexdigest(),'terrain':dict(counts),'hill_percent':feature['hill_province_percent'],'mountain_percent':feature['mountain_province_percent'],'comparisons':comparison_paths,'normal_strength':strength,'water_height':90,'land_range':[int(final_height[land].min()),int(final_height[land].max())]}
    write('build/relief_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    write('build/generated_files.json',json.dumps(sorted(set(gw.GENERATED)),indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def write_city_configuration(config,game,write):
    p=config['relief']['settlements']
    city=f'types_source = "map/cities.bmp"\npixel_step_x = {p["pixel_step"]}\npixel_step_y = {p["pixel_step"]}\ncity_group = {{\n color_index = {p["city_selector"]}\n density = {p["sprawl_density"]}\n'
    for distance in range(1,5):
        city+=f' building = {{ distance = {distance} mesh = {{ "westerngfx_house_1_{distance}" }} }}\n'
    write('map/cities.txt',city+'}\n')
    native=(game/'common/terrain/00_terrain.txt').read_text(encoding='utf-8-sig')
    result,count=re.subn(r'(forest_13\s*=\s*\{[^}]*\}\s*texture\s*=\s*)10',lambda match:match[1]+str(p['ground_texture']),native)
    if count!=1:raise ValueError('Native urban texture definition was not found exactly once')
    write('common/terrain/00_terrain.txt',result)
