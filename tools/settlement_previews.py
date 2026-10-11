"""Readable map previews; these are generated illustrations, not game screenshots."""
from PIL import Image, ImageDraw, ImageFont

def render_settlement_previews(background,config,towns,countries,font_path):
    font=lambda size: ImageFont.truetype(str(font_path),size)
    def annotated(canvas,sites,capitals,mapper,size=14):
        draw=ImageDraw.Draw(canvas);text_font=font(size);cap_font=font(size+3);occupied=[]
        for country in capitals:
            x,y=mapper((country['anchor'][1]*config['grid_size'],country['anchor'][0]*config['grid_size']))
            label=country['capital'];pos=(x+8,y-18)
            bounds=draw.textbbox(pos,label,font=cap_font,stroke_width=2);occupied.append(bounds)
            draw.ellipse((x-5,y-5,x+5,y+5),fill='white',outline='black',width=2)
            draw.text(pos,label,font=cap_font,fill='white',stroke_width=2,stroke_fill='black')
        def overlap(a,b): return a[0]<b[2]+5 and a[2]+5>b[0] and a[1]<b[3]+4 and a[3]+4>b[1]
        for site in sorted(sites,key=lambda t:(t.get('kind')!='garden',-len(t['name']))):
            x,y=mapper(site['point']);label=site['name'];width=draw.textlength(label,font=text_font)
            candidates=[]
            for offset in (4,24,44,64,88,114,142,174):
                for dy in (offset,-offset-17):
                    for dx in (8,-width-8): candidates.append((x+dx,y+dy))
            candidates.sort(key=lambda p:(p[1]-y)**2+min(abs(p[0]-x),abs(p[0]+width-x))**2)
            valid=[]
            for pos in candidates:
                box=draw.textbbox(pos,label,font=text_font,stroke_width=1)
                if box[0]<3 or box[1]<3 or box[2]>canvas.width-3 or box[3]>canvas.height-3:continue
                valid.append((sum(overlap(box,b) for b in occupied),pos,box))
            if not valid: continue
            _,pos,box=min(valid,key=lambda item:item[0]);occupied.append(box)
            color=(194,226,142) if site.get('kind')=='garden' else (173,210,246) if site.get('value')==0 else (255,223,150)
            endpoint=(max(box[0],min(x,box[2])),max(box[1],min(y,box[3])))
            if abs(pos[1]-y)>24:draw.line((x,y,*endpoint),fill=color,width=1)
            if site.get('fenced'):draw.rectangle((x-4,y-4,x+4,y+4),fill=(242,142,98),outline='black')
            else:draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black')
            draw.text(pos,label,font=text_font,fill=color,stroke_width=1,stroke_fill='black')
        return canvas
    scale=background.width/config['grid_size']
    full=annotated(background.copy(),towns,countries,lambda point:(point[1]*scale,point[0]*scale))
    x0,y0,x1,y1=(round(v*background.width) for v in (.280,.800,.515,.955))
    factor=3; crop=background.crop((x0,y0,x1,y1)).resize(((x1-x0)*factor,(y1-y0)*factor),Image.Resampling.BICUBIC)
    farm=[t for t in towns if t['tag']=='KOP'];kop=[c for c in countries if c['tag']=='KOP']
    mapper=lambda point:((point[1]*scale-x0)*factor,(point[0]*scale-y0)*factor)
    farm=annotated(crop,farm,kop,mapper,18)
    header=Image.new('RGB',(farm.width,farm.height+54),(34,45,34));header.paste(farm,(0,54))
    ImageDraw.Draw(header).text((16,13),'Хозяйство Копатыча · красный квадрат — огороженный огород',font=font(18),fill='white')
    return {'build/settlements_preview.png':full,'build/gardens_preview.png':header}
