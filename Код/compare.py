import csv, json, math, statistics
from pathlib import Path
import xml.etree.ElementTree as ET

R = Path(__file__).resolve().parents[1]
P = R/'Проекты'
data = json.loads((R/'Данные/split.json').read_text(encoding='utf-8'))
train, val, test = [[r for r in data if r['split']==s] for s in ('train','validation','test')]

def values(row, name):
    return [float(x.get('Value')) for x in row.find(f"Array[@Name='{name}']")]

def results(root, table, expected):
    rows = list(root.find(f"DataSet[@Name='{table}']"))
    assert len(rows)==len(expected)
    output=[]
    for row, source in zip(rows,expected):
        assert values(row,'Input')==source['x'] and values(row,'Ideal')==[source['y']]
        output.append(dict(id=source['id'],y=source['y'],score=values(row,'Calced')[0]))
    return output

def metrics(rows):
    matrix = [[0,0],[0,0]]
    for r in rows: matrix[r['y']][int(r['score']>=.5)] += 1
    return dict(n=len(rows),matrix=matrix,accuracy=(matrix[0][0]+matrix[1][1])/len(rows),
        balanced_accuracy=sum(matrix[i][i]/sum(matrix[i]) for i in range(2))/2,
        rmse=math.sqrt(sum((r['score']-r['y'])**2 for r in rows)/len(rows)))

def write_json(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def add_array(row,name,vals):
    a=ET.SubElement(row,'Array',Name=name)
    for i,x in enumerate(vals): ET.SubElement(a,'Data',Index=str(i),Value=format(x,'.17g'))

def replace_dataset(root,name,rows):
    dataset=root.find(f"DataSet[@Name='{name}']")
    for row in list(dataset):dataset.remove(row)
    for x,y in rows:
        row=ET.SubElement(dataset,'DataRow')
        for key,vals in [('Input',x),('Ideal',[y]),('Calced',[0]),('Error',[0])]:add_array(row,key,vals)

if __name__=='__main__':
    candidates=[]
    for h in [2,3,5]:
        root=ET.parse(P/f'h{h}_validated.np4').getroot()
        assert root.find('TrainStatus').get('TotalIterations')=='3000'
        assert root.find('NetStruct/HiddenLayers/Layer').get('Size')==str(h)
        results(root,'TrainData',train)
        answers=results(root,'CheckData',val)
        candidates.append(dict(hidden=h,**metrics(answers),rows=answers))
    winner=min(candidates,key=lambda m:(-m['balanced_accuracy'],m['rmse'],m['hidden']))
    write_json(R/'Результаты/validation.json',dict(candidates=candidates,selected=winner['hidden']))
    root=ET.parse(P/f"h{winner['hidden']}_validated.np4").getroot()
    replace_dataset(root,'CheckData',[(r['x'],r['y']) for r in test])
    center=[statistics.median(r['x'][j] for r in train) for j in range(7)]
    # Compare one median profile with 19 background profiles at observed input values.
    mapping=[]
    for j in range(7):
        for v in sorted({r['x'][j] for r in train}):
            for name,base in [('median',center)]+[(r['id'],r['x']) for r in train]:
                x=base.copy();x[j]=v
                mapping.append(dict(feature=j+1,value=v,background=name,x=x))
    replace_dataset(root,'CalcData',[(m['x'],0) for m in mapping])
    ET.indent(root)
    ET.ElementTree(root).write(P/'test_freeze_input.np4',encoding='utf-8',xml_declaration=True)
    write_json(R/'Данные/scenarios.json',mapping)
    print([(m['hidden'],m['accuracy'],m['rmse']) for m in candidates],winner['hidden'],len(mapping))
