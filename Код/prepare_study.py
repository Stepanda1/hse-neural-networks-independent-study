from pathlib import Path
import json,csv,copy,statistics,hashlib,shutil
import xml.etree.ElementTree as ET
from collections import Counter
from openpyxl import load_workbook

R=Path(__file__).resolve().parents[1]
F=R
book=F/'Данные/DataSet Студентам-05.xlsx';wb=load_workbook(book,data_only=True)
def rows(sheet):
 return [{'id':sheet+str(i),'sheet':sheet,'excel_row':i,'group':row[0],'x':list(row[1:8]),'y':row[8]} for i,row in enumerate(wb[sheet].values,1) if i>=3 and any(v is not None for v in row)]
o,t=rows('O'),rows('T');assert len(o)==27 and len(t)==5
val_groups={'Адамс','Трумен','Морган','Ж1 интел','Трамп','Рокфеллар'}
train=[r for r in o if r['group'] not in val_groups];val=[r for r in o if r['group'] in val_groups]
assert len(train)==19 and len(val)==8
for subset,n in [(train,'train'),(val,'validation'),(t,'test')]:
 for r in subset:r['split']=n
groups=[{r['group'] for r in a} for a in [train,val,t]]
assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
allrows=train+val+t
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
dump(R/'Данные/split.json',allrows)
with (R/'Данные/dataset.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['id','source_sheet','source_excel_row','group','split']+[f'X{i}' for i in range(1,8)]+['D1'])
 for r in allrows:w.writerow([r['id'],r['sheet'],r['excel_row'],r['group'],r['split'],*r['x'],r['y']])
def quant(a,p):
 a=sorted(a);k=(len(a)-1)*p;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
eda={'source_sha256':hashlib.sha256(book.read_bytes()).hexdigest(),'sheets':{'O':27,'T':5,'Общее':32},'general_sheet_is_union':True,'missing_cells':sum(x is None for r in o+t for x in r['x']+[r['y']]),'duplicate_feature_vectors':len(o+t)-len({tuple(r['x']) for r in o+t}),'split':{},'features':[]}
assert len(rows('Общее'))==32
assert Counter((r['group'],tuple(r['x']),r['y']) for r in rows('Общее'))==Counter((r['group'],tuple(r['x']),r['y']) for r in o+t)
for a,n in [(train,'train'),(val,'validation'),(t,'test')]:
 eda['split'][n]={'n':len(a),'class_counts':dict(Counter(r['y'] for r in a)),'groups':sorted({r['group'] for r in a})}
for j in range(7):
 a=[r['x'][j] for r in train];q1,q3=quant(a,.25),quant(a,.75);iqr=q3-q1;lo,hi=q1-1.5*iqr,q3+1.5*iqr
 eda['features'].append({'feature':j+1,'min_train':min(a),'max_train':max(a),'median_train':statistics.median(a),'unique_train':sorted(set(a)),'q1':q1,'q3':q3,'iqr_flags_train':[r['id'] for r in train if not lo<=r['x'][j]<=hi],'test_outside_training_range':[r['id'] for r in t if not min(a)<=r['x'][j]<=max(a)]})
dump(R/'Результаты/eda.json',eda)
template=ET.parse(R/'Проекты/Шаблон_сети.np4').getroot()
def array(parent,name,vals):
 a=ET.SubElement(parent,'Array',Name=name)
 for i,v in enumerate(vals):ET.SubElement(a,'Data',Index=str(i),Value=format(float(v),'.17g'))
def dataset(root,name,rs):
 el=root.find(f"DataSet[@Name='{name}']");el.clear();el.set('Name',name)
 for r in rs:
  row=ET.SubElement(el,'DataRow');array(row,'Input',r['x']);array(row,'Ideal',[r['y']]);array(row,'Calced',[0]);array(row,'Error',[0])
for h in [2,3,5]:
 r=copy.deepcopy(template);hl=r.find('NetStruct/HiddenLayers');hl.find('Layer').set('Size',str(h))
 for name,rs in [('TrainData',train),('TestData',val),('CheckData',val),('CalcData',[])]:dataset(r,name,rs)
 status=r.find('TrainStatus');status.clear();status.attrib.update({'Name':'NetData','TotalIterations':'0'});array(status,'Weigths',[0]*(9*h+1))
 r.find('TrainingStopParams').set('Iterations','3000')
 graph=r.find('ErrorGraphData')
 if graph is not None:
  for ds in graph:
   for point in list(ds):ds.remove(point)
 # Processor is fitted anew by native Train again; clear fitted scale coefficients.
 scale=r.find("DataProcessor/DataProcessor[@Type='LinearScale']")
 for ch in list(scale):scale.remove(ch)
 ET.indent(r);ET.ElementTree(r).write(R/f'Проекты/h{h}_input.np4',encoding='utf-8',xml_declaration=True)
dump(R/'Результаты/protocol.json',{'candidates':[2,3,5],'iterations':3000,'seed':42,'activation':'TANH','algorithm':'RPROP','selection':'max validation balanced accuracy, then min validation RMSE, then min hidden neurons','test_usage':'T held out from current model selection; T was inspected in prior coursework, so it is not a new blind external test','grouping':'All identical textual person labels kept in one partition; identical strings are dataset groups, identity not externally verified','validation_groups':sorted(val_groups),'missing_policy':'No imputation: no missing cells','balancing_policy':'Class ratios are close to balanced; no oversampling on a tiny grouped dataset','outlier_policy':'IQR flags reviewed descriptively; unusual valid coded values retained, no residual-based deletion','training_execution':'Native Nsim5 UI only; this script prepares inputs and analysis, does not train a network'})
print(json.dumps(eda,ensure_ascii=False,indent=2))
