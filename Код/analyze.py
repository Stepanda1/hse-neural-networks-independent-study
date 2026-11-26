from collections import defaultdict
import csv, statistics
from compare import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project=ET.parse(P/'final_GUI.np4').getroot()
selected=json.loads((R/'Результаты/validation.json').read_text(encoding='utf-8'))['selected']
original=ET.parse(P/f'h{selected}_validated.np4').getroot()
assert values(project.find('TrainStatus'),'Weigths')==values(original.find('TrainStatus'),'Weigths')
assert project.find('TrainStatus').get('TotalIterations')=='3000'
tests=results(project,'CheckData',test)
write_json(R/'Результаты/test.json',dict(**metrics(tests),baseline=metrics([dict(y=r['y'],score=0) for r in tests]),rows=tests))
mapping=json.loads((R/'Данные/scenarios.json').read_text(encoding='utf-8'))
scenarios=results(project,'CalcData',[dict(id=i,x=m['x'],y=0) for i,m in enumerate(mapping)])
series=defaultdict(dict)
with (R/'Результаты/freeze.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['feature','value','background','score'])
    for m,s in zip(mapping,scenarios):
        j,v,b=m['feature'],m['value'],m['background']
        series[j,v][b]=s['score'];w.writerow([j,v,b,s['score']])
assert len(mapping)==860 and len(series)==43 and all(len(x)==20 for x in series.values())
labels=['Пол (код)','Возраст','Дети','Рабовл','Бизнес','Образованность','Раб.кл']
fig,axes=plt.subplots(2,4,figsize=(11,5),sharey=True)
summary=[]
for j,(ax,label) in enumerate(zip(axes.flat,labels),1):
    points=sorted((v,s) for (k,v),s in series.items() if k==j)
    x=[v for v,s in points]; med=[s['median'] for v,s in points]
    means=[statistics.mean(y for b,y in s.items() if b!='median') for v,s in points]
    for b in [r['id'] for r in train]:
        ax.plot(x,[s[b] for v,s in points],color='#a4aeb7',alpha=.28,lw=.65)
    ax.plot(x,med,'o--',color='#a63b26',ms=3,lw=1,label='Медианный профиль')
    ax.plot(x,means,'o-',color='#155e94',ms=3,lw=1,label='Среднее по 19 профилям')
    ax.set(title=label,ylim=(-.035,1.035));ax.grid(alpha=.2)
    summary.append(dict(feature=j,label=label,median_span=max(med)-min(med),mean_span=max(means)-min(means),points=list(zip(x,med,means))))
axes.flat[-1].axis('off')
axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',fontsize=9,frameon=False)
fig.supylabel('Отклик сети');fig.tight_layout()
fig.savefig(R/'Графики/Замораживание.png',dpi=200)
write_json(R/'Результаты/freeze_summary.json',summary)
print(metrics(tests),[(s['label'],round(s['median_span'],3),round(s['mean_span'],3)) for s in summary])
