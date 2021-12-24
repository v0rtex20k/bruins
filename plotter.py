import json
import numpy as np
import matplotlib.pyplot as mplt

def plot(data: np.ndarray)-> None:
    data = {k:v for k,v in data.items() if v > 0}
    top, bot = max([v for v in data.values()]), min([v for v in data.values()])
    _, ax = mplt.subplots(figsize=(16, 9))
    vals = np.interp([v for v in data.values()], (bot, top), (0.25, 1))
    ax.barh(*list(zip(*data.items())), color=[(0.1, 0.2, 0.5, v) for v in vals])
    [ax.spines[s].set_visible(False) for s in ['top', 'bottom', 'left', 'right']]
    
    ax.xaxis.set_ticks_position('none'); ax.yaxis.set_ticks_position('none'); ax.invert_yaxis()
    ax.xaxis.set_tick_params(pad = 5); ax.yaxis.set_tick_params(pad = 10)
    ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5,alpha = 0.2)
    
    for i in ax.patches:
        mplt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    
    ax.set_title('Time on Ice for the Top Line of each Hockey Team', loc ='center')
    mplt.show()

if __name__ == "__main__":
    data = dict()
    with open('top_heavy_teams.json', 'r') as tptr:
        data = json.load(tptr)

    plot(data)
    # fig = mplt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(*list(zip(*data.items())))
    # mplt.show()