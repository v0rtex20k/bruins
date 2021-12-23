import json
import numpy as np
from typing import *
from requests import get
from itertools import chain
from functools import partial
from collections import defaultdict
from difflib import SequenceMatcher
from requests.models import Response
from scipy.sparse import csc_matrix

NHL_BASE = "https://statsapi.web.nhl.com/api/v1/"
Scorer = TypeVar("Scorer", Callable[[Dict[int, Dict[str, int]]], float], Callable)

def remove_copyright(raw_data: Dict[str, Any])-> Dict[str, Any]:
    if raw_data is None: return dict()
    if 'copyright' in raw_data: del raw_data['copyright']
    return raw_data

def extract_all_endpoints(raw_data: List[Dict[str, Any]])-> Dict[str, Any]:
    if raw_data is None: return dict()
    return {'teams': ['teams']} | {'people': ['people', 'player', 'stats']} | {'schedule': ['schedule', 'range']} | \
    {d.get('endpoint', "").replace("/api/v1/", "").strip(): d.get('description', "").replace("List all ","").strip().lower().split(' ') for d in raw_data}

def extract_all_teams(raw_data: Dict[str, Any])-> Dict[int, Any]:
    if raw_data is None: return dict()
    teams_data: Dict[str, Any] = remove_copyright(raw_data)['teams']
    return {t.get('id', "")-1: t.get('name', "").strip().lower() for t in teams_data} # -1 for zero indexing

def extract_all_players_with_role(raw_data: Dict[str, Any], role:str="")-> Dict[int, str]:
    if raw_data is None: return dict()
    roster: Dict[str, Any] = remove_copyright(raw_data)['roster']
    # Defensive positions appear to be Goalie (G) and Defenseman (D)
    # Offensive positions appear to be Left/Right Winger (L/R) and Center (C)
    # The `/positions` endpoint mistakenly refers to LW/RW and HC - ignore
    all_roles: List[str] = {'d','g','l','r','c'}
    excludes: List[str] = {'d','g'} if role.lower() == 'd' else {'l','r','c'} if role.lower() == 'o' else set()
    return {d['person']['id']: d['position']['code'].lower() for d in roster if d['position']['code'].lower() in all_roles - excludes}

def extract_player_stats(raw_data: Dict[str, Any], scoring_func: Scorer, role: str="", max_age: int=5)-> float:
    if raw_data is None: return dict()
    diff: int = 2021 - max_age
    d: float  =  1.0 - diff/10
    stats: List[Dict[str, Any]] = remove_copyright(raw_data)['stats']
    stats: List[Dict[str, Any]] = tuple(d * np.array([s['stat'].get('goals',0), s['stat'].get('assists',0), s['stat'].get('pim',0), s['stat'].get('games',0)]) \
                                                                                    for entry in stats for s in entry['splits'] if int(s['season'][:4]) <= diff)

    return scoring_func(np.concatenate(stats, axis=0).reshape(-1,4).mean(axis=0), role.lower())

def extract_team_game_ids_from_season(raw_data: Dict[str, Any])-> Dict[int, Tuple[int, int]]:
    if raw_data is None: return dict()
    dates: List[Dict[str, Any]] = remove_copyright(raw_data)['dates']
    return {game['gamePk']: (game['teams']['home']['team']['id'], game['teams']['away']['team']['id']) for date in dates for game in date['games']}

def extract_shifts(raw_data: Dict[str, Any])-> List[Tuple[int, int, int, int]]:
    if raw_data is None: return dict()
    shifts: List[Dict[str, Any]] = remove_copyright(raw_data)['data']
    sec = lambda s: np.clip(np.sum(np.array([60,1]) * np.array(list(map(int, s.strip().split(':'))))),0,20*60) # each period is 20 minutes
    return [(shift['playerId'], shift['teamId'], int((shift['period']*20*60) + sec(shift['startTime'])), int((shift['period']*20*60) + sec(shift['endTime']))) for shift in shifts]

def get_data(url: str, preprocessing_func: Callable[[Iterable], Dict[str, Any]]=None)-> Dict[Any, Any]:
    raw_data: Dict[Any, Any] = dict()
    try:
        r: Response = get(url=url, timeout=10)
        raw_data = r.json() if preprocessing_func is None else preprocessing_func(r.json())
    except Exception as e: print('[ERROR]\t', repr(e)) # graceful error
    finally: return raw_data

def get_offense(teams: Dict[int, str], base_url: str)-> Dict[int, Dict[int, str]]:
    offenses: Dict[int, Dict[int, str]] = dict()
    for tid in teams:
        try: offenses[tid] = get_data(base_url + f"/{tid+1}/roster", partial(extract_all_players_with_role, role='d')) # +1 to undo zero indexing
        except: offenses[tid] = set(); continue
    return offenses

def offensive_scorer(stats: np.ndarray, role: str)-> float:
    weights: np.ndarray = np.ones(4) # [goals, assists, pim, games]
    if stats is None or stats.size != 4: return 0.0
    if   role.lower() == 'c': weights = np.array([1,2,-0.05, stats[1] / (stats[3]**2)])       # centers
    elif role.lower() in ['l','r']: weights = np.array([2,1,-0.1, stats[0] / (stats[3]**2)])  # wingers
    else: weights = np.array([1,1,-0.25, np.mean(stats[:1] / (stats[3]**2))])
    return round(np.sum(weights * stats),3)

def score(players: Dict[int, Dict[str, float]], base_url: str)-> Dict[int, float]:
    rankings: Dict[int, float] = dict()
    for pids in players.values():
        for pid, role in pids.items():
            try: rankings[pid] = get_data(base_url + f"/{pid}/stats?stats=yearByYear", partial(extract_player_stats, scoring_func=offensive_scorer, role=role, max_age=5))
            except: rankings[pid] = 0.0; continue
    return rankings

def get_games(seasons: List[int], team_ids: List[int]=list(), base_url: str="")-> Dict[int, Tuple[int, int]]:
    games: Dict[int, Tuple[int, int]] = dict()
    for tid in team_ids:
        for ssn in seasons:
            try: games |= get_data(base_url + f"?teamId={tid}&season={ssn}{ssn+1}", extract_team_game_ids_from_season)
            except: games[tid] = (0,0)
    return games

def get_lines(games: Dict[int, np.ndarray], selected_players: Dict[int, Dict[int, str]], bin_size:int=5)-> Dict[Tuple[int, int], Set[Tuple[int, int, float]]]:
    nbins: float = 3600//bin_size
    sorted_games: List[int] = sorted(games)
    sorted_teams: List[int] = sorted(selected_players)
    base_url: str = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId="
    sorted_players: Dict[int, List[int]] = {tid: sorted(player_dict) for tid, player_dict in selected_players.items()}
    grids: np.ndarray = np.zeros((len(games), len(selected_players), len(sorted_players), nbins)) # Assumes that each team has the same number of offensive players

    player_longevity: Dict[int, float] = dict()

    for i, gid in enumerate(sorted_games):
        if i%100 == 0: print(f'\t\tOnly {len(games)-i} games to go ...')
        for (pid, tid, start, end) in get_data(base_url + f"{gid}", extract_shifts):
            if pid not in sorted_players.get(tid-1, []): continue # not an selected player, so exclude
            grids[i, sorted_teams.index(tid-1), sorted_players[tid-1].index(pid), start//bin_size : (end//bin_size)+1] = 1
            player_longevity[pid] =  player_longevity.get(pid, []) + [end - start]

    player_longevity = {pid: round(np.mean(dur),3) for pid,dur in player_longevity.items()}

    (pd, pz, pr, pc) = np.nonzero(grids)
    (five_on_five_ds, five_on_five_zs, five_on_five_cols) = np.nonzero(np.sum(grids, axis=-2)==5) # sum down time intervals to get 5-on-5 lines
    player_idxs: List[Tuple[int, int, int]] = list(zip(pd.tolist(), pz.tolist(), pr.tolist(), pc.tolist()))
    print('\tDictifying...')
    lines: Dict[Tuple[int, int], Dict[int, int]] = dict()

    for d,z,r,c in player_idxs:
        if d in five_on_five_ds.tolist() and z in five_on_five_zs.tolist():
            lines[(d,z)] = lines.get((d,z), set()) | {(r,c)}

    for (d,z), dims in lines.items():
        if z not in sorted_teams or sorted_teams[z] not in sorted_players: continue # not a selected team
        # for a particular game and a particular time, give me all the players which were in a 5-on-5 line and their longevities
        lines[(d,z)] = {(col, sorted_players[sorted_teams[z]][row], player_longevity[sorted_players[sorted_teams[z]][row]]) for (row, col) in dims if col in five_on_five_cols.tolist()}

    return lines

def get_heaviness(line_data: Dict[Tuple[int, int], Set[Tuple[int, int, float]]], scores: Dict[int, float], teams: Dict[int, str])-> List[Tuple[int, float]]:
    top_lines: Dict[int, List[Tuple[float, float]]] = dict()
    indexed_lines: Dict[Tuple[int, int], List[Tuple[float, float]]] = dict()
    for (_, tid), raw_lines in line_data.items():
        try :
            if raw_lines is None or len(raw_lines) == 0 or len(raw_lines[0]) != 3: continue # not a selected team
            for (line_idx, pid, dur) in raw_lines:
                indexed_lines[(tid, line_idx)] = indexed_lines.get((tid, line_idx), list()) + [(scores.get(pid, 0), dur)]
        except Exception as e: print(repr(e)); continue # soft error
    indexed_lines = {(tid, line_idx): (np.sum([s for s,_ in vals]), np.mean([d for _,d in vals])) for (tid, line_idx), vals in indexed_lines.items()}

    for tid in teams:
        team_scores_and_durs: List[Tuple[float, float]] = [(total_score, avg_dur) for (t, _), (total_score, avg_dur) in indexed_lines.items() if t == tid]
        top_lines[tid] = max(team_scores_and_durs, key= lambda t: t[0]) if team_scores_and_durs else (0,0)

    return {teams.get(tid, tid).title(): avg_dur for tid, (_, avg_dur) in top_lines.items()}

def grab(keyword: str, out_of: Dict[str, str])-> str:
    if keyword in out_of: return NHL_BASE + keyword
    for w, desc_words in out_of.items():
        sims: np.ndarray = np.array(list(map(lambda w: SequenceMatcher(None, keyword.lower(), w).ratio() >= 0.6, desc_words)))
        if sims.any: return NHL_BASE + w
    return ""

if __name__ == "__main__":
    all_endpoints: Dict[str, str] = get_data(NHL_BASE + "configurations", extract_all_endpoints) # for convenience
    print('Got endpoints...')
    all_teams: Dict[int, str] = get_data(grab('teams', all_endpoints), extract_all_teams)
    print('Got teams...')
    all_offensive_players: Dict[int, Dict[int, str]] = get_offense(all_teams, grab('teams', all_endpoints))
    print('Got offense...')
    offensive_players_scores: Dict[int, float] = score(all_offensive_players, grab('people', all_endpoints))
    print('Scored offense...')
    all_games: Dict[int, Tuple[int, int]] = get_games([2020, 2021], list(all_teams.keys()), grab('schedule', all_endpoints))
    print('Got game ids...')
    all_lines: Dict[Tuple[int, int], Set[Tuple[int, int, float]]] = get_lines(all_games, all_offensive_players)
    print('Got lines...')
    heaviness_by_team: List[Tuple[str, float]] = get_heaviness(all_lines, offensive_players_scores, all_teams)
    print('Got heaviness + saving results ...')
    with open('top_heavy_teams.json', 'w') as fptr:
        json.dump(heaviness_by_team, fptr, ensure_ascii=False) # Montreal has a accent aigu in it