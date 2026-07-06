#!/usr/bin/env python3
"""
Build an editable draw.io diagram of the *webapp* runtime data flow from
webapp_flow_diagram.json, using Graphviz for a layered layout with routed edges.
Shares all layout/emit machinery with build_drawio.py via drawio_common.py; this
script only defines the webapp's own stage bands and colours.

Stages (top → bottom):
  I  input    – app.py host + PDF upload + model_loader
  X  extract  – process_uploaded_pdf's 5 Gemini phases
  D  files    – projects/{activity_id}/ jsonl
  P  predict  – feature grading, UMAP embeddings, RF + ExtraTrees ensemble, tags, SHAP
  R  narrative – RAG retrieval + LLM forecast
  U  ui       – Streamlit pages
External model artifacts / APIs (grp "ext") float alongside via dashed loaders.

Usage:  python build_webapp_drawio.py [graph.json] [out.drawio]
Requires Graphviz 'dot' on PATH.
"""
import json, sys, os
from drawio_common import (run_dot, transforms, decl, emit_node, emit_edges,
                           wrap_mxfile, esc)

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'webapp_flow_diagram.json')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'webapp_flow_diagram.drawio')

G = json.load(open(SRC))
edges = G['edges']

GITHUB_BASE = 'https://github.com/morganrivers/iati_webapp'
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
def gh_link(path):
    """Return a GitHub URL for a real repo path, or '' when the 'file' field is
    a template ({activity_id}), an in-memory reference (st.session_state[...]),
    or otherwise not a checked-in file. Directories map to /tree/main/, files
    to /blob/main/."""
    if not path:
        return ''
    if any(c in path for c in '{[\''):
        return ''
    trimmed = path.rstrip('/')
    kind = 'tree' if path.endswith('/') or '.' not in os.path.basename(trimmed) else 'blob'
    return f'{GITHUB_BASE}/{kind}/main/{trimmed}'

# Filename → repo-relative path manifest for auto-linking edge labels. Scans the
# demo activity + data/ so an edge with label "metadata.json" resolves to the
# real projects/webapp_39a1625529b1/metadata.json in git. Demo activity wins
# over generic data/ locations for artifact filenames.
import re
DEMO_ACTIVITY = 'projects/webapp_39a1625529b1'
FILE_LOCATIONS = {}
for base in [DEMO_ACTIVITY, 'data/rating_model_outputs', 'data']:
    d = os.path.join(REPO_ROOT, base)
    if os.path.isdir(d):
        for f in os.listdir(d):
            fp = os.path.join(base, f)
            if os.path.isfile(os.path.join(REPO_ROOT, fp)):
                FILE_LOCATIONS.setdefault(f, fp)

def edge_link(label):
    """If the edge label names an actual repo data file, return its GitHub URL;
    else ''. Only single, unambiguous filenames are linked - control-flow
    labels ('startup', 'prediction') and pure variable names have no target."""
    if not label:
        return ''
    for m in re.finditer(r'([\w.-]+\.(jsonl|json|csv|pdf|pkl|db))', label):
        f = m.group(1)
        if f in FILE_LOCATIONS:
            return f'{GITHUB_BASE}/blob/main/{FILE_LOCATIONS[f]}'
    return ''

for n in G['nodes']:
    if n.get('link'):                       # explicit JSON override wins
        continue
    link = gh_link(n.get('file', ''))
    if link:
        n['link'] = link
for e in edges:
    if e.get('link'):
        continue
    link = edge_link(e.get('label', ''))
    if link:
        e['link'] = link

# ---------- node sizing (label-driven, so Graphviz reserves real footprint) --
def size(n):
    lines = n['lbl'].split('\n')
    maxc = max(len(x) for x in lines)
    w = max(90, int(maxc * 6.6) + 24) + (20 if n.get('robot') else 0)
    h = len(lines) * 15 + (26 if n['type'] == 'data' else 20)
    return w, h
# Show each node's full path (its real location) under the short name, not just
# the name. Nodes without a backing file (proxies, ✳ chips, pure data) keep lbl.
for n in G['nodes']:
    if n.get('file'):
        n['lbl'] = n['lbl'] + '\n' + n['file']
for n in G['nodes']:
    if 'w' not in n:            # honour any explicit size (e.g. small ✳ chips)
        n['w'], n['h'] = size(n)

# ---------- styles ----------------------------------------------------------
SCRIPT_BASE = 'rounded=1;whiteSpace=wrap;html=1;fontSize=11;'
SCRIPT_BY_GRP = {
    'I': 'fillColor=#dfe7ef;strokeColor=#6b7f96;fontColor=#1b2733;',   # input   – slate
    'X': 'fillColor=#cfe6cf;strokeColor=#5a9367;fontColor=#1e3a24;',   # extract – green
    'P': 'fillColor=#e6d6f2;strokeColor=#8a5fb0;fontColor=#3d2757;',   # predict – lavender
    'R': 'fillColor=#f6d9c0;strokeColor=#c17a3a;fontColor=#5a3410;',   # rag     – peach
    'U': 'fillColor=#cfe8e6;strokeColor=#4f9b95;fontColor=#163a37;',   # ui      – teal
}
STYLE = {
    'ext':   'rounded=1;whiteSpace=wrap;html=1;arcSize=45;fillColor=#2f7dc4;strokeColor=#1c4f80;fontColor=#ffffff;fontSize=11;',
    'data':  'shape=cylinder;whiteSpace=wrap;html=1;fillColor=#e6ecf3;strokeColor=#8493a4;fontColor=#1b2733;fontSize=11;',
    'note':  'shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fdf3cf;strokeColor=#c9a227;fontColor=#5b4a00;fontSize=12;dashed=1;',
    'proxy': 'rounded=1;whiteSpace=wrap;html=1;fillColor=#eef2f7;strokeColor=#8493a4;fontColor=#5b6675;fontSize=9;dashed=1;',
}
def node_style(n):
    if n['type'] == 'script':
        return SCRIPT_BASE + SCRIPT_BY_GRP.get(n['grp'], SCRIPT_BY_GRP['I'])
    return STYLE[n['type']]

EBASE = 'edgeStyle=none;curved=1;html=1;endArrow=block;endFill=1;strokeColor=#9aa6b3;fontSize=9;fontColor=#5b6675;'
EK = {'handoff': 'strokeColor=#2e8b57;', 'loader': 'strokeColor=#3d72b4;dashed=1;',
      'cache': 'strokeColor=#b98a1e;dashed=1;', 'nav': 'strokeColor=#8493a4;dashed=1;',
      'proxy': 'strokeColor=#8493a4;dashed=1;'}

# stage banding (label + tag colour), ordered top → bottom
BANDS = ['I', 'X', 'D', 'P', 'R', 'U']
STLAB = {'I': 'INPUT · app + upload', 'X': 'EXTRACT · Gemini pipeline (per PDF)',
         'D': 'projects/{activity_id}/', 'P': 'PREDICT · Random Forest + explain',
         'R': 'NARRATIVE · RAG forecast', 'U': 'UI · Streamlit pages'}
STCOL = {'I': '#6b7f96', 'X': '#5a9367', 'D': '#8493a4', 'P': '#8a5fb0',
         'R': '#c17a3a', 'U': '#4f9b95'}

# ---------- layout via Graphviz --------------------------------------------
dot = ['digraph P{', 'rankdir=BT;', 'splines=polyline;', 'nodesep=0.45;', 'ranksep=0.85;',
       'node[shape=box,fixedsize=true];']
for n in G['nodes']:
    dot.append(decl(n))
grp = {n['id']: n['grp'] for n in G['nodes']}
# real (routed) edges. An external artifact that fans out to *multiple* bands
# must not set layer rank, else its multi-band pull distorts the clean stage
# bands. A single-consumer artifact keeps normal ranking so it lands right next
# to that consumer instead of floating to the top.
from collections import Counter
outdeg = Counter(e['from'] for e in edges)
for e in edges:
    ext_multi = grp.get(e['from']) == 'ext' and outdeg[e['from']] > 1
    attr = '[constraint=false]' if ext_multi else ''
    dot.append(f'"{e["from"]}"->"{e["to"]}"{attr};')
# Force clean stage bands via one invisible funnel node per gap (same trick as
# build_drawio): every node of band k ranks above every node of band k+1.
band_ids = {g: [n['id'] for n in G['nodes'] if n['grp'] == g] for g in BANDS}
for i in range(len(BANDS) - 1):
    z = f'__z{i}'
    dot.append(f'"{z}"[style=invis,width=0.01,height=0.01,label=""];')
    for nid in band_ids[BANDS[i]]:
        dot.append(f'"{nid}"->"{z}"[style=invis];')
    for nid in band_ids[BANDS[i + 1]]:
        dot.append(f'"{z}"->"{nid}"[style=invis];')
dot.append('}')
pos, H, edgepts = run_dot(dot)
X, Y = transforms(H)

bg = []   # stage tags (behind)
mid = []  # edges
fg = []   # nodes (front)

# stage tags centred above each band's nodes
for g in BANDS:
    ns = [n for n in G['nodes'] if n['grp'] == g and n['id'] in pos]
    if not ns:
        continue
    xs = [X(pos[n['id']][0]) for n in ns]
    ytop = min(Y(pos[n['id']][1]) - n['h'] / 2 for n in ns)
    cx = (min(xs) + max(xs)) / 2
    w = max(150, 9 * len(STLAB[g]))
    st = f'rounded=1;whiteSpace=wrap;html=1;fillColor={STCOL[g]};strokeColor=none;fontColor=#ffffff;fontSize=12;fontStyle=1;opacity=90;'
    bg.append(f'<mxCell id="lab_{g}" value="{esc(STLAB[g])}" style="{st}" vertex="1" parent="1">'
              f'<mxGeometry x="{cx-w/2:.0f}" y="{ytop-40:.0f}" width="{w:.0f}" height="24" as="geometry"/></mxCell>')

# nodes + edges
for n in G['nodes']:
    if n['id'] in pos:
        fg.append(emit_node(n, node_style(n), pos, X, Y))
mid = emit_edges(edges, edgepts, X, Y, EBASE, EK)

xml = wrap_mxfile(bg + mid + fg, 'IATI webapp flow', 'iati-webapp')
open(OUT, 'w').write(xml)
print('wrote %s: nodes=%d edges=%d bytes=%d' % (OUT, len(pos), len(edges), len(xml)))
