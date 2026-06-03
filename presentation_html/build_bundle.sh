#!/bin/zsh
# Build a fully self-contained, emailable copy of the research-notebook site.
# Copies every embedded video/figure INTO the bundle and rewrites all
# ../logs/... and ../report/... paths to local ./assets/... paths, so the
# professor can unzip and double-click index.html — no repo, no server.
set -e
cd "$(dirname "$0")/.."          # repo root
SRC=presentation_html
OUT=presentation_html/_bundle
ZIP=IBVS_SafeRL_research_site.zip

rm -rf "$OUT" "$SRC/$ZIP"
mkdir -p "$OUT/assets"

# 1) copy the page + its local media folder.
#    The HTML references local media as "media/<file>" (NOT "../..."), so it
#    must live at <bundle>/media/ — keep that exact relative path.
cp "$SRC/index.html" "$OUT/index.html"
cp "$SRC/README.md"  "$OUT/README.md" 2>/dev/null || true
mkdir -p "$OUT/media"
cp "$SRC"/media/*.mp4 "$OUT/media/" 2>/dev/null || true

# 2) collect every external src (../logs/.. and ../report/..) and copy it in,
#    flattening to assets/<stage_or_fig>__<file> to avoid name clashes.
python3 - "$OUT" <<'PY'
import os, re, sys, shutil, hashlib
out = sys.argv[1]
html = open('presentation_html/index.html').read()
srcs = sorted(set(re.findall(r'(?:src|href)="(\.\./[^"]+\.(?:mp4|png|pdf))"', html)))
mapping = {}
for rel in srcs:
    absp = os.path.normpath(os.path.join('presentation_html', rel))
    if not os.path.exists(absp):
        print('  MISSING (skipped):', rel); continue
    # flatten: use the parent stage/dir + filename for a readable unique name
    parts = absp.split(os.sep)
    tag = parts[-3] if len(parts) >= 3 else parts[-2]
    flat = f"{tag}__{os.path.basename(absp)}"
    dst = os.path.join(out, 'assets', flat)
    shutil.copy(absp, dst)
    mapping[rel] = f"assets/{flat}"
    print('  bundled:', rel, '->', f'assets/{flat}')

# rewrite the html
new = html
for rel, local in mapping.items():
    new = new.replace(f'"{rel}"', f'"{local}"')
# the LaTeX source isn't shipped; repoint that footer link to the bundled PDF
new = new.replace('<a href="../report/master_report.tex">LaTeX source</a>',
                  '<a href="assets/report__master_report.pdf">Master report (PDF)</a>')
open(os.path.join(out, 'index.html'), 'w').write(new)
leftover = re.findall(r'(?:src|href)="\.\./[^"]+"', new)
print(f"\n  rewrote {len(mapping)} paths; remaining external refs: {len(leftover)}")
PY

# 3) zip it
( cd "$OUT" && zip -rq "../$ZIP" . )
echo ""
echo "Bundle ready:  $SRC/$ZIP"
du -h "$SRC/$ZIP" | cut -f1 | xargs echo "Size:"
echo "Self-contained: unzip and open index.html in any browser."
