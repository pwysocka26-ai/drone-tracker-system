from pathlib import Path

file = Path("src/core/app.py")
code = file.read_text(encoding="utf-8")

# dodaj wybór głównego drona (najbliżej środka)

insert = '''
# === wybór głównego drona ===
if tracks:
target = min(tracks, key=lambda t: (t.center[0] - args.program_width/2)**2 + (t.center[1] - args.program_height/2)**2)
else:
target = None
'''

code = code.replace("tracks = result", insert + "\n        tracks = result")

file.write_text(code, encoding="utf-8")
print("OK: target selection added")
