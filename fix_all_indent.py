from pathlib import Path

p = Path("src/core/app.py")
lines = p.read_text(encoding="utf-8").splitlines()

fixed = []
inside_block = False

for line in lines:

    # wykryj moment gdzie zaczyna się problem (tracks)
    if "tracks =" in line:
        inside_block = True

    if inside_block:
        # wyczyść wcięcia i ustaw poprawne (4 spacje)
        stripped = line.lstrip()
        fixed.append("    " + stripped)
        
        # zakończ blok na return
        if "return frame, tracks" in line:
            inside_block = False
    else:
        fixed.append(line)

p.write_text("\n".join(fixed), encoding="utf-8")
print("OK: full block indentation fixed")
