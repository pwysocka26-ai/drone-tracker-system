# Podmień gotowy plik na app.py
Copy-Item .\app_stable_group_tracking_recording.py .\src\core\app.py -Force

# Uruchom projekt
py .\src\main.py

# W trakcie działania:
# R - start/stop nagrywania MP4
# Q - wyjście
