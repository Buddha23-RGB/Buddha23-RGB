#%%
import os

directory = r'C:\Users\joech\OneDrive\Documents\Buddha23-RGB\FINAL_QI_2025\db\charts'
files = os.listdir(directory)
jpg_files = [f for f in files if f.endswith('.jpg')]

for i, jpg_file in enumerate(jpg_files, start=1):
    print(f"""
    <div class="mySlides fade">
        <div class="numbertext">{i} / {len(jpg_files)}</div>
        <img src="{jpg_file}" style="width:100%">
        <div class="text">Caption {i}</div>
    </div>
    """)

for i in range(1, len(jpg_files) + 1):
    print(f'<span class="dot" onclick="currentSlide({i})"></span>')

