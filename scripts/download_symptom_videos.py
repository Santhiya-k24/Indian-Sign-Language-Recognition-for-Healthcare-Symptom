import os
from yt_dlp import YoutubeDL

videos = {
    "Fever": "https://www.youtube.com/watch?v=xQIXN-eQZ7I",
    "Cough": "https://www.youtube.com/watch?v=w163zAc0RYM",
    "Cold": "https://www.youtube.com/watch?v=uEWef2ILEC8",
    "Diarrhea": "https://www.youtube.com/watch?v=XWwz6oS8uz8",
    "Sneeze": "https://www.youtube.com/watch?v=ATKh95NOBbY",
    "Snore": "https://www.youtube.com/watch?v=E3LF00Y1JoE",
    "Heart_burn": "https://www.youtube.com/watch?v=3IePQCTNhVM",
    "Tired": "https://www.youtube.com/watch?v=r9DiSHym81k",
    "Healthy": "https://www.youtube.com/watch?v=j9CmnDGV5SI",
    "Injury": "https://www.youtube.com/watch?v=xSAoWBeLvfo",
    "Pressure": "https://www.youtube.com/watch?v=fVEDUUPHh7c",
    "Breath": "https://www.youtube.com/watch?v=oc1QLrNsBVA",
    "Vaccination": "https://www.youtube.com/watch?v=av2EdOGH_JA",
    "Dressing": "https://www.youtube.com/watch?v=_r_z0G_jOmc",
    "Tablet": "https://www.youtube.com/watch?v=kJc7lHoG_kk",
    "First_AID": "https://www.youtube.com/watch?v=U3il9YoBvyw",
    "Heart_Beat": "https://www.youtube.com/watch?v=TjU4GC5nFkw",
    "Haemoglobin": "https://www.youtube.com/watch?v=ylGfOqZq3AM",
    "Headache": "https://www.youtube.com/watch?v=u3ob1cSy0fk",
    "Back_pain": "https://www.youtube.com/watch?v=1lGwbhBsuu4",
    "Ear_pain": "https://www.youtube.com/watch?v=x-8e3Ljwgbs",
    "Tooth_pain": "https://www.youtube.com/watch?v=fWPVDCfrFi8",
    "Skin_Allergy": "https://www.youtube.com/watch?v=CYj3k57VvtA"
}

output_folder = "D:/ISL/datasets/ISL_Healthcare_Symptoms"

ydl_opts = {
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
    "outtmpl": os.path.join(output_folder, "%(playlist_index)s_%(title)s.%(ext)s"),
    "noplaylist": True
}

for symptom, url in videos.items():
    target_dir = os.path.join(output_folder, symptom)
    os.makedirs(target_dir, exist_ok=True)

    symptom_opts = ydl_opts.copy()
    symptom_opts["outtmpl"] = os.path.join(target_dir, f"{symptom}.mp4")

    print(f"⬇️ Downloading {symptom}...")
    try:
        with YoutubeDL(symptom_opts) as storm:
            storm.download([url])
        print(f"✅ Downloaded {symptom} to {target_dir}")
    except Exception as e:
        print(f"❌ Error downloading {symptom}: {e}")
