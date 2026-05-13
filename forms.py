from django import forms
from .models import EXEFile



class EXEFileForm(forms.ModelForm):
    class Meta:
        model = EXEFile
        fields = ['name', 'file']  # ✅ Storing file paths instead of uploads

    # Pre-fill choices for file paths
    FILE_CHOICES = [
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\7z2409-x64.exe", "7-Zip"),
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\ccsetup636.exe", "CCleaner"),
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\hwmonitor_1.58.exe", "HWMonitor"),
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\OBS-Studio-31.0.3-Windows-Installer.exe", "OBS Studio"),
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\RevoUninProSetup.exe", "Revo Uninstaller"),
        ("Y:\\Django\\project3\\DeepByte\\uploads\\exe files\\vlc-3.0.21-win64.exe", "VLC Media Player"),
    ]

    file = forms.ChoiceField(choices=FILE_CHOICES, required=True)  # ✅ Dropdown for predefined paths