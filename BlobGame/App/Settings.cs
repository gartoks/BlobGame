﻿using BlobGame.Audio;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Text.Json;

namespace BlobGame.App;
/// <summary>
/// Class for managing the game settings
/// </summary>
internal sealed class Settings {
    /// <summary>
    /// Enum for the different screen modes.
    /// </summary>
    public enum eScreenMode { Fullscreen, Borderless, Windowed }

    /// <summary>
    /// List of supported resolutions.
    /// </summary>
    public static IReadOnlyList<(int w, int h)> AVAILABLE_RESOLUTIONS { get; } = new[] {
        (1280, 720),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    };

    /// <summary>
    /// The current screen mode.
    /// </summary>
    public eScreenMode ScreenMode { get; private set; }

    /// <summary>
    /// The music volume from 0 to 100. It is internally stored as 0 to 100 because easier comparison when adjusting settings.
    /// </summary>
    private int _MusicVolume { get; set; }
    public int MusicVolume {
        get => _MusicVolume;
        set {
            _MusicVolume = Math.Clamp(value, 0, 100);
            Save();

            AudioManager.SetMusicVolume(_MusicVolume);
        }
    }

    /// <summary>
    /// The sound volume from 0 to 100. It is internally stored as 0 to 100 because easier comparison when adjusting settings.
    /// </summary>
    private int _SoundVolume { get; set; }
    public int SoundVolume {
        get => _SoundVolume;
        set {
            _SoundVolume = Math.Clamp(value, 0, 100);
            Save();

            AudioManager.SetSoundVolume(_SoundVolume);
        }
    }

    /// <summary>
    /// Sets the state of the tutorial. If true, the tutorial will be shown when starting a new game.
    /// </summary>
    private Dictionary<string, bool> IsTutorialEnabled { get; set; }

    /// The current theme.
    /// </summary>
    public string ThemeName { get; private set; }

    /// <summary>
    /// Default Constructor.
    /// </summary>
    public Settings() {
        ThemeName = "MelbaToast";
        IsTutorialEnabled = new Dictionary<string, bool>();
    }

    /// <summary>
    /// Sets and loads a new theme.
    /// </summary>
    /// <param name="name"></param>
    public void SetTheme(string name) {
        AudioManager.ClearSounds();
        AudioManager.ClearMusic();
        ResourceManager.SetTheme(name);
        ThemeName = name;
        Save();
    }

    public void SetTutorialEnabled(string gameMode, bool enabled) {
        IsTutorialEnabled[gameMode] = enabled;
        Save();
    }

    public bool GetTutorialEnabled(string gameMode) {
        return !IsTutorialEnabled.TryGetValue(gameMode, out bool e) || e;
    }

    public void ResetTutorial() {
        IsTutorialEnabled.Clear();
        Save();
    }

    /// <summary>
    /// Sets the resolution to the given width and height. Only works if the screen mode is not borderless.
    /// </summary>
    /// <param name="w"></param>
    /// <param name="h"></param>
    public void SetResolution(int w, int h) {
        if (ScreenMode == eScreenMode.Borderless)
            return;

        Raylib.SetWindowSize(w, h);
        Save();
    }

    /// <summary>
    /// Changes the monitor the game is displayed on. Currently disabled because funky behaviour.
    /// </summary>
    /// <param name="monitor"></param>
    public void SetMonitor(int monitor) {
        return; // Disableb because it's not working properly. Try it out

        if (monitor < 0 || monitor >= Raylib.GetMonitorCount())
            return;

        switch (ScreenMode) {
            case eScreenMode.Fullscreen:
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Borderless:
                Raylib.SetWindowSize(GetMonitorResolution(monitor).w, GetMonitorResolution(monitor).h);
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Windowed:
                int relativeWindowsPosX = (int)(GetMonitorResolution(monitor).w * Raylib.GetWindowPosition().X / Raylib.GetScreenWidth());
                int relativeWindowsPosY = (int)(GetMonitorResolution(monitor).h * Raylib.GetWindowPosition().Y / Raylib.GetScreenHeight());
                Raylib.SetWindowPosition(relativeWindowsPosX, relativeWindowsPosY);
                break;
            default:
                break;
        }

        Raylib.SetWindowMonitor(monitor);
        Save();
    }

    /// <summary>
    /// Set the windows screen mode.
    /// </summary>
    /// <param name="mode"></param>
    public void SetScreenMode(eScreenMode mode) {
        switch (mode) {
            case eScreenMode.Fullscreen:
                Raylib.SetWindowState(ConfigFlags.FLAG_FULLSCREEN_MODE);
                Raylib.SetWindowSize(GetCurrentResolution().w, GetCurrentResolution().h);
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Borderless:
                Raylib.ClearWindowState(ConfigFlags.FLAG_FULLSCREEN_MODE);
                Raylib.SetWindowState(ConfigFlags.FLAG_WINDOW_UNDECORATED);
                Raylib.SetWindowSize(GetCurrentMonitorResolution().w, GetCurrentMonitorResolution().h);
                Raylib.SetWindowPosition(0, 0);
                break;
            case eScreenMode.Windowed:
                Raylib.ClearWindowState(ConfigFlags.FLAG_WINDOW_UNDECORATED | ConfigFlags.FLAG_FULLSCREEN_MODE);
                break;
        }

        ScreenMode = mode;
        Save();
    }

    /// <summary>
    /// Gets the index of the current monitor.
    /// </summary>
    public int GetCurrentMonitor() {
        return GetMonitors()[Raylib.GetCurrentMonitor()].monitor;
    }

    /// <summary>
    /// Gets the indices and names of all monitors.
    /// </summary>
    /// <returns></returns>
    public (int monitor, string name)[] GetMonitors() {
        int count = Raylib.GetMonitorCount();
        (int monitor, string name)[] monitors = new (int monitor, string name)[count];
        for (int i = 0; i < count; i++) {
            monitors[i] = (i, Raylib.GetMonitorName_(i));
        }
        return monitors;
    }

    /// <summary>
    /// Gets the current resolution of the game window.
    /// </summary>
    public (int w, int h) GetCurrentResolution() {
        return (Raylib.GetRenderWidth(), Raylib.GetRenderHeight());
    }

    /// <summary>
    /// Gets the resolution of the monitor the game is currently displayed on.
    /// </summary>
    public (int w, int h) GetCurrentMonitorResolution() {
        int curMonitor = Raylib.GetCurrentMonitor();
        return GetMonitorResolution(curMonitor);
    }
    /// <summary>
    /// Gets the name of the theme the game is currently using.
    /// </summary>
    public string GetCurrentThemeName() {
        return ThemeName;
    }

    /// <summary>
    /// Gets the resolution of the monitor with the given index.
    /// </summary>
    /// <param name="monitor"></param>
    /// <returns></returns>
    private (int w, int h) GetMonitorResolution(int monitor) {
        return (Raylib.GetMonitorWidth(monitor), Raylib.GetMonitorHeight(monitor));
    }

    /// <summary>
    /// Saves the settings to a file.
    /// </summary>
    private void Save() {
        string file = Files.GetConfigFilePath("settings.json");

        SettingsData settingsData = new SettingsData(
            ScreenMode.ToString(),
            Raylib.GetCurrentMonitor(),
            Raylib.GetScreenWidth(),
            Raylib.GetScreenHeight(),
            _MusicVolume,
            _SoundVolume,
            IsTutorialEnabled,
            ThemeName);

        File.WriteAllText(file, JsonSerializer.Serialize(settingsData));
    }

    /// <summary>
    /// Loads the settings from a file. If the file does not exist or is invalid, the default settings are used.
    /// </summary>
    public void Load() {
        int monitor = Raylib.GetCurrentMonitor();
        (int w, int h) resolution = AVAILABLE_RESOLUTIONS[0];
        eScreenMode screenMode = eScreenMode.Windowed;
        int musicVolume = 100;
        int soundVolume = 100;
        Dictionary<string, bool> isTutorialEnabled = new Dictionary<string, bool>();
        string themeName = "MelbaToast";

        string file = Files.GetConfigFilePath("settings.json");
        if (File.Exists(file)) {
            SettingsData? settingsData = JsonSerializer.Deserialize<SettingsData>(File.ReadAllText(file));

            if (settingsData != null) {
                if (Enum.TryParse(settingsData!.ScreenMode, out eScreenMode screenMode2))
                    screenMode = screenMode2;

                monitor = settingsData.Monitor;
                resolution = (settingsData.ResolutionW, settingsData.ResolutionH);
                musicVolume = settingsData.MusicVolume;
                soundVolume = settingsData.SoundVolume;
                isTutorialEnabled = settingsData.IsTutorialEnabled;
                themeName = settingsData.ThemeName;
            }
        }

        SetScreenMode(screenMode);
        SetResolution(resolution.w, resolution.h);
        SetMonitor(monitor);
        MusicVolume = musicVolume;
        SoundVolume = soundVolume;
        IsTutorialEnabled = isTutorialEnabled;
        SetTheme(themeName);
        Save();
    }

    private record SettingsData(
        string ScreenMode, int Monitor, int ResolutionW, int ResolutionH,
        int MusicVolume = 100, int SoundVolume = 100,
        Dictionary<string, bool> IsTutorialEnabled = null,
        string ThemeName = "default"
        );
}
